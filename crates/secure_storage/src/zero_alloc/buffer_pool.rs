//! # Buffer Pool Management
//!
//! High-performance buffer pool for reusing memory allocations
//! and minimizing heap fragmentation in non-critical paths.

use crate::error::{SecureStorageError, SecureStorageResult};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::debug;
use zeroize::Zeroize;

/// Buffer pool for reusable memory allocations
#[derive(Debug)]
pub struct BufferPool {
    /// Pool of available buffers
    buffers: Mutex<VecDeque<Vec<u8>>>,
    /// Buffer size for this pool
    buffer_size: usize,
    /// Maximum number of buffers to keep
    max_buffers: usize,
    /// Performance counters
    allocations: AtomicU64,
    deallocations: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl BufferPool {
    /// Create a new buffer pool
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(buffer_size: usize, max_buffers: usize) -> SecureStorageResult<Self> {
        if buffer_size == 0 {
            return Err(SecureStorageError::InvalidInput {
                field: "buffer_size".to_string(),
                reason: "Buffer size cannot be zero".to_string(),
            });
        }

        if max_buffers == 0 {
            return Err(SecureStorageError::InvalidInput {
                field: "max_buffers".to_string(),
                reason: "Max buffers cannot be zero".to_string(),
            });
        }

        Ok(Self {
            buffers: Mutex::new(VecDeque::with_capacity(max_buffers)),
            buffer_size,
            max_buffers,
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        })
    }

    /// Get a buffer from the pool
    ///
    /// # Errors
    ///
    /// Returns error if buffer allocation fails
    pub fn get_buffer(&self) -> SecureStorageResult<PooledBuffer<'_>> {
        self.allocations.fetch_add(1, Ordering::Relaxed);

        // Try to get buffer from pool
        {
            let mut buffers = self.buffers.lock();
            if let Some(mut buffer) = buffers.pop_front() {
                // Clear the buffer for security
                buffer.zeroize();
                buffer.resize(self.buffer_size, 0);

                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                debug!("Reused buffer from pool (size: {})", self.buffer_size);

                return Ok(PooledBuffer::new(buffer, self));
            }
        }

        // Pool is empty, allocate new buffer
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let buffer = vec![0u8; self.buffer_size];

        debug!("Allocated new buffer (size: {})", self.buffer_size);
        Ok(PooledBuffer::new(buffer, self))
    }

    /// Return a buffer to the pool
    fn return_buffer(&self, mut buffer: Vec<u8>) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);

        // Securely clear the buffer
        buffer.zeroize();

        // Try to return buffer to pool
        {
            let mut buffers = self.buffers.lock();
            if buffers.len() < self.max_buffers {
                buffers.push_back(buffer);
                drop(buffers);
                debug!("Returned buffer to pool");
            } else {
                drop(buffers);
                debug!("Pool full, dropping buffer");
            }
        }
    }

    /// Get pool statistics
    #[must_use]
    pub fn get_stats(&self) -> BufferPoolStats {
        let current_buffers = self.buffers.lock().len();

        BufferPoolStats {
            buffer_size: self.buffer_size,
            max_buffers: self.max_buffers,
            current_buffers,
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
        }
    }

    /// Clear all buffers from the pool
    pub fn clear(&self) {
        // Extract buffers to clear outside the lock
        let buffers_to_clear = {
            let mut buffers = self.buffers.lock();
            buffers.drain(..).collect::<Vec<_>>()
        };

        // Securely clear all buffers outside the lock
        for mut buffer in buffers_to_clear {
            buffer.zeroize();
        }
        debug!("Cleared all buffers from pool");
    }
}

/// RAII wrapper for pooled buffers
pub struct PooledBuffer<'a> {
    /// The actual buffer
    buffer: Option<Vec<u8>>,
    /// Reference to the pool for returning the buffer
    pool: &'a BufferPool,
}

impl<'a> PooledBuffer<'a> {
    /// Create a new pooled buffer
    const fn new(buffer: Vec<u8>, pool: &'a BufferPool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get mutable access to the buffer
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut().map_or(&mut [], Vec::as_mut_slice)
    }

    /// Get immutable access to the buffer
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().map_or(&[], Vec::as_slice)
    }

    /// Get the buffer size
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map_or(0, Vec::len)
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resize the buffer (within original capacity)
    ///
    /// # Errors
    ///
    /// Returns error if new size exceeds capacity
    pub fn resize(&mut self, new_size: usize) -> SecureStorageResult<()> {
        if let Some(ref mut buffer) = self.buffer {
            if new_size > buffer.capacity() {
                return Err(SecureStorageError::InvalidInput {
                    field: "new_size".to_string(),
                    reason: format!(
                        "New size {} exceeds capacity {}",
                        new_size,
                        buffer.capacity()
                    ),
                });
            }
            buffer.resize(new_size, 0);
        }
        Ok(())
    }

    /// Fill buffer with zeros
    pub fn zero(&mut self) {
        if let Some(ref mut buffer) = self.buffer {
            buffer.zeroize();
        }
    }
}

impl Drop for PooledBuffer<'_> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_buffer(buffer);
        }
    }
}

/// Buffer pool statistics
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    /// Size of each buffer in the pool
    pub buffer_size: usize,
    /// Maximum number of buffers
    pub max_buffers: usize,
    /// Current number of buffers in pool
    pub current_buffers: usize,
    /// Total allocations
    pub allocations: u64,
    /// Total deallocations
    pub deallocations: u64,
    /// Cache hits (reused buffers)
    pub cache_hits: u64,
    /// Cache misses (new allocations)
    pub cache_misses: u64,
}

impl BufferPoolStats {
    /// Calculate cache hit ratio
    #[must_use]
    pub fn cache_hit_ratio(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            f64::from(u32::try_from(self.cache_hits).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total_requests).unwrap_or(u32::MAX))
        }
    }

    /// Calculate pool utilization
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.max_buffers == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            f64::from(u32::try_from(self.current_buffers).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(self.max_buffers).unwrap_or(u32::MAX))
        }
    }
}

/// Global buffer pools for common sizes
pub struct GlobalBufferPools {
    /// Small buffers (1KB)
    pub small: BufferPool,
    /// Medium buffers (4KB)
    pub medium: BufferPool,
    /// Large buffers (16KB)
    pub large: BufferPool,
    /// Extra large buffers (64KB)
    pub extra_large: BufferPool,
}

impl GlobalBufferPools {
    /// Create global buffer pools
    ///
    /// # Errors
    ///
    /// Returns error if pool creation fails
    pub fn new() -> SecureStorageResult<Self> {
        Ok(Self {
            small: BufferPool::new(1024, 100)?,       // 1KB x 100 = 100KB
            medium: BufferPool::new(4096, 50)?,       // 4KB x 50 = 200KB
            large: BufferPool::new(16384, 20)?,       // 16KB x 20 = 320KB
            extra_large: BufferPool::new(65536, 10)?, // 64KB x 10 = 640KB
        })
    }

    /// Get appropriate buffer pool for size
    #[must_use]
    pub const fn get_pool_for_size(&self, size: usize) -> &BufferPool {
        if size <= 1024 {
            &self.small
        } else if size <= 4096 {
            &self.medium
        } else if size <= 16384 {
            &self.large
        } else {
            &self.extra_large
        }
    }

    /// Get combined statistics
    #[must_use]
    pub fn get_combined_stats(&self) -> CombinedPoolStats {
        CombinedPoolStats {
            small: self.small.get_stats(),
            medium: self.medium.get_stats(),
            large: self.large.get_stats(),
            extra_large: self.extra_large.get_stats(),
        }
    }

    /// Clear all pools
    pub fn clear_all(&self) {
        self.small.clear();
        self.medium.clear();
        self.large.clear();
        self.extra_large.clear();
    }
}

/// Combined statistics for all buffer pools
#[derive(Debug, Clone)]
pub struct CombinedPoolStats {
    /// Small buffer pool stats
    pub small: BufferPoolStats,
    /// Medium buffer pool stats
    pub medium: BufferPoolStats,
    /// Large buffer pool stats
    pub large: BufferPoolStats,
    /// Extra large buffer pool stats
    pub extra_large: BufferPoolStats,
}

impl CombinedPoolStats {
    /// Calculate total memory usage
    #[must_use]
    pub const fn total_memory_usage(&self) -> usize {
        (self.small.current_buffers * self.small.buffer_size)
            + (self.medium.current_buffers * self.medium.buffer_size)
            + (self.large.current_buffers * self.large.buffer_size)
            + (self.extra_large.current_buffers * self.extra_large.buffer_size)
    }

    /// Calculate total allocations
    #[must_use]
    pub const fn total_allocations(&self) -> u64 {
        self.small.allocations
            + self.medium.allocations
            + self.large.allocations
            + self.extra_large.allocations
    }

    /// Calculate overall cache hit ratio
    #[must_use]
    pub fn overall_cache_hit_ratio(&self) -> f64 {
        let total_hits = self.small.cache_hits
            + self.medium.cache_hits
            + self.large.cache_hits
            + self.extra_large.cache_hits;
        let total_misses = self.small.cache_misses
            + self.medium.cache_misses
            + self.large.cache_misses
            + self.extra_large.cache_misses;
        let total_requests = total_hits + total_misses;

        if total_requests == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            f64::from(u32::try_from(total_hits).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total_requests).unwrap_or(u32::MAX))
        }
    }
}
