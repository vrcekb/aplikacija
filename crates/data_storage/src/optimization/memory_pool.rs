//! Ultra-High Performance Memory Pool for Financial Applications
//!
//! This module provides memory pools optimized for `TallyIO`'s <1ms latency requirements.
//! All operations are designed for zero-allocation hot paths in financial trading systems.

use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Buffer size categories for optimal memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSize {
    Small,  // 1KB
    Medium, // 64KB
    Large,  // 1MB
}

impl BufferSize {
    /// Get buffer size from byte count
    #[must_use]
    pub const fn from_bytes(size: usize) -> Self {
        if size <= 1024 {
            Self::Small
        } else if size <= 64 * 1024 {
            Self::Medium
        } else {
            Self::Large
        }
    }

    /// Get actual buffer size in bytes
    #[must_use]
    pub const fn buffer_size(self) -> usize {
        match self {
            Self::Small => 1024,
            Self::Medium => 64 * 1024,
            Self::Large => 1024 * 1024,
        }
    }
}

/// High-performance pooled buffer
#[derive(Debug)]
pub struct PooledBuffer {
    data: Vec<u8>,
    size_category: BufferSize,
}

impl PooledBuffer {
    /// Create a new pooled buffer
    #[must_use]
    pub fn new(size_category: BufferSize) -> Self {
        let capacity = size_category.buffer_size();
        let data = vec![0; capacity];

        Self {
            data,
            size_category,
        }
    }

    /// Get mutable slice of buffer data
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get immutable slice of buffer data
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get buffer capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Reset buffer contents to zero
    pub fn reset(&mut self) {
        self.data.fill(0);
    }
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub small_buffer_pool_size: usize,
    pub medium_buffer_pool_size: usize,
    pub large_buffer_pool_size: usize,
    pub arena_chunk_size: usize,
    pub max_arenas: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            small_buffer_pool_size: 1000,
            medium_buffer_pool_size: 500,
            large_buffer_pool_size: 100,
            arena_chunk_size: 64 * 1024 * 1024, // 64MB
            max_arenas: 16,
        }
    }
}

/// Ultra-high performance memory pool
#[derive(Debug)]
pub struct MemoryPool {
    // Lock-free buffer pools
    small_buffers: SegQueue<PooledBuffer>,
    medium_buffers: SegQueue<PooledBuffer>,
    large_buffers: SegQueue<PooledBuffer>,

    // Performance metrics
    small_buffer_hits: AtomicUsize,
    small_buffer_misses: AtomicUsize,
    medium_buffer_hits: AtomicUsize,
    medium_buffer_misses: AtomicUsize,
    large_buffer_hits: AtomicUsize,
    large_buffer_misses: AtomicUsize,

    // Configuration
    config: MemoryPoolConfig,
}

impl MemoryPool {
    /// Create a new memory pool
    #[must_use]
    pub fn new(config: MemoryPoolConfig) -> Self {
        let pool = Self {
            small_buffers: SegQueue::new(),
            medium_buffers: SegQueue::new(),
            large_buffers: SegQueue::new(),
            small_buffer_hits: AtomicUsize::new(0),
            small_buffer_misses: AtomicUsize::new(0),
            medium_buffer_hits: AtomicUsize::new(0),
            medium_buffer_misses: AtomicUsize::new(0),
            large_buffer_hits: AtomicUsize::new(0),
            large_buffer_misses: AtomicUsize::new(0),
            config,
        };

        // Pre-populate pools
        pool.populate_pools();
        pool
    }

    /// Pre-populate buffer pools for optimal performance
    fn populate_pools(&self) {
        // Pre-allocate small buffers
        for _ in 0..self.config.small_buffer_pool_size {
            self.small_buffers
                .push(PooledBuffer::new(BufferSize::Small));
        }

        // Pre-allocate medium buffers
        for _ in 0..self.config.medium_buffer_pool_size {
            self.medium_buffers
                .push(PooledBuffer::new(BufferSize::Medium));
        }

        // Pre-allocate large buffers
        for _ in 0..self.config.large_buffer_pool_size {
            self.large_buffers
                .push(PooledBuffer::new(BufferSize::Large));
        }
    }

    /// Get a buffer from the pool (zero-allocation if available)
    #[must_use]
    pub fn get_buffer(&self, size: usize) -> PooledBuffer {
        let size_category = BufferSize::from_bytes(size);

        match size_category {
            BufferSize::Small => self.small_buffers.pop().map_or_else(
                || {
                    self.small_buffer_misses.fetch_add(1, Ordering::Relaxed);
                    PooledBuffer::new(BufferSize::Small)
                },
                |mut buffer| {
                    self.small_buffer_hits.fetch_add(1, Ordering::Relaxed);
                    buffer.reset();
                    buffer
                },
            ),
            BufferSize::Medium => self.medium_buffers.pop().map_or_else(
                || {
                    self.medium_buffer_misses.fetch_add(1, Ordering::Relaxed);
                    PooledBuffer::new(BufferSize::Medium)
                },
                |mut buffer| {
                    self.medium_buffer_hits.fetch_add(1, Ordering::Relaxed);
                    buffer.reset();
                    buffer
                },
            ),
            BufferSize::Large => self.large_buffers.pop().map_or_else(
                || {
                    self.large_buffer_misses.fetch_add(1, Ordering::Relaxed);
                    PooledBuffer::new(BufferSize::Large)
                },
                |mut buffer| {
                    self.large_buffer_hits.fetch_add(1, Ordering::Relaxed);
                    buffer.reset();
                    buffer
                },
            ),
        }
    }

    /// Return a buffer to the pool
    pub fn return_buffer(&self, buffer: PooledBuffer) {
        match buffer.size_category {
            BufferSize::Small => self.small_buffers.push(buffer),
            BufferSize::Medium => self.medium_buffers.push(buffer),
            BufferSize::Large => self.large_buffers.push(buffer),
        }
    }

    /// Get memory pool statistics
    #[must_use]
    pub fn stats(&self) -> MemoryPoolStats {
        let small_hits = self.small_buffer_hits.load(Ordering::Relaxed);
        let small_misses = self.small_buffer_misses.load(Ordering::Relaxed);
        let small_total = small_hits + small_misses;

        let medium_hits = self.medium_buffer_hits.load(Ordering::Relaxed);
        let medium_misses = self.medium_buffer_misses.load(Ordering::Relaxed);
        let medium_total = medium_hits + medium_misses;

        let large_hits = self.large_buffer_hits.load(Ordering::Relaxed);
        let large_misses = self.large_buffer_misses.load(Ordering::Relaxed);
        let large_total = large_hits + large_misses;

        MemoryPoolStats {
            small_buffer_hit_rate: if small_total > 0 {
                f64::from(u32::try_from(small_hits).unwrap_or(u32::MAX))
                    / f64::from(u32::try_from(small_total).unwrap_or(u32::MAX))
            } else {
                0.0_f64
            },
            medium_buffer_hit_rate: if medium_total > 0 {
                f64::from(u32::try_from(medium_hits).unwrap_or(u32::MAX))
                    / f64::from(u32::try_from(medium_total).unwrap_or(u32::MAX))
            } else {
                0.0_f64
            },
            large_buffer_hit_rate: if large_total > 0 {
                f64::from(u32::try_from(large_hits).unwrap_or(u32::MAX))
                    / f64::from(u32::try_from(large_total).unwrap_or(u32::MAX))
            } else {
                0.0_f64
            },
        }
    }
}

/// Memory pool performance statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub small_buffer_hit_rate: f64,
    pub medium_buffer_hit_rate: f64,
    pub large_buffer_hit_rate: f64,
}

/// Global memory pool instance
static GLOBAL_MEMORY_POOL: std::sync::OnceLock<Arc<MemoryPool>> = std::sync::OnceLock::new();

/// Initialize global memory pool
///
/// # Errors
///
/// Returns error if the memory pool is already initialized.
pub fn init_global_memory_pool(config: MemoryPoolConfig) -> Result<(), &'static str> {
    let pool = Arc::new(MemoryPool::new(config));
    GLOBAL_MEMORY_POOL
        .set(pool)
        .map_err(|_| "Memory pool already initialized")
}

/// Get global memory pool instance
///
/// # Errors
///
/// Returns None if the memory pool is not initialized.
#[must_use]
pub fn global_memory_pool() -> Option<&'static Arc<MemoryPool>> {
    GLOBAL_MEMORY_POOL.get()
}
