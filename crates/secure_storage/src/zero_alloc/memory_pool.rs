//! # Custom Memory Pool Allocator
//!
//! High-performance memory pool allocator for predictable allocation patterns
//! in financial trading applications.

#![allow(unsafe_code)] // Required for high-performance memory management

use crate::error::{SecureStorageError, SecureStorageResult};
use parking_lot::Mutex;
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tracing::{debug, error};

/// Memory pool allocator
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool of free blocks
    free_blocks: Mutex<Vec<NonNull<u8>>>,
    /// Block size for this pool
    block_size: usize,
    /// Total number of blocks
    total_blocks: usize,
    /// Number of allocated blocks
    allocated_blocks: AtomicUsize,
    /// Performance counters
    allocations: AtomicU64,
    deallocations: AtomicU64,
    /// Pool memory region
    pool_memory: NonNull<u8>,
    /// Pool size in bytes
    pool_size: usize,
}

impl MemoryPool {
    /// Create a new memory pool
    ///
    /// # Errors
    ///
    /// Returns error if pool creation fails
    ///
    /// # Safety
    ///
    /// This function allocates raw memory and must be used carefully
    pub unsafe fn new(block_size: usize, num_blocks: usize) -> SecureStorageResult<Self> {
        if block_size == 0 || num_blocks == 0 {
            return Err(SecureStorageError::InvalidInput {
                field: "pool_parameters".to_string(),
                reason: "Block size and number of blocks must be non-zero".to_string(),
            });
        }

        // Align block size to 8 bytes
        let aligned_block_size = (block_size + 7) & !7;
        let pool_size = aligned_block_size * num_blocks;

        // Allocate pool memory
        let layout = Layout::from_size_align(pool_size, 8).map_err(|_| {
            SecureStorageError::InvalidInput {
                field: "memory_layout".to_string(),
                reason: "Invalid memory layout".to_string(),
            }
        })?;

        let pool_memory = std::alloc::alloc(layout);
        if pool_memory.is_null() {
            return Err(SecureStorageError::InsufficientResources {
                resource: "memory".to_string(),
                reason: format!("Failed to allocate {pool_size} bytes for memory pool"),
            });
        }

        let pool_memory = NonNull::new_unchecked(pool_memory);

        // Initialize free block list
        let mut free_blocks = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let block_ptr = pool_memory.as_ptr().add(i * aligned_block_size);
            free_blocks.push(NonNull::new_unchecked(block_ptr));
        }

        debug!(
            "Created memory pool: {} blocks of {} bytes each (total: {} bytes)",
            num_blocks, aligned_block_size, pool_size
        );

        Ok(Self {
            free_blocks: Mutex::new(free_blocks),
            block_size: aligned_block_size,
            total_blocks: num_blocks,
            allocated_blocks: AtomicUsize::new(0),
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            pool_memory,
            pool_size,
        })
    }

    /// Allocate a block from the pool
    ///
    /// # Errors
    ///
    /// Returns error if no blocks are available
    pub fn allocate(&self) -> SecureStorageResult<NonNull<u8>> {
        self.allocations.fetch_add(1, Ordering::Relaxed);

        let mut free_blocks = self.free_blocks.lock();

        free_blocks.pop().map_or_else(
            || {
                Err(SecureStorageError::InsufficientResources {
                    resource: "memory_blocks".to_string(),
                    reason: format!(
                        "No free blocks available in pool ({} total)",
                        self.total_blocks
                    ),
                })
            },
            |block| {
                self.allocated_blocks.fetch_add(1, Ordering::Relaxed);
                debug!("Allocated block from pool");
                Ok(block)
            },
        )
    }

    /// Deallocate a block back to the pool
    ///
    /// # Errors
    ///
    /// Returns error if block is invalid
    ///
    /// # Safety
    ///
    /// The caller must ensure the block belongs to this pool
    pub unsafe fn deallocate(&self, block: NonNull<u8>) -> SecureStorageResult<()> {
        // Verify block belongs to this pool
        if !self.is_valid_block(block) {
            return Err(SecureStorageError::InvalidInput {
                field: "block_pointer".to_string(),
                reason: "Block does not belong to this pool".to_string(),
            });
        }

        self.deallocations.fetch_add(1, Ordering::Relaxed);

        // Zero the block for security
        std::ptr::write_bytes(block.as_ptr(), 0, self.block_size);

        self.free_blocks.lock().push(block);

        self.allocated_blocks.fetch_sub(1, Ordering::Relaxed);

        debug!("Deallocated block back to pool");
        Ok(())
    }

    /// Check if a block pointer is valid for this pool
    fn is_valid_block(&self, block: NonNull<u8>) -> bool {
        let block_addr = block.as_ptr() as usize;
        let pool_start = self.pool_memory.as_ptr() as usize;
        let pool_end = pool_start + self.pool_size;

        // Check if block is within pool bounds
        if block_addr < pool_start || block_addr >= pool_end {
            return false;
        }

        // Check if block is properly aligned
        let offset = block_addr - pool_start;
        offset % self.block_size == 0
    }

    /// Get pool statistics
    #[must_use]
    pub fn get_stats(&self) -> MemoryPoolStats {
        let free_blocks = self.free_blocks.lock().len();

        MemoryPoolStats {
            block_size: self.block_size,
            total_blocks: self.total_blocks,
            allocated_blocks: self.allocated_blocks.load(Ordering::Relaxed),
            free_blocks,
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
            pool_size: self.pool_size,
        }
    }

    /// Get block size
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get total number of blocks
    #[must_use]
    pub const fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Check if pool is full (all blocks allocated)
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.allocated_blocks.load(Ordering::Relaxed) >= self.total_blocks
    }

    /// Check if pool is empty (no blocks allocated)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.allocated_blocks.load(Ordering::Relaxed) == 0
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        // Deallocate pool memory
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.pool_size, 8);
            std::alloc::dealloc(self.pool_memory.as_ptr(), layout);
        }
        debug!("Destroyed memory pool");
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    /// Size of each block
    pub block_size: usize,
    /// Total number of blocks
    pub total_blocks: usize,
    /// Number of allocated blocks
    pub allocated_blocks: usize,
    /// Number of free blocks
    pub free_blocks: usize,
    /// Total allocations performed
    pub allocations: u64,
    /// Total deallocations performed
    pub deallocations: u64,
    /// Total pool size in bytes
    pub pool_size: usize,
}

impl MemoryPoolStats {
    /// Calculate utilization percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.total_blocks == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            f64::from(u32::try_from(self.allocated_blocks).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(self.total_blocks).unwrap_or(u32::MAX))
                * 100.0
        }
    }

    /// Calculate fragmentation (should be 0 for pool allocator)
    #[must_use]
    pub fn fragmentation(&self) -> f64 {
        // Pool allocators don't fragment, but we can measure allocation/deallocation imbalance
        let imbalance = self.allocations.saturating_sub(self.deallocations);
        if self.allocations == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            f64::from(u32::try_from(imbalance).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(self.allocations).unwrap_or(u32::MAX))
                * 100.0
        }
    }
}

/// Multi-size memory pool manager
#[derive(Debug)]
pub struct PoolManager {
    /// Small blocks (64 bytes)
    small: MemoryPool,
    /// Medium blocks (256 bytes)
    medium: MemoryPool,
    /// Large blocks (1024 bytes)
    large: MemoryPool,
    /// Extra large blocks (4096 bytes)
    xl: MemoryPool,
}

impl PoolManager {
    /// Create a new pool manager
    ///
    /// # Errors
    ///
    /// Returns error if pool creation fails
    pub fn new() -> SecureStorageResult<Self> {
        unsafe {
            Ok(Self {
                small: MemoryPool::new(64, 1000)?,  // 64KB total
                medium: MemoryPool::new(256, 500)?, // 128KB total
                large: MemoryPool::new(1024, 200)?, // 200KB total
                xl: MemoryPool::new(4096, 50)?,     // 200KB total
            })
        }
    }

    /// Allocate memory of specified size
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails
    pub fn allocate(&self, size: usize) -> SecureStorageResult<PooledAllocation<'_>> {
        let (pool, actual_size) = if size <= 64 {
            (&self.small, 64)
        } else if size <= 256 {
            (&self.medium, 256)
        } else if size <= 1024 {
            (&self.large, 1024)
        } else if size <= 4096 {
            (&self.xl, 4096)
        } else {
            return Err(SecureStorageError::InvalidInput {
                field: "allocation_size".to_string(),
                reason: format!("Size {size} exceeds maximum pool size"),
            });
        };

        let block = pool.allocate()?;
        Ok(PooledAllocation {
            ptr: block,
            size: actual_size,
            pool,
        })
    }

    /// Get combined statistics
    #[must_use]
    pub fn get_stats(&self) -> PoolManagerStats {
        PoolManagerStats {
            small: self.small.get_stats(),
            medium: self.medium.get_stats(),
            large: self.large.get_stats(),
            xl: self.xl.get_stats(),
        }
    }
}

/// RAII wrapper for pooled allocations
pub struct PooledAllocation<'a> {
    /// Pointer to allocated block
    ptr: NonNull<u8>,
    /// Size of the block
    size: usize,
    /// Reference to the pool for deallocation
    pool: &'a MemoryPool,
}

impl PooledAllocation<'_> {
    /// Get pointer to allocated memory
    #[must_use]
    pub const fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Get size of allocated block
    #[must_use]
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Get mutable slice to allocated memory
    ///
    /// # Safety
    ///
    /// Caller must ensure the slice is used safely
    #[must_use]
    pub const unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size)
    }

    /// Get immutable slice to allocated memory
    ///
    /// # Safety
    ///
    /// Caller must ensure the slice is used safely
    #[must_use]
    pub const unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr.as_ptr(), self.size)
    }
}

impl Drop for PooledAllocation<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Err(e) = self.pool.deallocate(self.ptr) {
                error!("Failed to deallocate pooled memory: {}", e);
            }
        }
    }
}

/// Combined pool manager statistics
#[derive(Debug, Clone)]
pub struct PoolManagerStats {
    /// Small pool statistics
    pub small: MemoryPoolStats,
    /// Medium pool statistics
    pub medium: MemoryPoolStats,
    /// Large pool statistics
    pub large: MemoryPoolStats,
    /// Extra large pool statistics
    pub xl: MemoryPoolStats,
}

impl PoolManagerStats {
    /// Calculate total memory usage
    #[must_use]
    pub const fn total_memory_usage(&self) -> usize {
        (self.small.allocated_blocks * self.small.block_size)
            + (self.medium.allocated_blocks * self.medium.block_size)
            + (self.large.allocated_blocks * self.large.block_size)
            + (self.xl.allocated_blocks * self.xl.block_size)
    }

    /// Calculate total pool capacity
    #[must_use]
    pub const fn total_capacity(&self) -> usize {
        self.small.pool_size + self.medium.pool_size + self.large.pool_size + self.xl.pool_size
    }

    /// Calculate overall utilization
    #[must_use]
    pub fn overall_utilization(&self) -> f64 {
        let total_capacity = self.total_capacity();
        if total_capacity == 0 {
            0.0
        } else {
            // Use precise division for financial calculations
            f64::from(u32::try_from(self.total_memory_usage()).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total_capacity).unwrap_or(u32::MAX))
                * 100.0
        }
    }
}
