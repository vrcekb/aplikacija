//! NUMA-Aware Memory Allocator - Optimized Memory Placement
//!
//! Implements NUMA-aware memory allocation with:
//! - Thread-local memory pools for optimal cache locality
//! - NUMA node affinity for memory allocations
//! - Cross-NUMA access minimization strategies
//! - Real-time allocation performance monitoring
//! - Integration with memory pressure monitoring

use std::alloc::{GlobalAlloc, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use thiserror::Error;

use crate::memory::pool::{MemoryPool, MemoryPoolConfig, MemoryPoolError, MemoryPoolFeatures};

#[cfg(feature = "numa")]
use crate::engine::numa::{NumaError, NumaScheduler};

/// NUMA allocator error types
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AllocatorError {
    /// Memory pool error
    #[error("Memory pool error: {0}")]
    PoolError(#[from] MemoryPoolError),

    /// NUMA error
    #[cfg(feature = "numa")]
    #[error("NUMA error: {0}")]
    NumaError(#[from] NumaError),

    /// Allocation failed
    #[error("Allocation failed: size {size}, alignment {alignment}")]
    AllocationFailed {
        /// Requested size
        size: usize,
        /// Requested alignment
        alignment: usize,
    },

    /// Invalid NUMA node
    #[error("Invalid NUMA node: {node_id}")]
    InvalidNumaNode {
        /// Node ID
        node_id: usize,
    },

    /// Thread not registered
    #[error("Thread not registered with NUMA allocator")]
    ThreadNotRegistered,
}

/// Allocator statistics
#[derive(Debug, Default)]
pub struct AllocatorStats {
    /// Total allocations
    pub total_allocations: AtomicU64,
    /// Total deallocations
    pub total_deallocations: AtomicU64,
    /// Local NUMA allocations
    pub local_numa_allocations: AtomicU64,
    /// Cross-NUMA allocations
    pub cross_numa_allocations: AtomicU64,
    /// Total bytes allocated
    pub total_bytes_allocated: AtomicU64,
    /// Average allocation time in nanoseconds
    pub avg_allocation_time_ns: AtomicU64,
    /// Cache misses due to cross-NUMA access
    pub numa_cache_misses: AtomicU64,
}

impl AllocatorStats {
    /// Get NUMA locality ratio (0.0-1.0)
    #[must_use]
    pub fn numa_locality_ratio(&self) -> f64 {
        let total = self.total_allocations.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0_f64;
        }

        let local = self.local_numa_allocations.load(Ordering::Relaxed);
        f64::from(u32::try_from(local).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(total).unwrap_or(u32::MAX))
    }

    /// Record allocation
    pub fn record_allocation(&self, size: u64, is_local: bool, allocation_time_ns: u64) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        self.total_bytes_allocated
            .fetch_add(size, Ordering::Relaxed);

        if is_local {
            self.local_numa_allocations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cross_numa_allocations.fetch_add(1, Ordering::Relaxed);
            self.numa_cache_misses.fetch_add(1, Ordering::Relaxed);
        }

        // Update average allocation time
        let total_allocs = self.total_allocations.load(Ordering::Relaxed);
        if total_allocs > 0 {
            let current_avg = self.avg_allocation_time_ns.load(Ordering::Relaxed);
            let new_avg = (current_avg * (total_allocs - 1) + allocation_time_ns) / total_allocs;
            self.avg_allocation_time_ns
                .store(new_avg, Ordering::Relaxed);
        }
    }

    /// Record deallocation
    pub fn record_deallocation(&self) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
    }
}

/// Thread-local memory pool information
#[derive(Debug)]
#[allow(dead_code)] // Fields used for future NUMA optimizations
struct ThreadLocalPool {
    numa_node: usize,
    pool: Arc<MemoryPool>,
    thread_id: thread::ThreadId,
}

/// NUMA-aware memory allocator
pub struct NumaAllocator {
    /// NUMA scheduler for thread placement
    #[cfg(feature = "numa")]
    numa_scheduler: Arc<NumaScheduler>,
    /// Memory pools per NUMA node
    numa_pools: Vec<Arc<MemoryPool>>,
    /// Thread-local pool assignments
    thread_pools: Mutex<HashMap<thread::ThreadId, ThreadLocalPool>>,
    /// Allocator statistics
    stats: Arc<AllocatorStats>,
    /// Fallback allocator for large allocations
    fallback_allocator: std::alloc::System,
    /// Maximum pool allocation size
    max_pool_size: usize,
}

impl NumaAllocator {
    /// Create new NUMA-aware allocator
    ///
    /// # Errors
    ///
    /// Returns error if NUMA initialization fails
    pub fn new(numa_scheduler: Arc<NumaScheduler>) -> Result<Self, AllocatorError> {
        let numa_node_count = numa_scheduler
            .stats()
            .total_assignments
            .load(Ordering::Relaxed);
        let node_count = if numa_node_count == 0 {
            2
        } else {
            usize::try_from(numa_node_count).unwrap_or(2)
        };

        let mut numa_pools = Vec::with_capacity(node_count);

        // Create memory pool for each NUMA node
        for node_id in 0..node_count {
            let config = MemoryPoolConfig {
                name: format!("numa_pool_{node_id}"),
                block_size: 4096,     // 4KB blocks
                initial_blocks: 2048, // 8MB initial
                max_blocks: 32768,    // 128MB max
                alignment: 64,        // Cache line alignment
                features: MemoryPoolFeatures {
                    corruption_detection: true,
                    enable_stats: true,
                    enable_huge_pages: false,
                    enable_prefaulting: true,
                },
                numa_node: Some(node_id),
                thread_cache_size: 32,
            };

            let pool = MemoryPool::new(config)?;
            numa_pools.push(Arc::new(pool));
        }

        Ok(Self {
            numa_scheduler,
            numa_pools,
            thread_pools: Mutex::new(HashMap::new()),
            stats: Arc::new(AllocatorStats::default()),
            fallback_allocator: std::alloc::System,
            max_pool_size: 4096, // Use pool for allocations <= 4KB
        })
    }

    /// Register current thread with allocator
    ///
    /// # Errors
    ///
    /// Returns error if thread registration fails
    pub fn register_current_thread(&self) -> Result<(), AllocatorError> {
        let thread_id = thread::current().id();

        // Assign thread to NUMA node
        let core_id = self.numa_scheduler.assign_current_thread()?;

        // Determine NUMA node from core (simplified)
        let numa_node = core_id.id % self.numa_pools.len();

        let pool = self
            .numa_pools
            .get(numa_node)
            .ok_or(AllocatorError::InvalidNumaNode { node_id: numa_node })?
            .clone();

        let thread_pool = ThreadLocalPool {
            numa_node,
            pool,
            thread_id,
        };

        if let Ok(mut thread_pools) = self.thread_pools.lock() {
            thread_pools.insert(thread_id, thread_pool);
        }

        Ok(())
    }

    /// Allocate memory with NUMA awareness
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails
    ///
    /// # Panics
    ///
    /// Panics if no NUMA pools are available (should never happen in production)
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocatorError> {
        let start_time = std::time::Instant::now();
        let thread_id = thread::current().id();

        // Use fallback allocator for large allocations
        if layout.size() > self.max_pool_size {
            let ptr = unsafe { self.fallback_allocator.alloc(layout) };
            if ptr.is_null() {
                return Err(AllocatorError::AllocationFailed {
                    size: layout.size(),
                    alignment: layout.align(),
                });
            }

            let allocation_time = start_time.elapsed().as_nanos();
            self.stats.record_allocation(
                u64::try_from(layout.size()).unwrap_or(u64::MAX),
                false, // Fallback is not NUMA-local
                u64::try_from(allocation_time).unwrap_or(u64::MAX),
            );

            return NonNull::new(ptr).ok_or_else(|| AllocatorError::AllocationFailed {
                size: layout.size(),
                alignment: layout.align(),
            });
        }

        // Get thread's NUMA pool - use simpler approach for production reliability
        let (pool, is_local) = if let Ok(thread_pools) = self.thread_pools.lock() {
            if let Some(thread_pool) = thread_pools.get(&thread_id) {
                (thread_pool.pool.clone(), true)
            } else {
                // Thread not registered, use first pool
                let first_pool =
                    self.numa_pools
                        .first()
                        .ok_or_else(|| AllocatorError::AllocationFailed {
                            size: layout.size(),
                            alignment: layout.align(),
                        })?;
                (first_pool.clone(), false)
            }
        } else {
            // Lock failed, use first pool as fallback
            let first_pool =
                self.numa_pools
                    .first()
                    .ok_or_else(|| AllocatorError::AllocationFailed {
                        size: layout.size(),
                        alignment: layout.align(),
                    })?;
            (first_pool.clone(), false)
        };

        // Allocate from pool
        let ptr = pool.allocate().map_err(AllocatorError::PoolError)?;

        let allocation_time = start_time.elapsed().as_nanos();
        self.stats.record_allocation(
            u64::try_from(layout.size()).unwrap_or(u64::MAX),
            is_local,
            u64::try_from(allocation_time).unwrap_or(u64::MAX),
        );

        Ok(ptr)
    }

    /// Deallocate memory
    ///
    /// # Errors
    ///
    /// Returns error if deallocation fails
    ///
    /// # Panics
    ///
    /// Panics if no NUMA pools are available (should never happen in production)
    pub fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<(), AllocatorError> {
        // Use fallback allocator for large allocations
        if layout.size() > self.max_pool_size {
            unsafe {
                self.fallback_allocator.dealloc(ptr.as_ptr(), layout);
            }
            self.stats.record_deallocation();
            return Ok(());
        }

        // Find the pool that owns this pointer (simplified - in production use metadata)
        let thread_id = thread::current().id();

        let pool = if let Ok(thread_pools) = self.thread_pools.lock() {
            if let Some(thread_pool) = thread_pools.get(&thread_id) {
                thread_pool.pool.clone()
            } else {
                self.numa_pools
                    .first()
                    .ok_or_else(|| AllocatorError::AllocationFailed {
                        size: layout.size(),
                        alignment: layout.align(),
                    })?
                    .clone()
            }
        } else {
            self.numa_pools
                .first()
                .ok_or_else(|| AllocatorError::AllocationFailed {
                    size: layout.size(),
                    alignment: layout.align(),
                })?
                .clone()
        };

        pool.deallocate(ptr).map_err(AllocatorError::PoolError)?;
        self.stats.record_deallocation();

        Ok(())
    }

    /// Get allocator statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<AllocatorStats> {
        &self.stats
    }

    /// Get NUMA pool for specific node
    #[must_use]
    pub fn numa_pool(&self, node_id: usize) -> Option<&Arc<MemoryPool>> {
        self.numa_pools.get(node_id)
    }

    /// Get number of NUMA nodes
    #[must_use]
    pub const fn numa_node_count(&self) -> usize {
        self.numa_pools.len()
    }

    /// Check if thread is registered
    #[must_use]
    pub fn is_thread_registered(&self) -> bool {
        let thread_id = thread::current().id();
        self.thread_pools
            .lock()
            .is_ok_and(|thread_pools| thread_pools.contains_key(&thread_id))
    }
}

unsafe impl GlobalAlloc for NumaAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocate(layout)
            .map_or(std::ptr::null_mut(), std::ptr::NonNull::as_ptr)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(non_null_ptr) = NonNull::new(ptr) {
            let _ = self.deallocate(non_null_ptr, layout);
        }
    }
}
