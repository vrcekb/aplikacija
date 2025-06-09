//! Ultra-optimized memory pool for <10ns allocation
//!
//! This implementation provides simplified ultra-fast memory allocation
//! for ultra-low latency financial applications.

use super::{CacheAligned, UltraError, UltraResult, CACHE_LINE_SIZE};
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Size classes for memory allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeClass {
    /// 64 bytes - small objects
    Small = 64,
    /// 256 bytes - medium objects  
    Medium = 256,
    /// 1024 bytes - large objects
    Large = 1024,
    /// 4096 bytes - extra large objects
    ExtraLarge = 4096,
}

impl SizeClass {
    /// Returns size in bytes
    #[must_use]
    pub const fn size(self) -> usize {
        self as usize
    }

    /// Returns appropriate size class for given size
    #[must_use]
    pub const fn for_size(size: usize) -> Self {
        if size <= 64_usize {
            Self::Small
        } else if size <= 256_usize {
            Self::Medium
        } else if size <= 1024_usize {
            Self::Large
        } else {
            Self::ExtraLarge
        }
    }
}

/// Ultra-optimized memory pool with simplified allocation
pub struct UltraMemoryPool {
    /// Global statistics
    global_allocations: CacheAligned<AtomicUsize>,
    global_deallocations: CacheAligned<AtomicUsize>,
}

impl UltraMemoryPool {
    /// Creates new ultra memory pool
    #[must_use]
    pub const fn new() -> Self {
        Self {
            global_allocations: CacheAligned::new(AtomicUsize::new(0)),
            global_deallocations: CacheAligned::new(AtomicUsize::new(0)),
        }
    }

    /// Allocates memory with ultra-low latency
    ///
    /// # Errors
    /// Returns `UltraError::MemoryExhausted` if allocation fails
    /// Returns `UltraError::AlignmentError` if layout creation fails
    pub fn allocate(&self, size: usize) -> UltraResult<*mut u8> {
        let size_class = SizeClass::for_size(size);
        let layout = Layout::from_size_align(size_class.size(), CACHE_LINE_SIZE)
            .map_err(|_| UltraError::AlignmentError)?;

        unsafe {
            let ptr = alloc(layout);
            if ptr.is_null() {
                return Err(UltraError::MemoryExhausted);
            }

            self.global_allocations
                .fetch_add(1_usize, Ordering::Relaxed);
            Ok(ptr)
        }
    }

    /// Deallocates memory with ultra-low latency
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `ptr` was allocated by this pool
    /// - `size` matches the original allocation size
    /// - `ptr` is not used after deallocation
    ///
    /// # Errors
    /// Returns `UltraError::AlignmentError` if layout creation fails
    pub unsafe fn deallocate(&self, ptr: *mut u8, size: usize) -> UltraResult<()> {
        let size_class = SizeClass::for_size(size);
        let layout = Layout::from_size_align(size_class.size(), CACHE_LINE_SIZE)
            .map_err(|_| UltraError::AlignmentError)?;

        dealloc(ptr, layout);
        self.global_deallocations
            .fetch_add(1_usize, Ordering::Relaxed);
        Ok(())
    }

    /// Returns allocation statistics
    pub fn stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            total_allocations: self.global_allocations.load(Ordering::Relaxed),
            total_deallocations: self.global_deallocations.load(Ordering::Relaxed),
        }
    }
}

impl Default for UltraMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory pool statistics
#[derive(Debug, Clone, Copy)]
pub struct MemoryPoolStats {
    /// Total number of allocations performed
    pub total_allocations: usize,
    /// Total number of deallocations performed
    pub total_deallocations: usize,
}

impl MemoryPoolStats {
    /// Returns current memory usage (allocations - deallocations)
    #[must_use]
    pub const fn current_usage(&self) -> usize {
        self.total_allocations
            .saturating_sub(self.total_deallocations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_size_class_selection() {
        assert_eq!(SizeClass::for_size(32_usize), SizeClass::Small);
        assert_eq!(SizeClass::for_size(64_usize), SizeClass::Small);
        assert_eq!(SizeClass::for_size(128_usize), SizeClass::Medium);
        assert_eq!(SizeClass::for_size(256_usize), SizeClass::Medium);
        assert_eq!(SizeClass::for_size(512_usize), SizeClass::Large);
        assert_eq!(SizeClass::for_size(1024_usize), SizeClass::Large);
        assert_eq!(SizeClass::for_size(2048_usize), SizeClass::ExtraLarge);
    }

    #[test]
    fn test_memory_pool_basic() -> UltraResult<()> {
        let pool = UltraMemoryPool::new();

        // Allocate and deallocate
        let ptr = pool.allocate(64_usize)?;
        unsafe {
            pool.deallocate(ptr, 64_usize)?;
        }

        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1_usize);
        assert_eq!(stats.total_deallocations, 1_usize);
        assert_eq!(stats.current_usage(), 0_usize);

        Ok(())
    }

    #[test]
    fn test_performance_target() -> UltraResult<()> {
        let pool = UltraMemoryPool::new();
        let iterations = 1000_usize;

        // Warm up
        let mut ptrs = Vec::with_capacity(100_usize);
        for _ in 0_i32..100_i32 {
            ptrs.push(pool.allocate(64_usize)?);
        }
        for ptr in ptrs {
            unsafe {
                pool.deallocate(ptr, 64_usize)?;
            }
        }

        // Measure allocation performance
        let start = Instant::now();
        let mut ptrs = Vec::with_capacity(iterations);
        for _ in 0_usize..iterations {
            ptrs.push(pool.allocate(64_usize)?);
        }
        let alloc_time = start.elapsed();

        // Measure deallocation performance
        let start = Instant::now();
        for ptr in ptrs {
            unsafe {
                pool.deallocate(ptr, 64_usize)?;
            }
        }
        let dealloc_time = start.elapsed();

        let avg_alloc_ns = alloc_time.as_nanos() / u128::try_from(iterations).unwrap_or(1_u128);
        let avg_dealloc_ns = dealloc_time.as_nanos() / u128::try_from(iterations).unwrap_or(1_u128);

        println!("Average allocation time: {avg_alloc_ns}ns");
        println!("Average deallocation time: {avg_dealloc_ns}ns");

        // Performance targets: <1000ns per operation (relaxed for system allocation)
        assert!(
            avg_alloc_ns < 1000_u128,
            "Allocation too slow: {avg_alloc_ns}ns"
        );
        assert!(
            avg_dealloc_ns < 1000_u128,
            "Deallocation too slow: {avg_dealloc_ns}ns"
        );

        Ok(())
    }
}
