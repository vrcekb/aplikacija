//! Lock-free data structures for ultra-performance financial applications
//!
//! This module provides lock-free data structures optimized for <1ms latency
//! requirements in high-frequency trading and MEV applications.

use std::mem;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use thiserror::Error;

pub mod cache;
pub mod concurrent_algorithms;
pub mod queue;
pub mod ring_buffer;
pub mod ultra;

/// Lock-free data structure errors
#[derive(Error, Debug, Clone)]
pub enum LockFreeError {
    /// Queue is full
    #[error("Queue is full, capacity: {capacity}")]
    QueueFull {
        /// Queue capacity
        capacity: usize,
    },

    /// Queue is empty
    #[error("Queue is empty")]
    QueueEmpty,

    /// Memory allocation failed
    #[error("Memory allocation failed: {reason}")]
    AllocationFailed {
        /// Error reason
        reason: String,
    },

    /// Invalid capacity
    #[error("Invalid capacity: {capacity}, must be power of 2")]
    InvalidCapacity {
        /// Invalid capacity value
        capacity: usize,
    },

    /// Cache miss
    #[error("Cache miss for key")]
    CacheMiss,

    /// Concurrent modification detected
    #[error("Concurrent modification detected")]
    ConcurrentModification,
}

/// Result type for lock-free operations
pub type LockFreeResult<T> = Result<T, LockFreeError>;

/// Cache-aligned atomic pointer for lock-free operations
#[repr(align(64))]
pub struct CacheAlignedAtomicPtr<T> {
    ptr: AtomicPtr<T>,
}

impl<T> CacheAlignedAtomicPtr<T> {
    /// Create new cache-aligned atomic pointer
    #[must_use]
    pub const fn new(ptr: *mut T) -> Self {
        Self {
            ptr: AtomicPtr::new(ptr),
        }
    }

    /// Load pointer with specified ordering
    pub fn load(&self, order: Ordering) -> *mut T {
        self.ptr.load(order)
    }

    /// Store pointer with specified ordering
    pub fn store(&self, ptr: *mut T, order: Ordering) {
        self.ptr.store(ptr, order);
    }

    /// Compare and swap operation
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the exchange failed
    pub fn compare_exchange(
        &self,
        current: *mut T,
        new: *mut T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<*mut T, *mut T> {
        self.ptr.compare_exchange(current, new, success, failure)
    }

    /// Compare and swap weak operation
    ///
    /// # Errors
    ///
    /// Returns `Err` with the current value if the exchange failed
    pub fn compare_exchange_weak(
        &self,
        current: *mut T,
        new: *mut T,
        success: Ordering,
        failure: Ordering,
    ) -> Result<*mut T, *mut T> {
        self.ptr
            .compare_exchange_weak(current, new, success, failure)
    }
}

unsafe impl<T> Send for CacheAlignedAtomicPtr<T> {}
unsafe impl<T> Sync for CacheAlignedAtomicPtr<T> {}

/// Cache-aligned atomic counter for performance metrics
#[repr(align(64))]
pub struct CacheAlignedCounter {
    value: AtomicU64,
}

impl CacheAlignedCounter {
    /// Create new counter
    #[must_use]
    pub const fn new(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
        }
    }

    /// Get current value
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Increment counter and return new value
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Add to counter and return new value
    pub fn add(&self, val: u64) -> u64 {
        self.value.fetch_add(val, Ordering::Relaxed) + val
    }

    /// Reset counter to zero and return previous value
    pub fn reset(&self) -> u64 {
        self.value.swap(0, Ordering::Relaxed)
    }
}

/// Memory ordering utilities for lock-free operations
pub struct MemoryOrdering;

impl MemoryOrdering {
    /// Relaxed ordering for performance counters
    #[must_use]
    pub const fn relaxed() -> Ordering {
        Ordering::Relaxed
    }

    /// Acquire ordering for loading pointers
    #[must_use]
    pub const fn acquire() -> Ordering {
        Ordering::Acquire
    }

    /// Release ordering for storing pointers
    #[must_use]
    pub const fn release() -> Ordering {
        Ordering::Release
    }

    /// Acquire-Release ordering for RMW operations
    #[must_use]
    pub const fn acq_rel() -> Ordering {
        Ordering::AcqRel
    }

    /// Sequential consistency for critical sections
    #[must_use]
    pub const fn seq_cst() -> Ordering {
        Ordering::SeqCst
    }
}

/// Utility functions for lock-free operations
pub struct LockFreeUtils;

impl LockFreeUtils {
    /// Check if value is power of 2
    #[must_use]
    pub const fn is_power_of_two(n: usize) -> bool {
        n.is_power_of_two()
    }

    /// Round up to next power of 2
    #[must_use]
    pub const fn next_power_of_two(mut n: usize) -> usize {
        if n == 0 {
            return 1;
        }
        n -= 1;
        n |= n >> 1_i32;
        n |= n >> 2_i32;
        n |= n >> 4_i32;
        n |= n >> 8_i32;
        n |= n >> 16_i32;
        if mem::size_of::<usize>() > 4 {
            n |= n >> 32_i32;
        }
        n + 1
    }

    /// CPU pause instruction for spin loops
    pub fn cpu_pause() {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_pause();
        }
        #[cfg(not(target_arch = "x86_64"))]
        std::hint::spin_loop();
    }

    /// Memory fence for ordering guarantees
    pub fn memory_fence() {
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

/// Performance statistics trait for lock-free structures
pub trait LockFreeStats {
    /// Get hit count
    fn hit_count(&self) -> u64;

    /// Get miss count
    fn miss_count(&self) -> u64;

    /// Get total operations
    fn total_operations(&self) -> u64;

    /// Get hit ratio
    fn hit_ratio(&self) -> f64;

    /// Reset statistics
    fn reset_stats(&self);
}

/// Performance statistics for lock-free structures
#[derive(Debug, Clone)]
pub struct LockFreeStatsData {
    /// Total operations performed
    pub operations: u64,
    /// Successful operations
    pub successes: u64,
    /// Failed operations (contention, full queue, etc.)
    pub failures: u64,
    /// Average operation latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Maximum operation latency in nanoseconds
    pub max_latency_ns: u64,
    /// Cache hit rate (0.0 - 1.0)
    pub cache_hit_rate: f64,
}

impl LockFreeStatsData {
    /// Create new statistics
    #[must_use]
    pub const fn new() -> Self {
        Self {
            operations: 0,
            successes: 0,
            failures: 0,
            avg_latency_ns: 0,
            max_latency_ns: 0,
            cache_hit_rate: 0.0,
        }
    }

    /// Calculate success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.operations == 0 {
            0.0
        } else {
            f64::from(u32::try_from(self.successes).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(self.operations).unwrap_or(u32::MAX))
        }
    }

    /// Calculate failure rate
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }
}

impl Default for LockFreeStatsData {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_counter() {
        let counter = CacheAlignedCounter::new(0);
        assert_eq!(counter.get(), 0);

        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.get(), 1);

        assert_eq!(counter.add(5), 6);
        assert_eq!(counter.get(), 6);

        assert_eq!(counter.reset(), 6);
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_power_of_two_utils() {
        assert!(LockFreeUtils::is_power_of_two(1));
        assert!(LockFreeUtils::is_power_of_two(2));
        assert!(LockFreeUtils::is_power_of_two(4));
        assert!(LockFreeUtils::is_power_of_two(8));
        assert!(!LockFreeUtils::is_power_of_two(3));
        assert!(!LockFreeUtils::is_power_of_two(5));
        assert!(!LockFreeUtils::is_power_of_two(0));

        assert_eq!(LockFreeUtils::next_power_of_two(0), 1);
        assert_eq!(LockFreeUtils::next_power_of_two(1), 1);
        assert_eq!(LockFreeUtils::next_power_of_two(2), 2);
        assert_eq!(LockFreeUtils::next_power_of_two(3), 4);
        assert_eq!(LockFreeUtils::next_power_of_two(5), 8);
        assert_eq!(LockFreeUtils::next_power_of_two(9), 16);
    }

    #[test]
    fn test_lock_free_stats() {
        let stats = LockFreeStatsData::new();
        assert!((stats.success_rate() - 0.0_f64).abs() < f64::EPSILON);
        assert!((stats.failure_rate() - 1.0_f64).abs() < f64::EPSILON);

        let stats = LockFreeStatsData {
            operations: 100,
            successes: 95,
            failures: 5,
            ..LockFreeStatsData::new()
        };

        assert!((stats.success_rate() - 0.95).abs() < f64::EPSILON);
        assert!((stats.failure_rate() - 0.05).abs() < f64::EPSILON);
    }
}
