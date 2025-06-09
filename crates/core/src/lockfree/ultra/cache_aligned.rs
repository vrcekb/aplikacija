//! Cache-aligned data structures for optimal performance
//!
//! Provides cache-aligned wrappers to prevent false sharing and optimize
//! memory access patterns for ultra-low latency operations.

use super::{UltraError, UltraResult, CACHE_LINE_SIZE};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache-aligned wrapper to prevent false sharing
#[repr(C, align(64))]
#[derive(Debug)]
pub struct CacheAligned<T> {
    value: T,
    _padding: [u8; 0], // Zero-sized padding for alignment
}

impl<T> CacheAligned<T> {
    /// Creates new cache-aligned value
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self {
            value,
            _padding: [],
        }
    }

    /// Gets reference to inner value
    #[must_use]
    pub const fn get(&self) -> &T {
        &self.value
    }

    /// Gets mutable reference to inner value
    pub const fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }

    /// Consumes wrapper and returns inner value
    #[must_use]
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<T: Clone> Clone for CacheAligned<T> {
    fn clone(&self) -> Self {
        Self::new(self.value.clone())
    }
}

impl<T: Default> Default for CacheAligned<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// Cache-padded atomic counter for high-performance scenarios
#[repr(C, align(64))]
#[derive(Debug)]
pub struct CachePaddedAtomic {
    counter: AtomicUsize,
    _pad: [u8; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
}

impl CachePaddedAtomic {
    /// Creates new cache-padded atomic counter
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self {
            counter: AtomicUsize::new(value),
            _pad: [0; CACHE_LINE_SIZE - std::mem::size_of::<AtomicUsize>()],
        }
    }

    /// Loads value with specified ordering
    #[must_use]
    pub fn load(&self, order: Ordering) -> usize {
        self.counter.load(order)
    }

    /// Stores value with specified ordering
    pub fn store(&self, val: usize, order: Ordering) {
        self.counter.store(val, order);
    }

    /// Fetch and add with specified ordering
    pub fn fetch_add(&self, val: usize, order: Ordering) -> usize {
        self.counter.fetch_add(val, order)
    }

    /// Compare and swap with specified ordering
    ///
    /// # Errors
    /// Returns `Err` with the current value if the exchange failed
    pub fn compare_exchange(
        &self,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        self.counter
            .compare_exchange(current, new, success, failure)
    }

    /// Compare and swap weak with specified ordering
    ///
    /// # Errors
    /// Returns `Err` with the current value if the exchange failed
    pub fn compare_exchange_weak(
        &self,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering,
    ) -> Result<usize, usize> {
        self.counter
            .compare_exchange_weak(current, new, success, failure)
    }
}

impl Default for CachePaddedAtomic {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Validates memory alignment for optimal performance
///
/// # Errors
/// Returns `UltraError::AlignmentError` if pointer is not aligned to the specified boundary
pub fn validate_alignment<T>(ptr: *const T, alignment: usize) -> UltraResult<()> {
    if (ptr as usize) % alignment != 0 {
        return Err(UltraError::AlignmentError);
    }
    Ok(())
}

/// Gets cache line size at runtime
#[must_use]
pub const fn cache_line_size() -> usize {
    CACHE_LINE_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_cache_aligned_size() {
        let aligned = CacheAligned::new(42u64);
        assert_eq!(mem::align_of_val(&aligned), 64);
    }

    #[test]
    fn test_cache_aligned_operations() {
        let mut aligned = CacheAligned::new(42_i32);
        assert_eq!(*aligned, 42_i32);
        assert_eq!(aligned.get(), &42_i32);

        *aligned = 100_i32;
        assert_eq!(*aligned, 100_i32);

        let value = aligned.into_inner();
        assert_eq!(value, 100_i32);
    }

    #[test]
    fn test_cache_padded_atomic() {
        let atomic = CachePaddedAtomic::new(0);
        assert_eq!(mem::align_of_val(&atomic), 64);
        assert_eq!(atomic.load(Ordering::Relaxed), 0);

        atomic.store(42, Ordering::Relaxed);
        assert_eq!(atomic.load(Ordering::Relaxed), 42);

        let old = atomic.fetch_add(10, Ordering::Relaxed);
        assert_eq!(old, 42);
        assert_eq!(atomic.load(Ordering::Relaxed), 52);
    }

    #[test]
    fn test_alignment_validation() {
        let value = 42_u64;
        let ptr = &raw const value;

        // Should pass for reasonable alignments
        assert!(validate_alignment(ptr, 1_usize).is_ok());
        assert!(validate_alignment(ptr, 8_usize).is_ok());

        // Test with misaligned pointer
        let misaligned = (ptr as usize + 1_usize) as *const u64;
        assert!(validate_alignment(misaligned, 8_usize).is_err());
    }
}
