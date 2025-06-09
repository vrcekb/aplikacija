//! CPU cache optimization utilities for ultra-low latency trading
//!
//! This module provides cache-aware data structures and algorithms
//! optimized for modern CPU architectures and cache hierarchies.

use std::alloc::{alloc, dealloc, Layout};
use std::mem;
use std::ptr;
use thiserror::Error;

/// CPU cache optimization errors
#[derive(Error, Debug, Clone)]
pub enum CacheOptimizationError {
    /// Memory allocation failed
    #[error("Memory allocation failed: {reason}")]
    AllocationFailed {
        /// Error reason
        reason: String,
    },

    /// Invalid alignment
    #[error("Invalid alignment: {alignment}, must be power of 2")]
    InvalidAlignment {
        /// Invalid alignment value
        alignment: usize,
    },

    /// Cache line size detection failed
    #[error("Failed to detect cache line size")]
    CacheLineSizeDetectionFailed,
}

/// Result type for cache optimization operations
pub type CacheOptimizationResult<T> = Result<T, CacheOptimizationError>;

/// CPU cache line size constants
pub struct CacheLineSize;

impl CacheLineSize {
    /// Common cache line size (64 bytes)
    pub const COMMON: usize = 64;

    /// Intel cache line size
    pub const INTEL: usize = 64;

    /// AMD cache line size
    pub const AMD: usize = 64;

    /// ARM cache line size
    pub const ARM: usize = 64;

    /// Detect cache line size at runtime
    #[must_use]
    pub const fn detect() -> usize {
        // For now, return common size
        // In a real implementation, we'd use CPUID or similar
        Self::COMMON
    }
}

/// Cache-aligned memory allocator
pub struct CacheAlignedAllocator {
    alignment: usize,
}

impl CacheAlignedAllocator {
    /// Create new cache-aligned allocator
    ///
    /// # Errors
    ///
    /// Returns error if alignment is not a power of 2
    pub const fn new(alignment: usize) -> CacheOptimizationResult<Self> {
        if !alignment.is_power_of_two() {
            return Err(CacheOptimizationError::InvalidAlignment { alignment });
        }

        Ok(Self { alignment })
    }

    /// Create allocator with cache line alignment
    ///
    /// # Errors
    ///
    /// Returns error if cache line size detection fails
    pub const fn cache_line_aligned() -> CacheOptimizationResult<Self> {
        let alignment = CacheLineSize::detect();
        Self::new(alignment)
    }

    /// Allocate cache-aligned memory
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails
    ///
    /// # Safety
    ///
    /// The returned pointer must be deallocated with `deallocate`
    pub unsafe fn allocate<T>(&self, count: usize) -> CacheOptimizationResult<*mut T> {
        let size = mem::size_of::<T>() * count;
        let layout = Layout::from_size_align(size, self.alignment).map_err(|_| {
            CacheOptimizationError::AllocationFailed {
                reason: "Invalid layout".to_string(),
            }
        })?;

        let ptr = alloc(layout);
        if ptr.is_null() {
            return Err(CacheOptimizationError::AllocationFailed {
                reason: "System allocator returned null".to_string(),
            });
        }

        Ok(ptr.cast::<T>())
    }

    /// Deallocate cache-aligned memory
    ///
    /// # Safety
    ///
    /// The pointer must have been allocated with `allocate`
    pub unsafe fn deallocate<T>(&self, ptr: *mut T, count: usize) {
        if !ptr.is_null() {
            let size = mem::size_of::<T>() * count;
            let layout = Layout::from_size_align_unchecked(size, self.alignment);
            dealloc(ptr.cast::<u8>(), layout);
        }
    }
}

/// Cache-aligned vector for hot data
pub struct CacheAlignedVec<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
    allocator: CacheAlignedAllocator,
}

impl<T> CacheAlignedVec<T> {
    /// Create new cache-aligned vector
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails
    pub fn new() -> CacheOptimizationResult<Self> {
        Self::with_capacity(0)
    }

    /// Create cache-aligned vector with capacity
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails
    pub fn with_capacity(capacity: usize) -> CacheOptimizationResult<Self> {
        let allocator = CacheAlignedAllocator::cache_line_aligned()?;

        let ptr = if capacity == 0 {
            ptr::null_mut()
        } else {
            unsafe { allocator.allocate::<T>(capacity)? }
        };

        Ok(Self {
            ptr,
            len: 0,
            capacity,
            allocator,
        })
    }

    /// Get current length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Push element to vector
    ///
    /// # Errors
    ///
    /// Returns error if reallocation fails
    pub fn push(&mut self, value: T) -> CacheOptimizationResult<()> {
        if self.len == self.capacity {
            self.grow()?;
        }

        unsafe {
            ptr::write(self.ptr.add(self.len), value);
        }
        self.len += 1;

        Ok(())
    }

    /// Pop element from vector
    #[must_use]
    pub const fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe { Some(ptr::read(self.ptr.add(self.len))) }
        }
    }

    /// Get element at index
    ///
    /// # Safety
    ///
    /// Index must be within bounds
    #[must_use]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        &*self.ptr.add(index)
    }

    /// Get mutable element at index
    ///
    /// # Safety
    ///
    /// Index must be within bounds
    #[must_use]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        &mut *self.ptr.add(index)
    }

    /// Grow vector capacity
    fn grow(&mut self) -> CacheOptimizationResult<()> {
        let new_capacity = if self.capacity == 0 {
            4
        } else {
            self.capacity * 2
        };

        let new_ptr = unsafe { self.allocator.allocate::<T>(new_capacity)? };

        if !self.ptr.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len);
                self.allocator.deallocate(self.ptr, self.capacity);
            }
        }

        self.ptr = new_ptr;
        self.capacity = new_capacity;

        Ok(())
    }
}

impl<T> Drop for CacheAlignedVec<T> {
    fn drop(&mut self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                ptr::drop_in_place(self.ptr.add(i));
            }
        }

        // Deallocate memory
        if !self.ptr.is_null() {
            unsafe {
                self.allocator.deallocate(self.ptr, self.capacity);
            }
        }
    }
}

unsafe impl<T: Send> Send for CacheAlignedVec<T> {}
unsafe impl<T: Sync> Sync for CacheAlignedVec<T> {}

/// Prefetch utilities for cache optimization
pub struct Prefetch;

impl Prefetch {
    /// Prefetch data for read access
    pub fn read<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            let _ = ptr;
        }
    }

    /// Prefetch data for write access
    pub fn write<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            let _ = ptr;
        }
    }

    /// Prefetch cache line for temporal locality
    pub fn temporal<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T1);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            let _ = ptr;
        }
    }

    /// Prefetch cache line for non-temporal access
    pub fn non_temporal<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_NTA);
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback for other architectures
            let _ = ptr;
        }
    }
}

/// Cache-friendly data layout utilities
pub struct CacheLayout;

impl CacheLayout {
    /// Calculate optimal padding for cache alignment
    #[must_use]
    pub const fn padding_for_alignment(size: usize, alignment: usize) -> usize {
        let remainder = size % alignment;
        if remainder == 0 {
            0
        } else {
            alignment - remainder
        }
    }

    /// Round size up to cache line boundary
    #[must_use]
    pub const fn round_to_cache_line(size: usize) -> usize {
        let cache_line_size = CacheLineSize::COMMON;
        let remainder = size % cache_line_size;
        if remainder == 0 {
            size
        } else {
            size + cache_line_size - remainder
        }
    }

    /// Check if pointer is cache-aligned
    #[must_use]
    pub fn is_cache_aligned<T>(ptr: *const T) -> bool {
        let addr = ptr as usize;
        addr % CacheLineSize::COMMON == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_line_size() {
        let size = CacheLineSize::detect();
        assert_eq!(size, CacheLineSize::COMMON);
    }

    #[test]
    fn test_cache_aligned_allocator() -> CacheOptimizationResult<()> {
        let allocator = CacheAlignedAllocator::cache_line_aligned()?;

        unsafe {
            let ptr: *mut u64 = allocator.allocate(10)?;
            assert!(!ptr.is_null());
            assert_eq!(ptr as usize % CacheLineSize::COMMON, 0);

            allocator.deallocate(ptr, 10);
        }

        Ok(())
    }

    #[test]
    fn test_cache_aligned_vec() -> CacheOptimizationResult<()> {
        let mut vec: CacheAlignedVec<i32> = CacheAlignedVec::new()?;
        assert!(vec.is_empty());

        vec.push(42_i32)?;
        assert_eq!(vec.len(), 1_usize);

        let value = vec.pop();
        assert_eq!(value, Some(42_i32));
        assert!(vec.is_empty());

        Ok(())
    }

    #[test]
    fn test_cache_layout_utilities() {
        assert_eq!(
            CacheLayout::padding_for_alignment(60_usize, 64_usize),
            4_usize
        );
        assert_eq!(
            CacheLayout::padding_for_alignment(64_usize, 64_usize),
            0_usize
        );

        assert_eq!(CacheLayout::round_to_cache_line(60_usize), 64_usize);
        assert_eq!(CacheLayout::round_to_cache_line(64_usize), 64_usize);
        assert_eq!(CacheLayout::round_to_cache_line(100_usize), 128_usize);
    }
}
