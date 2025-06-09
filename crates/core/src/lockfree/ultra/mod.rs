//! Ultra-optimized lock-free data structures for <1ms latency
//!
//! This module contains the highest-performance lock-free implementations
//! designed specifically for MEV trading and financial applications where
//! sub-millisecond latency is critical.

pub mod cache_aligned;
pub mod memory_pool;
pub mod spsc_queue;

pub use cache_aligned::CacheAligned;
pub use memory_pool::UltraMemoryPool;
pub use spsc_queue::UltraSPSCQueue;

use thiserror::Error;

/// Ultra-optimized lock-free operation errors
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum UltraError {
    /// Queue is full - cannot enqueue
    #[error("Queue is full")]
    QueueFull,

    /// Queue is empty - cannot dequeue
    #[error("Queue is empty")]
    QueueEmpty,

    /// Memory pool exhausted
    #[error("Memory pool exhausted")]
    MemoryExhausted,

    /// Invalid capacity (must be power of 2)
    #[error("Invalid capacity: must be power of 2")]
    InvalidCapacity,

    /// Alignment error
    #[error("Alignment error")]
    AlignmentError,
}

/// Result type for ultra lock-free operations
pub type UltraResult<T> = Result<T, UltraError>;

/// Cache line size for optimal performance
pub const CACHE_LINE_SIZE: usize = 64;

/// Maximum supported capacity for ultra structures
pub const MAX_ULTRA_CAPACITY: usize = 1 << 20; // 1M elements

/// Validates that capacity is power of 2 and within limits
///
/// # Errors
/// Returns `UltraError::InvalidCapacity` if capacity is 0, not power of 2, or exceeds maximum
pub const fn validate_capacity(capacity: usize) -> UltraResult<usize> {
    if capacity == 0 || capacity > MAX_ULTRA_CAPACITY {
        return Err(UltraError::InvalidCapacity);
    }

    // Check if power of 2
    if capacity & (capacity - 1) != 0 {
        return Err(UltraError::InvalidCapacity);
    }

    Ok(capacity)
}

/// Rounds up to next power of 2
#[must_use]
pub const fn next_power_of_2(mut n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    n -= 1;
    n |= n >> 1_i32;
    n |= n >> 2_i32;
    n |= n >> 4_i32;
    n |= n >> 8_i32;
    n |= n >> 16_i32;
    n |= n >> 32_i32;
    n + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_capacity() {
        assert!(validate_capacity(0).is_err());
        assert!(validate_capacity(1).is_ok());
        assert!(validate_capacity(2).is_ok());
        assert!(validate_capacity(3).is_err());
        assert!(validate_capacity(4).is_ok());
        assert!(validate_capacity(1024).is_ok());
        assert!(validate_capacity(1025).is_err());
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(1023), 1024);
    }
}
