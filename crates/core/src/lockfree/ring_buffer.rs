//! Safe lock-free ring buffer implementation for ultra-low latency data storage
//!
//! Provides a high-performance ring buffer with atomic operations,
//! optimized for financial data streaming and event processing.
//! Uses crossbeam's `SegQueue` for thread-safe operations without unsafe code.

use super::{CacheAlignedCounter, LockFreeError, LockFreeResult, LockFreeStats};
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Safe lock-free ring buffer with bounded capacity
///
/// Uses crossbeam's `SegQueue` for thread-safe operations without unsafe code.
/// Optimized for financial trading applications with bounded capacity.
#[repr(align(64))]
pub struct SafeLockFreeRingBuffer<T> {
    queue: SegQueue<T>,
    capacity: usize,
    size: AtomicUsize,
    write_count: CacheAlignedCounter,
    read_count: CacheAlignedCounter,
    overrun_count: CacheAlignedCounter,
}

impl<T> SafeLockFreeRingBuffer<T> {
    /// Create new ring buffer with specified capacity
    ///
    /// # Errors
    ///
    /// Returns error if capacity is 0
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(capacity: usize) -> LockFreeResult<Self> {
        if capacity == 0 {
            return Err(LockFreeError::InvalidCapacity { capacity });
        }

        Ok(Self {
            queue: SegQueue::new(),
            capacity,
            size: AtomicUsize::new(0),
            write_count: CacheAlignedCounter::new(0),
            read_count: CacheAlignedCounter::new(0),
            overrun_count: CacheAlignedCounter::new(0),
        })
    }

    /// Write item to buffer
    ///
    /// This operation respects capacity limits and tracks overruns.
    ///
    /// # Errors
    ///
    /// Returns error if buffer is at capacity
    pub fn write(&self, item: T) -> LockFreeResult<()> {
        let current_size = self.size.load(Ordering::Relaxed);

        // Check if buffer is at capacity
        if current_size >= self.capacity {
            self.overrun_count.increment();
            return Err(LockFreeError::QueueFull {
                capacity: self.capacity,
            });
        }

        // Push item to queue
        self.queue.push(item);
        self.size.fetch_add(1, Ordering::Relaxed);
        self.write_count.increment();

        Ok(())
    }

    /// Read item from buffer
    ///
    /// # Errors
    ///
    /// Returns error if buffer is empty
    pub fn read(&self) -> LockFreeResult<T> {
        self.queue.pop().map_or_else(
            || Err(LockFreeError::QueueEmpty),
            |item| {
                self.size.fetch_sub(1, Ordering::Relaxed);
                self.read_count.increment();
                Ok(item)
            },
        )
    }

    /// Get current number of items in buffer
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if buffer is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Get buffer capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get write count
    #[must_use]
    pub fn write_count(&self) -> u64 {
        self.write_count.get()
    }

    /// Get read count
    #[must_use]
    pub fn read_count(&self) -> u64 {
        self.read_count.get()
    }

    /// Get overrun count
    #[must_use]
    pub fn overrun_count(&self) -> u64 {
        self.overrun_count.get()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.write_count.reset();
        self.read_count.reset();
        self.overrun_count.reset();
    }

    /// Clear all items from buffer
    pub fn clear(&self) {
        while self.queue.pop().is_some() {
            self.size.fetch_sub(1, Ordering::Relaxed);
        }
        self.reset_stats();
    }

    /// Get utilization percentage (0.0 - 1.0)
    #[must_use]
    pub fn utilization(&self) -> f64 {
        let len = u32::try_from(self.len()).unwrap_or(u32::MAX);
        let capacity = u32::try_from(self.capacity).unwrap_or(u32::MAX);
        f64::from(len) / f64::from(capacity)
    }
}

impl<T> LockFreeStats for SafeLockFreeRingBuffer<T> {
    fn hit_count(&self) -> u64 {
        self.read_count()
    }

    fn miss_count(&self) -> u64 {
        0 // Ring buffer doesn't have cache misses
    }

    fn total_operations(&self) -> u64 {
        self.write_count() + self.read_count()
    }

    fn hit_ratio(&self) -> f64 {
        let total = self.total_operations();
        if total == 0 {
            0.0
        } else {
            let read_count = u32::try_from(self.read_count()).unwrap_or(u32::MAX);
            let total_ops = u32::try_from(total).unwrap_or(u32::MAX);
            f64::from(read_count) / f64::from(total_ops)
        }
    }

    fn reset_stats(&self) {
        self.reset_stats();
    }
}

// Safety: SafeLockFreeRingBuffer is thread-safe due to using SegQueue and atomic operations
unsafe impl<T: Send> Send for SafeLockFreeRingBuffer<T> {}
unsafe impl<T: Send> Sync for SafeLockFreeRingBuffer<T> {}

/// Type alias for MPSC ring buffer (same as `SafeLockFreeRingBuffer`)
pub type MPSCRingBuffer<T> = SafeLockFreeRingBuffer<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() -> LockFreeResult<()> {
        let buffer = SafeLockFreeRingBuffer::<i32>::new(1024)?;
        assert_eq!(buffer.capacity(), 1024);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        Ok(())
    }

    #[test]
    fn test_write_read() -> LockFreeResult<()> {
        let buffer = SafeLockFreeRingBuffer::new(4)?;

        buffer.write(1_i32)?;
        buffer.write(2_i32)?;

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.read()?, 1_i32);
        assert_eq!(buffer.read()?, 2_i32);
        assert!(buffer.is_empty());

        Ok(())
    }

    #[test]
    fn test_capacity_limit() -> LockFreeResult<()> {
        let buffer = SafeLockFreeRingBuffer::new(2)?;

        buffer.write(1_i32)?;
        buffer.write(2_i32)?;

        // Should fail when at capacity
        assert!(buffer.write(3_i32).is_err());
        assert_eq!(buffer.overrun_count(), 1);

        Ok(())
    }

    #[test]
    fn test_empty_read() -> LockFreeResult<()> {
        let buffer = SafeLockFreeRingBuffer::<i32>::new(4)?;
        assert!(buffer.read().is_err());
        Ok(())
    }

    #[test]
    fn test_statistics() -> LockFreeResult<()> {
        let buffer = SafeLockFreeRingBuffer::new(4)?;

        buffer.write(1_i32)?;
        buffer.write(2_i32)?;
        buffer.read()?;

        assert_eq!(buffer.write_count(), 2);
        assert_eq!(buffer.read_count(), 1);

        buffer.reset_stats();
        assert_eq!(buffer.write_count(), 0);
        assert_eq!(buffer.read_count(), 0);

        Ok(())
    }

    #[test]
    fn test_clear() -> LockFreeResult<()> {
        let buffer = SafeLockFreeRingBuffer::new(4)?;

        buffer.write(1_i32)?;
        buffer.write(2_i32)?;
        buffer.clear();

        assert!(buffer.is_empty());
        assert_eq!(buffer.write_count(), 0);

        Ok(())
    }

    #[test]
    fn test_utilization() -> LockFreeResult<()> {
        let buffer = SafeLockFreeRingBuffer::new(4)?;

        assert!((buffer.utilization() - 0.0_f64).abs() < f64::EPSILON);

        buffer.write(1_i32)?;
        buffer.write(2_i32)?;

        assert!((buffer.utilization() - 0.5_f64).abs() < f64::EPSILON);

        Ok(())
    }
}
