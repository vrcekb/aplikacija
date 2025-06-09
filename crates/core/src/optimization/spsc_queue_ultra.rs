//! Ultra-optimized SPSC Queue - Target: <100ns per operation
//!
//! Production-ready Single Producer Single Consumer queue with cache-aligned
//! memory layout and zero-allocation hot paths for `TallyIO` financial application.

use std::{
    alloc::{alloc, dealloc, Layout},
    mem::{size_of, MaybeUninit},
    sync::atomic::{AtomicUsize, Ordering},
};

use thiserror::Error;

/// SPSC Queue errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum SPSCError {
    /// Queue is full
    #[error("Queue is full")]
    QueueFull,

    /// Queue is empty
    #[error("Queue is empty")]
    QueueEmpty,

    /// Invalid capacity (must be power of 2)
    #[error("Invalid capacity: {capacity} (must be power of 2)")]
    InvalidCapacity {
        /// Invalid capacity value
        capacity: usize,
    },

    /// Memory allocation failed
    #[error("Memory allocation failed")]
    AllocationFailed,

    /// Layout error
    #[error("Layout error: {reason}")]
    LayoutError {
        /// Error reason
        reason: String,
    },
}

/// Result type for SPSC operations
pub type SPSCResult<T> = Result<T, SPSCError>;

/// Cache-aligned SPSC Queue for ultra-low latency operations
///
/// This implementation uses cache-line alignment to prevent false sharing
/// between producer and consumer, achieving <100ns per operation.
#[repr(C, align(64))]
pub struct UltraSPSCQueue<T> {
    // Producer cache line (64 bytes)
    head: AtomicUsize,
    _pad1: [u8; 64 - size_of::<AtomicUsize>()],

    // Consumer cache line (64 bytes)
    tail: AtomicUsize,
    _pad2: [u8; 64 - size_of::<AtomicUsize>()],

    // Metadata (shared, read-only after initialization)
    capacity: usize,
    mask: usize,

    // Data buffer (cache-aligned)
    buffer: *mut MaybeUninit<T>,
}

// Safety: UltraSPSCQueue is Send + Sync for T: Send
unsafe impl<T: Send> Send for UltraSPSCQueue<T> {}
unsafe impl<T: Send> Sync for UltraSPSCQueue<T> {}

impl<T> UltraSPSCQueue<T> {
    /// Create new ultra-optimized SPSC queue
    ///
    /// # Arguments
    ///
    /// * `capacity` - Queue capacity (must be power of 2)
    ///
    /// # Errors
    ///
    /// Returns error if capacity is not power of 2 or allocation fails
    ///
    /// # Examples
    ///
    /// ```
    /// use tallyio_core::optimization::UltraSPSCQueue;
    ///
    /// let queue = UltraSPSCQueue::<u64>::new(1024)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(capacity: usize) -> SPSCResult<Self> {
        if !capacity.is_power_of_two() || capacity == 0 {
            return Err(SPSCError::InvalidCapacity { capacity });
        }

        let layout =
            Layout::array::<MaybeUninit<T>>(capacity).map_err(|_| SPSCError::LayoutError {
                reason: "Failed to create layout".to_string(),
            })?;

        let buffer = unsafe { alloc(layout).cast::<MaybeUninit<T>>() };
        if buffer.is_null() {
            return Err(SPSCError::AllocationFailed);
        }

        Ok(Self {
            head: AtomicUsize::new(0),
            _pad1: [0; 64 - size_of::<AtomicUsize>()],
            tail: AtomicUsize::new(0),
            _pad2: [0; 64 - size_of::<AtomicUsize>()],
            capacity,
            mask: capacity - 1,
            buffer,
        })
    }

    /// Ultra-fast enqueue operation - Target: <50ns
    ///
    /// # Arguments
    ///
    /// * `item` - Item to enqueue
    ///
    /// # Errors
    ///
    /// Returns `SPSCError::QueueFull` if queue is full
    ///
    /// # Examples
    ///
    /// ```
    /// use tallyio_core::optimization::UltraSPSCQueue;
    ///
    /// let queue = UltraSPSCQueue::<u64>::new(1024)?;
    /// queue.try_enqueue(42)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn try_enqueue(&self, item: T) -> SPSCResult<()> {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & self.mask;

        // Check if queue is full (relaxed load for speed)
        if next_head == self.tail.load(Ordering::Acquire) {
            return Err(SPSCError::QueueFull);
        }

        // Write data - safe because we checked capacity
        unsafe {
            (*self.buffer.add(head)).write(item);
        }

        // Publish the write with release ordering
        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Ultra-fast dequeue operation - Target: <50ns
    ///
    /// # Errors
    ///
    /// Returns `SPSCError::QueueEmpty` if queue is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use tallyio_core::optimization::UltraSPSCQueue;
    ///
    /// let queue = UltraSPSCQueue::<u64>::new(1024)?;
    /// queue.try_enqueue(42)?;
    /// let item = queue.try_dequeue()?;
    /// assert_eq!(item, 42);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn try_dequeue(&self) -> SPSCResult<T> {
        let tail = self.tail.load(Ordering::Relaxed);

        // Check if queue is empty
        if tail == self.head.load(Ordering::Acquire) {
            return Err(SPSCError::QueueEmpty);
        }

        // Read data - safe because we checked availability
        let item = unsafe { (*self.buffer.add(tail)).assume_init_read() };

        // Update tail with release ordering
        self.tail.store((tail + 1) & self.mask, Ordering::Release);
        Ok(item)
    }

    /// Check if queue is empty (non-blocking)
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tail.load(Ordering::Relaxed) == self.head.load(Ordering::Relaxed)
    }

    /// Check if queue is full (non-blocking)
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & self.mask;
        next_head == self.tail.load(Ordering::Relaxed)
    }

    /// Get current queue length (approximate)
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        (head.wrapping_sub(tail)) & self.mask
    }

    /// Get queue capacity
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> Drop for UltraSPSCQueue<T> {
    fn drop(&mut self) {
        // Drain remaining items to prevent memory leaks
        while self.try_dequeue().is_ok() {}

        // Deallocate buffer
        unsafe {
            if let Ok(layout) = Layout::array::<MaybeUninit<T>>(self.capacity) {
                dealloc(self.buffer.cast::<u8>(), layout);
            }
            // If layout creation fails during drop, we can't safely deallocate
            // This should never happen if the queue was properly constructed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_new_valid_capacity() -> SPSCResult<()> {
        let queue = UltraSPSCQueue::<u64>::new(1024)?;
        assert_eq!(queue.capacity(), 1024);
        assert!(queue.is_empty());
        assert!(!queue.is_full());
        Ok(())
    }

    #[test]
    fn test_new_invalid_capacity() {
        assert!(UltraSPSCQueue::<u64>::new(0).is_err());
        assert!(UltraSPSCQueue::<u64>::new(1023).is_err()); // Not power of 2
    }

    #[test]
    fn test_enqueue_dequeue() -> SPSCResult<()> {
        let queue = UltraSPSCQueue::<u64>::new(1024)?;

        queue.try_enqueue(42)?;
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        let item = queue.try_dequeue()?;
        assert_eq!(item, 42);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        Ok(())
    }

    #[test]
    fn test_queue_full() -> SPSCResult<()> {
        let queue = UltraSPSCQueue::<u64>::new(4)?; // Small queue

        // Fill queue (capacity - 1 due to ring buffer implementation)
        for i in 0..3 {
            queue.try_enqueue(i)?;
        }

        // Queue should be full now
        assert!(queue.is_full());
        assert!(queue.try_enqueue(999).is_err());

        Ok(())
    }

    #[test]
    fn test_queue_empty() -> SPSCResult<()> {
        let queue = UltraSPSCQueue::<u64>::new(1024)?;

        assert!(queue.is_empty());
        assert!(queue.try_dequeue().is_err());

        Ok(())
    }

    #[test]
    fn test_ultra_performance() -> SPSCResult<()> {
        let queue = UltraSPSCQueue::<u64>::new(1024)?;

        let start = Instant::now();
        for i in 0..1000 {
            queue.try_enqueue(i)?;
            assert_eq!(queue.try_dequeue()?, i);
        }
        let elapsed = start.elapsed();

        // Target: <100ns per operation pair (enqueue + dequeue)
        let ns_per_op = elapsed.as_nanos() / 2000; // 2 ops per iteration
        println!("Performance: {ns_per_op}ns per operation");

        // This is a performance test - should pass in optimized builds
        // In debug builds, this might be slower due to lack of optimizations
        #[cfg(not(debug_assertions))]
        assert!(
            ns_per_op < 100,
            "Too slow: {}ns per op (target: <100ns)",
            ns_per_op
        );

        Ok(())
    }

    #[test]
    fn test_wraparound() -> SPSCResult<()> {
        let queue = UltraSPSCQueue::<u64>::new(4)?;

        // Fill and empty queue multiple times to test wraparound
        for cycle in 0..10 {
            for i in 0..3 {
                queue.try_enqueue(cycle * 10 + i)?;
            }

            for i in 0..3 {
                let item = queue.try_dequeue()?;
                assert_eq!(item, cycle * 10 + i);
            }

            assert!(queue.is_empty());
        }

        Ok(())
    }
}
