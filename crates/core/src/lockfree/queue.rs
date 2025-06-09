//! Lock-free queue implementations for ultra-low latency trading
//!
//! Provides SPSC (Single Producer Single Consumer) and MPSC (Multi Producer Single Consumer)
//! queues optimized for financial applications with <1ms latency requirements.

use super::{
    CacheAlignedAtomicPtr, CacheAlignedCounter, LockFreeError, LockFreeResult, LockFreeUtils,
    MemoryOrdering,
};
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::sync::atomic::AtomicUsize;

/// Node in the lock-free queue
#[repr(align(64))]
struct QueueNode<T> {
    data: Option<T>,
    next: CacheAlignedAtomicPtr<QueueNode<T>>,
}

impl<T> QueueNode<T> {
    /// Create new node with data
    fn new_with_data(data: T) -> *mut Self {
        let layout = Layout::new::<Self>();
        let ptr = unsafe { alloc(layout).cast::<Self>() };
        if ptr.is_null() {
            return ptr::null_mut();
        }

        unsafe {
            ptr::write(
                ptr,
                Self {
                    data: Some(data),
                    next: CacheAlignedAtomicPtr::new(ptr::null_mut()),
                },
            );
        }

        ptr
    }

    /// Create new empty sentinel node
    fn new_sentinel() -> *mut Self {
        let layout = Layout::new::<Self>();
        let ptr = unsafe { alloc(layout).cast::<Self>() };
        if ptr.is_null() {
            return ptr::null_mut();
        }

        unsafe {
            ptr::write(
                ptr,
                Self {
                    data: None,
                    next: CacheAlignedAtomicPtr::new(ptr::null_mut()),
                },
            );
        }

        ptr
    }

    /// Safely deallocate node
    unsafe fn deallocate(ptr: *mut Self) {
        if !ptr.is_null() {
            ptr::drop_in_place(ptr);
            let layout = Layout::new::<Self>();
            dealloc(ptr.cast::<u8>(), layout);
        }
    }
}

/// Single Producer Single Consumer lock-free queue
///
/// Optimized for ultra-low latency with cache-aligned structures
/// and minimal atomic operations.
pub struct SPSCQueue<T> {
    head: CacheAlignedAtomicPtr<QueueNode<T>>,
    tail: CacheAlignedAtomicPtr<QueueNode<T>>,
    capacity: usize,
    size: AtomicUsize,
    stats: CacheAlignedCounter,
}

impl<T> SPSCQueue<T> {
    /// Create new SPSC queue with specified capacity
    ///
    /// # Errors
    ///
    /// Returns error if capacity is 0 or memory allocation fails
    pub fn new(capacity: usize) -> LockFreeResult<Self> {
        if capacity == 0 {
            return Err(LockFreeError::InvalidCapacity { capacity });
        }

        // Create sentinel node
        let sentinel = QueueNode::new_sentinel();
        if sentinel.is_null() {
            return Err(LockFreeError::AllocationFailed {
                reason: "Failed to allocate sentinel node".to_string(),
            });
        }

        Ok(Self {
            head: CacheAlignedAtomicPtr::new(sentinel),
            tail: CacheAlignedAtomicPtr::new(sentinel),
            capacity,
            size: AtomicUsize::new(0),
            stats: CacheAlignedCounter::new(0),
        })
    }

    /// Enqueue item (producer side)
    ///
    /// # Errors
    ///
    /// Returns error if queue is full or memory allocation fails
    pub fn enqueue(&self, item: T) -> LockFreeResult<()> {
        // Check capacity
        if self.size.load(MemoryOrdering::relaxed()) >= self.capacity {
            return Err(LockFreeError::QueueFull {
                capacity: self.capacity,
            });
        }

        // Create new node
        let new_node = QueueNode::new_with_data(item);
        if new_node.is_null() {
            return Err(LockFreeError::AllocationFailed {
                reason: "Failed to allocate queue node".to_string(),
            });
        }

        // Update tail
        let prev_tail = self.tail.load(MemoryOrdering::acquire());
        unsafe {
            (*prev_tail).next.store(new_node, MemoryOrdering::release());
        }
        self.tail.store(new_node, MemoryOrdering::release());

        // Update size
        self.size.fetch_add(1, MemoryOrdering::relaxed());
        self.stats.increment();

        Ok(())
    }

    /// Dequeue item (consumer side)
    ///
    /// # Errors
    ///
    /// Returns error if queue is empty
    pub fn dequeue(&self) -> LockFreeResult<T> {
        let head = self.head.load(MemoryOrdering::acquire());
        let next = unsafe { (*head).next.load(MemoryOrdering::acquire()) };

        if next.is_null() {
            return Err(LockFreeError::QueueEmpty);
        }

        // Extract data
        let data = unsafe { (*next).data.take() };

        // Update head
        self.head.store(next, MemoryOrdering::release());

        // Deallocate old head
        unsafe {
            QueueNode::deallocate(head);
        }

        // Update size
        self.size.fetch_sub(1, MemoryOrdering::relaxed());

        data.ok_or(LockFreeError::QueueEmpty)
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.load(MemoryOrdering::relaxed()) == 0
    }

    /// Get current queue size
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(MemoryOrdering::relaxed())
    }

    /// Get queue capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get operation statistics
    #[must_use]
    pub fn stats(&self) -> u64 {
        self.stats.get()
    }

    /// Reset statistics
    pub fn reset_stats(&self) -> u64 {
        self.stats.reset()
    }
}

impl<T> Drop for SPSCQueue<T> {
    fn drop(&mut self) {
        // Drain remaining items
        while self.dequeue().is_ok() {
            // Continue draining
        }

        // Deallocate sentinel
        let head = self.head.load(MemoryOrdering::relaxed());
        unsafe {
            QueueNode::deallocate(head);
        }
    }
}

unsafe impl<T: Send> Send for SPSCQueue<T> {}
unsafe impl<T: Send> Sync for SPSCQueue<T> {}

/// Multi Producer Single Consumer lock-free queue
///
/// Uses compare-and-swap operations for thread-safe enqueueing
/// while maintaining single consumer optimization.
pub struct MPSCQueue<T> {
    head: CacheAlignedAtomicPtr<QueueNode<T>>,
    tail: CacheAlignedAtomicPtr<QueueNode<T>>,
    capacity: usize,
    size: AtomicUsize,
    enqueue_stats: CacheAlignedCounter,
    dequeue_stats: CacheAlignedCounter,
}

impl<T> MPSCQueue<T> {
    /// Create new MPSC queue with specified capacity
    ///
    /// # Errors
    ///
    /// Returns error if capacity is 0 or memory allocation fails
    pub fn new(capacity: usize) -> LockFreeResult<Self> {
        if capacity == 0 {
            return Err(LockFreeError::InvalidCapacity { capacity });
        }

        // Create sentinel node
        let sentinel = QueueNode::new_sentinel();
        if sentinel.is_null() {
            return Err(LockFreeError::AllocationFailed {
                reason: "Failed to allocate sentinel node".to_string(),
            });
        }

        Ok(Self {
            head: CacheAlignedAtomicPtr::new(sentinel),
            tail: CacheAlignedAtomicPtr::new(sentinel),
            capacity,
            size: AtomicUsize::new(0),
            enqueue_stats: CacheAlignedCounter::new(0),
            dequeue_stats: CacheAlignedCounter::new(0),
        })
    }

    /// Enqueue item (multi-producer safe)
    ///
    /// # Errors
    ///
    /// Returns error if queue is full or memory allocation fails
    pub fn enqueue(&self, item: T) -> LockFreeResult<()> {
        // Check capacity
        if self.size.load(MemoryOrdering::relaxed()) >= self.capacity {
            return Err(LockFreeError::QueueFull {
                capacity: self.capacity,
            });
        }

        // Create new node
        let new_node = QueueNode::new_with_data(item);
        if new_node.is_null() {
            return Err(LockFreeError::AllocationFailed {
                reason: "Failed to allocate queue node".to_string(),
            });
        }

        // CAS loop to update tail
        loop {
            let tail = self.tail.load(MemoryOrdering::acquire());
            let next = unsafe { (*tail).next.load(MemoryOrdering::acquire()) };

            if next.is_null() {
                // Try to link new node
                if unsafe {
                    (*tail).next.compare_exchange_weak(
                        ptr::null_mut(),
                        new_node,
                        MemoryOrdering::release(),
                        MemoryOrdering::relaxed(),
                    )
                }
                .is_ok()
                {
                    // Successfully linked, now update tail
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        new_node,
                        MemoryOrdering::release(),
                        MemoryOrdering::relaxed(),
                    );
                    break;
                }
            } else {
                // Help advance tail
                let _ = self.tail.compare_exchange_weak(
                    tail,
                    next,
                    MemoryOrdering::release(),
                    MemoryOrdering::relaxed(),
                );
            }
            // Someone else linked, try again
            LockFreeUtils::cpu_pause();
        }

        // Update size and stats
        self.size.fetch_add(1, MemoryOrdering::relaxed());
        self.enqueue_stats.increment();

        Ok(())
    }

    /// Dequeue item (single consumer)
    ///
    /// # Errors
    ///
    /// Returns error if queue is empty
    pub fn dequeue(&self) -> LockFreeResult<T> {
        let head = self.head.load(MemoryOrdering::acquire());
        let next = unsafe { (*head).next.load(MemoryOrdering::acquire()) };

        if next.is_null() {
            return Err(LockFreeError::QueueEmpty);
        }

        // Extract data
        let data = unsafe { (*next).data.take() };

        // Update head
        self.head.store(next, MemoryOrdering::release());

        // Deallocate old head
        unsafe {
            QueueNode::deallocate(head);
        }

        // Update size and stats
        self.size.fetch_sub(1, MemoryOrdering::relaxed());
        self.dequeue_stats.increment();

        data.ok_or(LockFreeError::QueueEmpty)
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.load(MemoryOrdering::relaxed()) == 0
    }

    /// Get current queue size
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(MemoryOrdering::relaxed())
    }

    /// Get queue capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get enqueue statistics
    #[must_use]
    pub fn enqueue_stats(&self) -> u64 {
        self.enqueue_stats.get()
    }

    /// Get dequeue statistics
    #[must_use]
    pub fn dequeue_stats(&self) -> u64 {
        self.dequeue_stats.get()
    }
}

impl<T> Drop for MPSCQueue<T> {
    fn drop(&mut self) {
        // Drain remaining items
        while self.dequeue().is_ok() {
            // Continue draining
        }

        // Deallocate sentinel
        let head = self.head.load(MemoryOrdering::relaxed());
        unsafe {
            QueueNode::deallocate(head);
        }
    }
}

unsafe impl<T: Send> Send for MPSCQueue<T> {}
unsafe impl<T: Send> Sync for MPSCQueue<T> {}
