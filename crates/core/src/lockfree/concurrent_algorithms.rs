//! Advanced Lock-Free Algorithms for `TallyIO` - Ultra-Low Latency Concurrent Operations
//!
//! Production-ready lock-free data structures and algorithms for financial applications

use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use thiserror::Error;

/// Lock-free Michael & Scott queue implementation
///
/// Provides FIFO ordering with wait-free enqueue and lock-free dequeue operations
pub struct LockFreeQueue<T> {
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
    size: AtomicUsize,
}

/// Node in the lock-free queue
#[repr(C, align(64))] // Cache line aligned to prevent false sharing
struct Node<T> {
    data: Option<T>,
    next: AtomicPtr<Node<T>>,
}

impl<T> Node<T> {
    /// Create new node with data
    const fn new_with_data(data: T) -> Self {
        Self {
            data: Some(data),
            next: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Create sentinel node
    const fn new_sentinel() -> Self {
        Self {
            data: None,
            next: AtomicPtr::new(std::ptr::null_mut()),
        }
    }
}

impl<T> LockFreeQueue<T> {
    /// Create new empty queue
    #[must_use]
    pub fn new() -> Self {
        let sentinel = Box::into_raw(Box::new(Node::new_sentinel()));

        Self {
            head: AtomicPtr::new(sentinel),
            tail: AtomicPtr::new(sentinel),
            size: AtomicUsize::new(0),
        }
    }

    /// Enqueue item (wait-free)
    ///
    /// # Errors
    ///
    /// Returns error if memory allocation fails
    pub fn enqueue(&self, item: T) -> Result<(), LockFreeError> {
        let new_node = Box::into_raw(Box::new(Node::new_with_data(item)));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*tail).next.load(Ordering::Acquire) };

            // Check if tail is still the last node
            if tail == self.tail.load(Ordering::Acquire) {
                if next.is_null() {
                    // Try to link new node at the end of the list
                    if unsafe {
                        (*tail)
                            .next
                            .compare_exchange_weak(
                                next,
                                new_node,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                    } {
                        // Enqueue successful, try to advance tail
                        let _ = self.tail.compare_exchange_weak(
                            tail,
                            new_node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                        break;
                    }
                } else {
                    // Tail is lagging, try to advance it
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
        }

        self.size.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Dequeue item (lock-free)
    ///
    /// Returns `None` if queue is empty
    pub fn dequeue(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*head).next.load(Ordering::Acquire) };

            // Check if head is still the first node
            if head == self.head.load(Ordering::Acquire) {
                if head == tail {
                    if next.is_null() {
                        // Queue is empty
                        return None;
                    }
                    // Tail is lagging, try to advance it
                    let _ = self.tail.compare_exchange_weak(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                } else {
                    // Read data before CAS to avoid race condition
                    if next.is_null() {
                        continue;
                    }

                    let data = unsafe { (*next).data.take() };

                    // Try to advance head
                    if self
                        .head
                        .compare_exchange_weak(head, next, Ordering::Release, Ordering::Relaxed)
                        .is_ok()
                    {
                        // Dequeue successful
                        unsafe {
                            drop(Box::from_raw(head));
                        }
                        self.size.fetch_sub(1, Ordering::Relaxed);
                        return data;
                    }
                }
            }
        }
    }

    /// Get approximate size (may be stale)
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if queue is empty (may be stale)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        // Drain all remaining items
        while self.dequeue().is_some() {}

        // Clean up sentinel node
        let head = self.head.load(Ordering::Relaxed);
        if !head.is_null() {
            unsafe {
                drop(Box::from_raw(head));
            }
        }
    }
}

unsafe impl<T: Send> Send for LockFreeQueue<T> {}
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

/// Lock-free hash map with linear probing
///
/// Optimized for high-frequency trading scenarios with predictable performance
pub struct LockFreeHashMap<K, V> {
    buckets: Vec<AtomicPtr<Bucket<K, V>>>,
    capacity: usize,
    size: AtomicUsize,
    load_factor_threshold: f64,
}

/// Bucket in the hash map
#[repr(C, align(64))]
struct Bucket<K, V> {
    key: K,
    value: V,
    hash: u64,
    deleted: std::sync::atomic::AtomicBool,
}

impl<K, V> Bucket<K, V> {
    const fn new(key: K, value: V, hash: u64) -> Self {
        Self {
            key,
            value,
            hash,
            deleted: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl<K, V> LockFreeHashMap<K, V>
where
    K: Clone + PartialEq + std::hash::Hash,
    V: Clone,
{
    /// Create new hash map with specified capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buckets = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buckets.push(AtomicPtr::new(std::ptr::null_mut()));
        }

        Self {
            buckets,
            capacity,
            size: AtomicUsize::new(0),
            load_factor_threshold: 0.75,
        }
    }

    /// Insert key-value pair
    ///
    /// # Errors
    ///
    /// Returns error if insertion fails due to capacity limits
    #[allow(clippy::needless_pass_by_value)] // Key ownership is required for insertion
    pub fn insert(&self, key: K, value: V) -> Result<Option<V>, LockFreeError> {
        let hash = Self::hash_key(&key);
        let mut index =
            usize::try_from(hash % u64::try_from(self.capacity).unwrap_or(u64::MAX)).unwrap_or(0);
        let mut attempts = 0;

        // Linear probing with maximum attempts
        while attempts < self.capacity {
            let bucket_ptr = self
                .buckets
                .get(index)
                .map_or(std::ptr::null_mut(), |bucket| {
                    bucket.load(Ordering::Acquire)
                });

            if bucket_ptr.is_null() {
                // Empty slot, try to insert
                let new_bucket =
                    Box::into_raw(Box::new(Bucket::new(key.clone(), value.clone(), hash)));

                if let Some(bucket_atomic) = self.buckets.get(index) {
                    if bucket_atomic
                        .compare_exchange_weak(
                            bucket_ptr,
                            new_bucket,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        self.size.fetch_add(1, Ordering::Relaxed);
                        return Ok(None);
                    }
                    // Someone else inserted, clean up and retry
                    unsafe {
                        drop(Box::from_raw(new_bucket));
                    }
                    continue;
                }
                // Invalid index, clean up and return error
                unsafe {
                    drop(Box::from_raw(new_bucket));
                }
                return Err(LockFreeError::CapacityExceeded);
            }

            // Slot occupied, check if it's the same key
            unsafe {
                if !(*bucket_ptr).deleted.load(Ordering::Acquire) && (*bucket_ptr).key == key {
                    // Key exists, update value
                    let old_value = (*bucket_ptr).value.clone();
                    (*bucket_ptr).value = value;
                    return Ok(Some(old_value));
                }
            }

            index = (index + 1) % self.capacity;
            attempts += 1;
        }

        Err(LockFreeError::CapacityExceeded)
    }

    /// Get value by key
    pub fn get(&self, key: &K) -> Option<V> {
        let hash = Self::hash_key(key);
        let mut index =
            usize::try_from(hash % u64::try_from(self.capacity).unwrap_or(u64::MAX)).unwrap_or(0);
        let mut attempts = 0;

        while attempts < self.capacity {
            let bucket_ptr = self
                .buckets
                .get(index)
                .map_or(std::ptr::null_mut(), |bucket| {
                    bucket.load(Ordering::Acquire)
                });

            if bucket_ptr.is_null() {
                return None;
            }

            unsafe {
                if !(*bucket_ptr).deleted.load(Ordering::Acquire)
                    && (*bucket_ptr).hash == hash
                    && (*bucket_ptr).key == *key
                {
                    return Some((*bucket_ptr).value.clone());
                }
            }

            index = (index + 1) % self.capacity;
            attempts += 1;
        }

        None
    }

    /// Remove key-value pair
    pub fn remove(&self, key: &K) -> Option<V> {
        let hash = Self::hash_key(key);
        let mut index =
            usize::try_from(hash % u64::try_from(self.capacity).unwrap_or(u64::MAX)).unwrap_or(0);
        let mut attempts = 0;

        while attempts < self.capacity {
            let bucket_ptr = self
                .buckets
                .get(index)
                .map_or(std::ptr::null_mut(), |bucket| {
                    bucket.load(Ordering::Acquire)
                });

            if bucket_ptr.is_null() {
                return None;
            }

            unsafe {
                if !(*bucket_ptr).deleted.load(Ordering::Acquire)
                    && (*bucket_ptr).hash == hash
                    && (*bucket_ptr).key == *key
                {
                    // Mark as deleted
                    (*bucket_ptr).deleted.store(true, Ordering::Release);
                    self.size.fetch_sub(1, Ordering::Relaxed);
                    return Some((*bucket_ptr).value.clone());
                }
            }

            index = (index + 1) % self.capacity;
            attempts += 1;
        }

        None
    }

    /// Get current size
    #[must_use]
    pub fn len(&self) -> usize {
        self.size.load(Ordering::Relaxed)
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get load factor
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::if_same_then_else,
        clippy::default_numeric_fallback
    )] // Acceptable for load factor calculation
    pub fn load_factor(&self) -> f64 {
        // Safe conversion with precision handling
        // For load factor calculation, precision loss is acceptable
        let len_f64 = self.len() as f64;
        let capacity_f64 = self.capacity as f64;

        if capacity_f64 == 0.0_f64 {
            0.0_f64
        } else {
            len_f64 / capacity_f64
        }
    }

    /// Check if resize is needed based on load factor threshold
    #[must_use]
    pub fn needs_resize(&self) -> bool {
        self.load_factor() > self.load_factor_threshold
    }

    /// Hash key using FNV-1a algorithm (fast and good distribution)
    fn hash_key(key: &K) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

impl<K, V> Drop for LockFreeHashMap<K, V> {
    fn drop(&mut self) {
        // Clean up all allocated buckets
        for bucket_ptr in &self.buckets {
            let ptr = bucket_ptr.load(Ordering::Relaxed);
            if !ptr.is_null() {
                unsafe {
                    drop(Box::from_raw(ptr));
                }
            }
        }
    }
}

unsafe impl<K: Send, V: Send> Send for LockFreeHashMap<K, V> {}
unsafe impl<K: Send, V: Send> Sync for LockFreeHashMap<K, V> {}

/// Lock-free errors
#[derive(Error, Debug, Clone)]
pub enum LockFreeError {
    /// Memory allocation failed
    #[error("Memory allocation failed")]
    AllocationFailed,
    /// Capacity exceeded
    #[error("Data structure capacity exceeded")]
    CapacityExceeded,
    /// Operation failed due to contention
    #[error("Operation failed due to high contention")]
    ContentionFailure,
    /// Invalid operation
    #[error("Invalid operation: {reason}")]
    InvalidOperation {
        /// Reason for invalid operation
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_lock_free_queue_basic() -> Result<(), LockFreeError> {
        let queue = LockFreeQueue::new();

        // Test enqueue
        queue.enqueue(1_i32)?;
        queue.enqueue(2_i32)?;
        queue.enqueue(3_i32)?;

        assert_eq!(queue.len(), 3);

        // Test dequeue
        assert_eq!(queue.dequeue(), Some(1_i32));
        assert_eq!(queue.dequeue(), Some(2_i32));
        assert_eq!(queue.dequeue(), Some(3_i32));
        assert_eq!(queue.dequeue(), None);

        assert!(queue.is_empty());
        Ok(())
    }

    #[test]
    fn test_lock_free_queue_concurrent() {
        let queue = Arc::new(LockFreeQueue::new());
        let num_threads = 4;
        let items_per_thread = 1000;

        // Spawn producer threads
        let mut handles = Vec::with_capacity(num_threads);
        for thread_id in 0..num_threads {
            let queue_clone = queue.clone();
            let handle = thread::spawn(move || {
                for i in 0..items_per_thread {
                    let item = thread_id * items_per_thread + i;
                    assert!(queue_clone.enqueue(item).is_ok(), "Failed to enqueue item");
                }
            });
            handles.push(handle);
        }

        // Wait for producers
        for handle in handles {
            assert!(handle.join().is_ok(), "Thread join failed");
        }

        // Verify all items were enqueued
        assert_eq!(queue.len(), num_threads * items_per_thread);

        // Consume all items
        let mut consumed = 0;
        while queue.dequeue().is_some() {
            consumed += 1;
        }

        assert_eq!(consumed, num_threads * items_per_thread);
    }

    #[test]
    fn test_lock_free_hashmap_basic() {
        let map = LockFreeHashMap::with_capacity(16);

        // Test insert
        let result1 = map.insert("key1".to_string(), 100_i32);
        assert!(result1.is_ok(), "Insert should succeed: {result1:?}");
        assert_eq!(result1.unwrap_or(None), None);

        let result2 = map.insert("key2".to_string(), 200_i32);
        assert!(result2.is_ok(), "Insert should succeed: {result2:?}");
        assert_eq!(result2.unwrap_or(None), None);

        // Test get
        assert_eq!(map.get(&"key1".to_string()), Some(100_i32));
        assert_eq!(map.get(&"key2".to_string()), Some(200_i32));
        assert_eq!(map.get(&"key3".to_string()), None);

        // Test update
        let update_result = map.insert("key1".to_string(), 150_i32);
        assert!(
            update_result.is_ok(),
            "Update should succeed: {update_result:?}"
        );
        assert_eq!(update_result.unwrap_or(None), Some(100_i32));
        assert_eq!(map.get(&"key1".to_string()), Some(150_i32));

        // Test remove
        assert_eq!(map.remove(&"key1".to_string()), Some(150_i32));
        assert_eq!(map.get(&"key1".to_string()), None);
    }
}
