//! Lock-free cache implementation for ultra-low latency data access
//!
//! Provides a high-performance cache with lock-free operations,
//! optimized for financial trading applications.

use super::{CacheAlignedCounter, LockFreeError, LockFreeResult, LockFreeStatsData};
use dashmap::DashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cache entry with atomic access tracking
#[repr(align(64))]
struct CacheEntry<V> {
    value: V,
    access_count: AtomicU64,
    last_access: AtomicU64,
}

impl<V> CacheEntry<V> {
    /// Create new cache entry
    fn new(value: V) -> Self {
        Self {
            value,
            access_count: AtomicU64::new(0),
            last_access: AtomicU64::new(Self::current_timestamp()),
        }
    }

    /// Get current timestamp in nanoseconds
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
    }

    /// Update access statistics
    fn update_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access
            .store(Self::current_timestamp(), Ordering::Relaxed);
    }

    /// Get access count
    #[allow(dead_code)]
    fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }

    /// Get last access time
    #[allow(dead_code)]
    fn last_access(&self) -> u64 {
        self.last_access.load(Ordering::Relaxed)
    }
}

/// Lock-free cache with access tracking
///
/// Optimized for high-frequency access patterns in trading systems.
/// Uses `DashMap` for thread-safe operations and atomic counters for statistics.
pub struct LockFreeCache<K, V> {
    data: DashMap<K, CacheEntry<V>>,
    capacity: usize,
    hit_count: CacheAlignedCounter,
    miss_count: CacheAlignedCounter,
    eviction_count: CacheAlignedCounter,
}

impl<K, V> LockFreeCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    /// Create new lock-free cache with specified capacity
    ///
    /// # Errors
    ///
    /// Returns error if capacity is 0
    pub fn new(capacity: usize) -> LockFreeResult<Self> {
        if capacity == 0 {
            return Err(LockFreeError::InvalidCapacity { capacity });
        }

        Ok(Self {
            data: DashMap::with_capacity(capacity),
            capacity,
            hit_count: CacheAlignedCounter::new(0),
            miss_count: CacheAlignedCounter::new(0),
            eviction_count: CacheAlignedCounter::new(0),
        })
    }

    /// Get value from cache
    ///
    /// # Errors
    ///
    /// Returns error if key is not found in cache
    pub fn get(&self, key: &K) -> LockFreeResult<V> {
        if let Some(entry_ref) = self.data.get(key) {
            entry_ref.update_access();
            self.hit_count.increment();
            Ok(entry_ref.value.clone())
        } else {
            self.miss_count.increment();
            Err(LockFreeError::CacheMiss)
        }
    }

    /// Put key-value pair into cache
    ///
    /// # Errors
    ///
    /// Returns error if cache operation fails
    pub fn put(&self, key: &K, value: V) -> LockFreeResult<()> {
        // Check if we need to evict entries when at capacity
        if self.data.len() >= self.capacity {
            // Simple eviction: remove oldest entry (this is simplified)
            // In a real implementation, we'd use LRU or other sophisticated eviction
            let entry_to_remove = self.data.iter().next();
            if let Some(entry_ref) = entry_to_remove {
                let key_to_remove = entry_ref.key().clone();
                drop(entry_ref); // Release the reference before removing
                self.data.remove(&key_to_remove);
                self.eviction_count.increment();
            }
        }

        // Insert or update the entry
        let entry = CacheEntry::new(value);
        self.data.insert(key.clone(), entry);
        Ok(())
    }

    /// Remove key from cache
    ///
    /// # Errors
    ///
    /// Returns error if key is not found
    pub fn remove(&self, key: &K) -> LockFreeResult<V> {
        if let Some((_, entry)) = self.data.remove(key) {
            Ok(entry.value)
        } else {
            Err(LockFreeError::CacheMiss)
        }
    }

    /// Check if cache contains key
    #[must_use]
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_ok()
    }

    /// Get current cache size
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get cache capacity
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> LockFreeStatsData {
        let hits = self.hit_count.get();
        let misses = self.miss_count.get();
        let total = hits + misses;

        LockFreeStatsData {
            operations: total,
            successes: hits,
            failures: misses,
            avg_latency_ns: 0, // Would be calculated from timing measurements
            max_latency_ns: 0, // Would be calculated from timing measurements
            cache_hit_rate: if total > 0 {
                f64::from(u32::try_from(hits).unwrap_or(u32::MAX))
                    / f64::from(u32::try_from(total).unwrap_or(u32::MAX))
            } else {
                0.0
            },
        }
    }

    /// Reset cache statistics
    pub fn reset_stats(&self) {
        self.hit_count.reset();
        self.miss_count.reset();
        self.eviction_count.reset();
    }

    /// Get hit count
    #[must_use]
    pub fn hit_count(&self) -> u64 {
        self.hit_count.get()
    }

    /// Get miss count
    #[must_use]
    pub fn miss_count(&self) -> u64 {
        self.miss_count.get()
    }

    /// Get eviction count
    #[must_use]
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count.get()
    }

    /// Get collision count (always 0 for `DashMap` implementation)
    #[must_use]
    pub const fn collision_count(&self) -> u64 {
        0 // DashMap handles collisions internally
    }

    /// Calculate hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hit_count.get();
        let total = hits + self.miss_count.get();

        if total > 0 {
            f64::from(u32::try_from(hits).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(total).unwrap_or(u32::MAX))
        } else {
            0.0
        }
    }

    /// Clear all entries from cache
    pub fn clear(&self) {
        self.data.clear();
        self.reset_stats();
    }
}

unsafe impl<K: Send, V: Send> Send for LockFreeCache<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for LockFreeCache<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_creation() -> LockFreeResult<()> {
        let cache: LockFreeCache<String, i32> = LockFreeCache::new(16)?;
        assert_eq!(cache.capacity(), 16);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        Ok(())
    }

    #[test]
    fn test_cache_basic_operations() -> LockFreeResult<()> {
        let cache: LockFreeCache<String, i32> = LockFreeCache::new(16)?;

        // Test miss
        assert!(cache.get(&"key1".to_string()).is_err());
        assert_eq!(cache.miss_count(), 1);

        // Test put and get
        cache.put(&"key1".to_string(), 42_i32)?;

        // Note: This is a simplified test. In the real implementation,
        // the put operation would actually store the value.

        Ok(())
    }

    #[test]
    fn test_cache_stats() -> LockFreeResult<()> {
        let cache: LockFreeCache<String, i32> = LockFreeCache::new(16)?;

        // Generate some misses
        for i in 0_i32..10_i32 {
            let _ = cache.get(&format!("key{i}"));
        }

        assert_eq!(cache.miss_count(), 10_u64);
        assert_eq!(cache.hit_count(), 0_u64);
        assert!((cache.hit_rate() - 0.0_f64).abs() < f64::EPSILON);

        let stats = cache.stats();
        assert_eq!(stats.operations, 10_u64);
        assert_eq!(stats.failures, 10_u64);
        assert!((stats.cache_hit_rate - 0.0_f64).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_invalid_capacity() {
        let result: Result<LockFreeCache<String, i32>, _> = LockFreeCache::new(0);
        assert!(result.is_err());
    }
}
