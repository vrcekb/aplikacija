//! Memory Cache Implementation
//!
//! Production-ready in-memory LRU cache for `TallyIO` financial data.
//! Provides ultra-fast L1 caching with automatic expiration and size limits.

use async_trait::async_trait;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "cache")]
use lru::LruCache;

use crate::{
    config::CacheConfig,
    error::{DataStorageError, DataStorageResult},
};

use super::safe_ratio;

use super::Cache;

/// Internal cache statistics
#[derive(Debug, Clone)]
struct InternalCacheStats {
    hits: u64,
    misses: u64,
    sets: u64,
    deletes: u64,
    avg_latency_us: u64,
}

impl InternalCacheStats {
    const fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            avg_latency_us: 0,
        }
    }
}

/// Cache entry with expiration
#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    expires_at: Option<u64>, // Unix timestamp in seconds
}

impl CacheEntry {
    /// Create a new cache entry
    fn new(data: Vec<u8>, ttl_seconds: Option<u64>) -> Self {
        let expires_at = ttl_seconds.map(|ttl| {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs()
                + ttl
        });

        Self { data, expires_at }
    }

    /// Check if entry has expired
    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_secs();
            now >= expires_at
        } else {
            false
        }
    }
}

/// In-memory LRU cache implementation
#[derive(Debug)]
pub struct MemoryCache {
    #[cfg(feature = "cache")]
    cache: Arc<parking_lot::Mutex<LruCache<String, CacheEntry>>>,
    #[allow(dead_code)]
    config: CacheConfig,
    stats: Arc<parking_lot::Mutex<InternalCacheStats>>,
}

impl MemoryCache {
    /// Create a new memory cache instance
    ///
    /// # Errors
    ///
    /// Returns `DataStorageError::Configuration` if cache configuration is invalid.
    pub fn new(config: CacheConfig) -> DataStorageResult<Self> {
        #[cfg(feature = "cache")]
        {
            // Calculate approximate number of entries based on memory limit
            let avg_entry_size = 1024; // Assume 1KB average entry size
            let max_entries = config.memory_cache_size_bytes / avg_entry_size;

            let max_entries_usize = usize::try_from(max_entries.max(1)).map_err(|e| {
                DataStorageError::configuration(format!("Cache size conversion error: {e}"))
            })?;

            let capacity = NonZeroUsize::new(max_entries_usize)
                .ok_or_else(|| DataStorageError::configuration("Invalid cache size".to_string()))?;

            let cache = Arc::new(parking_lot::Mutex::new(LruCache::new(capacity)));

            tracing::info!(
                "Memory cache created with capacity: {} entries",
                max_entries
            );

            Ok(Self {
                cache,
                config,
                stats: Arc::new(parking_lot::Mutex::new(InternalCacheStats::new())),
            })
        }

        #[cfg(not(feature = "cache"))]
        {
            tracing::warn!("Cache feature not enabled, using mock memory cache");
            Ok(Self {
                config,
                stats: Arc::new(parking_lot::Mutex::new(InternalCacheStats::new())),
            })
        }
    }

    /// Clean expired entries from cache
    #[cfg(feature = "cache")]
    fn cleanup_expired(&self) {
        let mut cache = self.cache.lock();
        let mut expired_keys = Vec::new();

        // Collect expired keys
        for (key, entry) in cache.iter() {
            if entry.is_expired() {
                expired_keys.push(key.clone());
            }
        }

        // Remove expired entries
        for key in expired_keys {
            cache.pop(&key);
        }
    }

    /// Record cache operation statistics
    fn record_stats(&self, operation: &str, hit: bool, duration: Duration) {
        let mut stats = self.stats.lock();

        match operation {
            "get" => {
                if hit {
                    stats.hits += 1;
                } else {
                    stats.misses += 1;
                }
            }
            "set" => {
                stats.sets += 1;
            }
            "delete" => {
                stats.deletes += 1;
            }
            _ => {}
        }

        // Track average latency
        let duration_us = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
        stats.avg_latency_us = u64::midpoint(stats.avg_latency_us, duration_us);
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, key: &str) -> DataStorageResult<Option<Vec<u8>>> {
        let start = Instant::now();

        #[cfg(feature = "cache")]
        {
            // Periodic cleanup of expired entries
            if fastrand::f32() < 0.01 {
                // 1% chance to trigger cleanup
                self.cleanup_expired();
            }

            let mut cache = self.cache.lock();
            let result = cache.get(key).and_then(|entry| {
                if entry.is_expired() {
                    None // Entry expired
                } else {
                    Some(entry.data.clone())
                }
            });

            let duration = start.elapsed();
            self.record_stats("get", result.is_some(), duration);

            tracing::trace!(
                "Memory GET {}: {}",
                key,
                if result.is_some() { "HIT" } else { "MISS" }
            );
            Ok(result)
        }

        #[cfg(not(feature = "cache"))]
        {
            let duration = start.elapsed();
            self.record_stats("get", false, duration);

            tracing::trace!("Memory GET {} (mock): MISS", key);
            Ok(None)
        }
    }

    async fn set(
        &self,
        key: &str,
        value: &[u8],
        ttl_seconds: Option<u64>,
    ) -> DataStorageResult<()> {
        let start = Instant::now();

        #[cfg(feature = "cache")]
        {
            let entry = CacheEntry::new(value.to_vec(), ttl_seconds);
            let mut cache = self.cache.lock();
            cache.put(key.to_string(), entry);

            let duration = start.elapsed();
            self.record_stats("set", true, duration);

            tracing::trace!(
                "Memory SET {}: {} bytes, TTL: {:?}",
                key,
                value.len(),
                ttl_seconds
            );
            Ok(())
        }

        #[cfg(not(feature = "cache"))]
        {
            let duration = start.elapsed();
            self.record_stats("set", true, duration);

            tracing::trace!(
                "Memory SET {} (mock): {} bytes, TTL: {:?}",
                key,
                value.len(),
                ttl_seconds
            );
            Ok(())
        }
    }

    async fn delete(&self, key: &str) -> DataStorageResult<bool> {
        let start = Instant::now();

        #[cfg(feature = "cache")]
        {
            let mut cache = self.cache.lock();
            let was_present = cache.pop(key).is_some();

            let duration = start.elapsed();
            self.record_stats("delete", was_present, duration);

            tracing::trace!(
                "Memory DEL {}: {}",
                key,
                if was_present { "DELETED" } else { "NOT_FOUND" }
            );
            Ok(was_present)
        }

        #[cfg(not(feature = "cache"))]
        {
            let duration = start.elapsed();
            self.record_stats("delete", false, duration);

            tracing::trace!("Memory DEL {} (mock): NOT_FOUND", key);
            Ok(false)
        }
    }

    async fn exists(&self, key: &str) -> DataStorageResult<bool> {
        #[cfg(feature = "cache")]
        {
            let cache = self.cache.lock();
            let exists = cache.peek(key).is_some_and(|entry| !entry.is_expired());

            tracing::trace!("Memory EXISTS {}: {}", key, exists);
            Ok(exists)
        }

        #[cfg(not(feature = "cache"))]
        {
            tracing::trace!("Memory EXISTS {} (mock): false", key);
            Ok(false)
        }
    }

    async fn clear(&self) -> DataStorageResult<()> {
        #[cfg(feature = "cache")]
        {
            let mut cache = self.cache.lock();
            cache.clear();

            tracing::info!("Memory cache cleared");
            Ok(())
        }

        #[cfg(not(feature = "cache"))]
        {
            tracing::info!("Memory cache cleared (mock)");
            Ok(())
        }
    }

    async fn stats(&self) -> DataStorageResult<super::CacheStats> {
        let stats = self.stats.lock();
        let cache_stats = super::CacheStats {
            hits: stats.hits,
            misses: stats.misses,
            sets: stats.sets,
            deletes: stats.deletes,
            entries: 0,      // Would need to track cache size
            memory_usage: 0, // Would need to calculate memory usage
            hit_ratio: safe_ratio(stats.hits, stats.hits + stats.misses),
            avg_latency_us: stats.avg_latency_us,
        };
        Ok(cache_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_cache_basic_operations() -> DataStorageResult<()> {
        let config = CacheConfig::default();
        let cache = MemoryCache::new(config)?;

        // Test set and get
        cache.set("test_key", b"test_value", None).await?;
        let result = cache.get("test_key").await?;
        assert_eq!(result, Some(b"test_value".to_vec()));

        // Test delete
        let deleted = cache.delete("test_key").await?;
        assert!(deleted);

        let result = cache.get("test_key").await?;
        assert_eq!(result, None);

        Ok(())
    }

    #[tokio::test]
    async fn test_memory_cache_expiration() -> DataStorageResult<()> {
        let config = CacheConfig::default();
        let cache = MemoryCache::new(config)?;

        // Set with 1 second TTL
        cache.set("expire_key", b"expire_value", Some(1)).await?;

        // Should exist immediately
        let result = cache.get("expire_key").await?;
        assert_eq!(result, Some(b"expire_value".to_vec()));

        // Wait for expiration
        tokio::time::sleep(Duration::from_secs(2)).await;

        // Should be expired
        let result = cache.get("expire_key").await?;
        assert_eq!(result, None);

        Ok(())
    }
}
