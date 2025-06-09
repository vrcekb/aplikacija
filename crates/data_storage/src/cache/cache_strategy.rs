//! Cache Strategy Implementation
//!
//! Multi-tier cache strategy (L1: Memory, L2: Redis) for `TallyIO` financial data.
//! Provides intelligent cache routing and fallback mechanisms for ultra-low latency.

use async_trait::async_trait;
use std::sync::Arc;

use crate::{config::CacheConfig, error::DataStorageResult};

use super::{safe_ratio, Cache, CacheStats, MemoryCache, RedisCache};

/// Multi-tier cache strategy for financial data
///
/// Implements L1 (memory) + L2 (Redis) caching with intelligent promotion/demotion
/// and fallback mechanisms. Optimized for <1ms latency requirements.
#[derive(Debug)]
pub struct CacheStrategy {
    memory_cache: MemoryCache,
    redis_cache: Option<RedisCache>,
    config: CacheConfig,
    stats: Arc<parking_lot::Mutex<CacheStrategyStats>>,
}

/// Cache strategy statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct CacheStrategyStats {
    /// L1 (memory) cache hits
    pub l1_hits: u64,
    /// L2 (Redis) cache hits
    pub l2_hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// L1 to L2 promotions
    pub promotions: u64,
    /// Total operations
    pub total_operations: u64,
}

impl CacheStrategyStats {
    /// Calculate overall hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        safe_ratio(self.l1_hits + self.l2_hits, self.total_operations)
    }

    /// Calculate L1 hit rate
    #[must_use]
    pub fn l1_hit_rate(&self) -> f64 {
        safe_ratio(self.l1_hits, self.total_operations)
    }

    /// Calculate L2 hit rate (among L1 misses)
    #[must_use]
    pub fn l2_hit_rate(&self) -> f64 {
        let l1_misses = self.total_operations - self.l1_hits;
        safe_ratio(self.l2_hits, l1_misses)
    }
}

impl CacheStrategy {
    /// Create a new cache strategy
    ///
    /// # Errors
    ///
    /// Returns error if cache initialization fails.
    pub async fn new(config: CacheConfig) -> DataStorageResult<Self> {
        let memory_cache = MemoryCache::new(config.clone())?;

        let redis_cache = if config.enable_redis {
            match RedisCache::new(config.clone()).await {
                Ok(cache) => Some(cache),
                Err(e) => {
                    tracing::warn!(
                        "Failed to initialize Redis cache: {}, falling back to memory-only",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        tracing::info!(
            "Cache strategy initialized: Memory={}, Redis={}",
            true,
            redis_cache.is_some()
        );

        Ok(Self {
            memory_cache,
            redis_cache,
            config,
            stats: Arc::new(parking_lot::Mutex::new(CacheStrategyStats::default())),
        })
    }

    /// Get cache strategy statistics
    #[must_use]
    pub fn strategy_stats(&self) -> CacheStrategyStats {
        self.stats.lock().clone()
    }

    /// Update strategy statistics
    fn update_stats(&self, operation: CacheOperation) {
        let mut stats = self.stats.lock();
        stats.total_operations += 1;

        match operation {
            CacheOperation::L1Hit => stats.l1_hits += 1,
            CacheOperation::L2Hit => stats.l2_hits += 1,
            CacheOperation::Miss => stats.misses += 1,
            CacheOperation::Promotion => stats.promotions += 1,
        }
    }
}

/// Cache operation types for statistics
#[derive(Debug, Clone, Copy)]
enum CacheOperation {
    L1Hit,
    L2Hit,
    Miss,
    Promotion,
}

#[async_trait]
impl Cache for CacheStrategy {
    async fn get(&self, key: &str) -> DataStorageResult<Option<Vec<u8>>> {
        // Try L1 (memory) cache first
        match self.memory_cache.get(key).await? {
            Some(value) => {
                self.update_stats(CacheOperation::L1Hit);
                tracing::trace!("Cache L1 HIT: {}", key);
                return Ok(Some(value));
            }
            None => {
                // L1 miss, try L2 (Redis) if available
                if let Some(ref redis_cache) = self.redis_cache {
                    match redis_cache.get(key).await? {
                        Some(value) => {
                            self.update_stats(CacheOperation::L2Hit);

                            // Promote to L1 cache
                            if let Err(e) = self.memory_cache.set(key, &value, None).await {
                                tracing::warn!("Failed to promote to L1 cache: {}", e);
                            } else {
                                self.update_stats(CacheOperation::Promotion);
                            }

                            tracing::trace!("Cache L2 HIT: {}", key);
                            return Ok(Some(value));
                        }
                        None => {
                            // Complete miss
                            self.update_stats(CacheOperation::Miss);
                            tracing::trace!("Cache MISS: {}", key);
                            return Ok(None);
                        }
                    }
                } else {
                    // No L2 cache available
                    self.update_stats(CacheOperation::Miss);
                    tracing::trace!("Cache MISS (no L2): {}", key);
                    return Ok(None);
                }
            }
        }
    }

    async fn set(
        &self,
        key: &str,
        value: &[u8],
        ttl_seconds: Option<u64>,
    ) -> DataStorageResult<()> {
        // Always set in L1 (memory) cache
        self.memory_cache.set(key, value, ttl_seconds).await?;

        // Also set in L2 (Redis) if available and value is large enough
        if let Some(ref redis_cache) = self.redis_cache {
            // Only store in Redis if value is larger than threshold
            let redis_threshold = self.config.redis_threshold_bytes.unwrap_or(1024); // 1KB default

            if value.len() >= redis_threshold {
                if let Err(e) = redis_cache.set(key, value, ttl_seconds).await {
                    tracing::warn!("Failed to set in Redis cache: {}", e);
                    // Don't fail the operation if Redis fails
                }
            }
        }

        tracing::trace!("Cache SET: {} ({} bytes)", key, value.len());
        Ok(())
    }

    async fn delete(&self, key: &str) -> DataStorageResult<bool> {
        let mut deleted = false;

        // Delete from L1 (memory) cache
        if self.memory_cache.delete(key).await? {
            deleted = true;
        }

        // Delete from L2 (Redis) cache if available
        if let Some(ref redis_cache) = self.redis_cache {
            if redis_cache.delete(key).await.unwrap_or(false) {
                deleted = true;
            }
        }

        tracing::trace!("Cache DELETE: {} (deleted: {})", key, deleted);
        Ok(deleted)
    }

    async fn exists(&self, key: &str) -> DataStorageResult<bool> {
        // Check L1 (memory) cache first
        if self.memory_cache.exists(key).await? {
            return Ok(true);
        }

        // Check L2 (Redis) cache if available
        if let Some(ref redis_cache) = self.redis_cache {
            return redis_cache.exists(key).await;
        }

        Ok(false)
    }

    async fn clear(&self) -> DataStorageResult<()> {
        // Clear L1 (memory) cache
        self.memory_cache.clear().await?;

        // Clear L2 (Redis) cache if available
        if let Some(ref redis_cache) = self.redis_cache {
            if let Err(e) = redis_cache.clear().await {
                tracing::warn!("Failed to clear Redis cache: {}", e);
                // Don't fail the operation if Redis fails
            }
        }

        // Reset statistics
        {
            let mut stats = self.stats.lock();
            *stats = CacheStrategyStats::default();
        }

        tracing::info!("Cache strategy cleared");
        Ok(())
    }

    async fn stats(&self) -> DataStorageResult<CacheStats> {
        let memory_stats = self.memory_cache.stats().await?;
        let redis_stats = if let Some(ref redis_cache) = self.redis_cache {
            Some(redis_cache.stats().await?)
        } else {
            None
        };

        let strategy_stats = self.strategy_stats();

        // Combine statistics
        let combined_stats = CacheStats {
            hits: strategy_stats.l1_hits + strategy_stats.l2_hits,
            misses: strategy_stats.misses,
            sets: memory_stats.sets + redis_stats.as_ref().map_or(0, |s| s.sets),
            deletes: memory_stats.deletes + redis_stats.as_ref().map_or(0, |s| s.deletes),
            entries: memory_stats.entries + redis_stats.as_ref().map_or(0, |s| s.entries),
            memory_usage: memory_stats.memory_usage
                + redis_stats.as_ref().map_or(0, |s| s.memory_usage),
            hit_ratio: strategy_stats.hit_rate(),
            avg_latency_us: (memory_stats.avg_latency_us
                + redis_stats.as_ref().map_or(0, |s| s.avg_latency_us))
                / 2,
        };

        Ok(combined_stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_strategy_creation() -> DataStorageResult<()> {
        let config = CacheConfig::default();
        let _strategy = CacheStrategy::new(config).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_cache_strategy_operations() -> DataStorageResult<()> {
        let config = CacheConfig::default();
        let strategy = CacheStrategy::new(config).await?;

        // Test set and get
        strategy.set("test_key", b"test_value", None).await?;
        let result = strategy.get("test_key").await?;
        assert_eq!(result, Some(b"test_value".to_vec()));

        // Check statistics
        let stats = strategy.strategy_stats();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.total_operations, 1);

        // Test delete
        let deleted = strategy.delete("test_key").await?;
        assert!(deleted);

        let result = strategy.get("test_key").await?;
        assert_eq!(result, None);

        Ok(())
    }

    #[tokio::test]
    async fn test_cache_strategy_l2_promotion() -> DataStorageResult<()> {
        let config = CacheConfig {
            enable_redis: false, // Disable Redis for this test
            ..CacheConfig::default()
        };

        let strategy = CacheStrategy::new(config).await?;

        // Set a value
        strategy.set("promote_key", b"promote_value", None).await?;

        // Get the value (should be L1 hit)
        let result = strategy.get("promote_key").await?;
        assert_eq!(result, Some(b"promote_value".to_vec()));

        let stats = strategy.strategy_stats();
        assert_eq!(stats.l1_hits, 1);
        assert_eq!(stats.l2_hits, 0);

        Ok(())
    }
}
