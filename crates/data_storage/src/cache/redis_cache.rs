//! Redis Cache Implementation
//!
//! Production-ready Redis cache implementation for `TallyIO` financial data.
//! Provides ultra-fast caching with connection pooling and error handling.

use async_trait::async_trait;
#[cfg(feature = "cache")]
use deadpool_redis::{Config as RedisConfig, Pool as RedisPool, Runtime};
#[cfg(feature = "cache")]
use redis::AsyncCommands;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

/// Redis cache implementation with connection pooling
pub struct RedisCache {
    #[cfg(feature = "cache")]
    pool: RedisPool,
    config: CacheConfig,
    stats: Arc<parking_lot::Mutex<InternalCacheStats>>,
}

impl std::fmt::Debug for RedisCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisCache")
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish()
    }
}

impl RedisCache {
    /// Create a new Redis cache instance
    ///
    /// # Errors
    ///
    /// Returns `DataStorageError::Configuration` if Redis configuration is invalid,
    /// or `DataStorageError::Connection` if Redis connection fails.
    pub async fn new(config: CacheConfig) -> DataStorageResult<Self> {
        #[cfg(feature = "cache")]
        {
            let pool = Self::create_redis_pool(&config)?;
            Self::verify_redis_connection(&pool).await?;

            Ok(Self {
                pool,
                config,
                stats: Arc::new(parking_lot::Mutex::new(InternalCacheStats::new())),
            })
        }

        #[cfg(not(feature = "cache"))]
        {
            tracing::warn!("Cache feature not enabled, using mock Redis implementation");
            Ok(Self {
                config,
                stats: Arc::new(parking_lot::Mutex::new(InternalCacheStats::new())),
            })
        }
    }

    /// Create optimized Redis connection pool
    #[cfg(feature = "cache")]
    fn create_redis_pool(config: &CacheConfig) -> DataStorageResult<RedisPool> {
        let redis_url = config
            .redis_url
            .as_ref()
            .ok_or_else(|| DataStorageError::configuration("Redis URL not configured"))?;

        let redis_config = RedisConfig::from_url(redis_url);

        let pool = redis_config
            .create_pool(Some(Runtime::Tokio1))
            .map_err(|e| {
                DataStorageError::connection(format!("Failed to create Redis pool: {e}"))
            })?;

        tracing::info!("Redis connection pool created successfully");
        Ok(pool)
    }

    /// Verify Redis connection is working
    #[cfg(feature = "cache")]
    async fn verify_redis_connection(pool: &RedisPool) -> DataStorageResult<()> {
        let mut conn = pool.get().await.map_err(|e| {
            DataStorageError::connection(format!("Failed to get Redis connection: {e}"))
        })?;

        // Use a simple command to test connection
        let _: String = redis::cmd("PING")
            .query_async(&mut *conn)
            .await
            .map_err(|e| DataStorageError::connection(format!("Redis ping failed: {e}")))?;

        tracing::info!("Redis connection verified successfully");
        Ok(())
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
impl Cache for RedisCache {
    async fn get(&self, key: &str) -> DataStorageResult<Option<Vec<u8>>> {
        let start = Instant::now();

        #[cfg(feature = "cache")]
        {
            let mut conn = self.pool.get().await.map_err(|e| {
                DataStorageError::connection(format!("Failed to get Redis connection: {e}"))
            })?;

            let result: Option<Vec<u8>> = conn
                .get(key)
                .await
                .map_err(|e| DataStorageError::cache("redis_get", e.to_string()))?;

            let duration = start.elapsed();
            self.record_stats("get", result.is_some(), duration);

            tracing::trace!(
                "Redis GET {}: {}",
                key,
                if result.is_some() { "HIT" } else { "MISS" }
            );
            Ok(result)
        }

        #[cfg(not(feature = "cache"))]
        {
            let duration = start.elapsed();
            self.record_stats("get", false, duration);

            tracing::trace!("Redis GET {} (mock): MISS", key);
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
            let mut conn = self.pool.get().await.map_err(|e| {
                DataStorageError::connection(format!("Failed to get Redis connection: {e}"))
            })?;

            if let Some(ttl) = ttl_seconds {
                conn.set_ex::<_, _, ()>(key, value, ttl)
                    .await
                    .map_err(|e| DataStorageError::cache("redis_set_ex", e.to_string()))?;
            } else {
                conn.set::<_, _, ()>(key, value)
                    .await
                    .map_err(|e| DataStorageError::cache("redis_set", e.to_string()))?;
            }

            let duration = start.elapsed();
            self.record_stats("set", true, duration);

            tracing::trace!(
                "Redis SET {}: {} bytes, TTL: {:?}",
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
                "Redis SET {} (mock): {} bytes, TTL: {:?}",
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
            let mut conn = self.pool.get().await.map_err(|e| {
                DataStorageError::connection(format!("Failed to get Redis connection: {e}"))
            })?;

            let deleted: u64 = conn
                .del(key)
                .await
                .map_err(|e| DataStorageError::cache("redis_del", e.to_string()))?;

            let duration = start.elapsed();
            self.record_stats("delete", deleted > 0, duration);

            let was_deleted = deleted > 0;
            tracing::trace!(
                "Redis DEL {}: {}",
                key,
                if was_deleted { "DELETED" } else { "NOT_FOUND" }
            );
            Ok(was_deleted)
        }

        #[cfg(not(feature = "cache"))]
        {
            let duration = start.elapsed();
            self.record_stats("delete", false, duration);

            tracing::trace!("Redis DEL {} (mock): NOT_FOUND", key);
            Ok(false)
        }
    }

    async fn exists(&self, key: &str) -> DataStorageResult<bool> {
        #[cfg(feature = "cache")]
        {
            let mut conn = self.pool.get().await.map_err(|e| {
                DataStorageError::connection(format!("Failed to get Redis connection: {e}"))
            })?;

            let exists: bool = conn
                .exists(key)
                .await
                .map_err(|e| DataStorageError::cache("redis_exists", e.to_string()))?;

            tracing::trace!("Redis EXISTS {}: {}", key, exists);
            Ok(exists)
        }

        #[cfg(not(feature = "cache"))]
        {
            tracing::trace!("Redis EXISTS {} (mock): false", key);
            Ok(false)
        }
    }

    async fn clear(&self) -> DataStorageResult<()> {
        #[cfg(feature = "cache")]
        {
            let mut conn = self.pool.get().await.map_err(|e| {
                DataStorageError::connection(format!("Failed to get Redis connection: {e}"))
            })?;

            redis::cmd("FLUSHDB")
                .query_async::<_, ()>(&mut *conn)
                .await
                .map_err(|e| DataStorageError::cache("redis_flushdb", e.to_string()))?;

            tracing::info!("Redis cache cleared");
            Ok(())
        }

        #[cfg(not(feature = "cache"))]
        {
            tracing::info!("Redis cache cleared (mock)");
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
            entries: 0,      // Redis doesn't track this easily
            memory_usage: 0, // Redis doesn't track this easily
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
    async fn test_redis_cache_creation() -> DataStorageResult<()> {
        let config = CacheConfig {
            redis_url: Some("redis://localhost:6379".to_string()),
            ..CacheConfig::default()
        };

        // Try to create cache, but don't fail if Redis is not available
        match RedisCache::new(config).await {
            Ok(_cache) => {
                // Redis is available, test passed
                Ok(())
            }
            Err(DataStorageError::Database {
                operation: _,
                reason,
            }) if reason.contains("refused") => {
                // Redis server not available, skip test
                println!("Redis server not available, skipping test");
                Ok(())
            }
            Err(e) => Err(e), // Other errors should fail the test
        }
    }
}
