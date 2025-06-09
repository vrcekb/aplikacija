//! Cache implementations for data storage
//!
//! Provides multi-level caching with Redis and in-memory cache support.

/// Safe casting utilities for financial calculations
///
/// These functions prevent precision loss in financial calculations
/// by using appropriate types and bounds checking.
mod safe_cast {
    /// Safely convert u64 to f64 with precision warning for large values
    ///
    /// # Panics
    ///
    /// Never panics - returns converted value with warning for precision loss
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Intentional for financial calculations with warning
    pub fn u64_to_f64_safe(value: u64) -> f64 {
        // f64 mantissa is 52 bits, so values > 2^52 lose precision
        const MAX_SAFE_U64_FOR_F64: u64 = (1_u64 << 52) - 1;

        if value > MAX_SAFE_U64_FOR_F64 {
            tracing::warn!("Precision loss detected in u64->f64 conversion: {}", value);
        }

        value as f64
    }

    /// Calculate ratio safely without precision loss
    #[must_use]
    pub fn safe_ratio(numerator: u64, denominator: u64) -> f64 {
        if denominator == 0 {
            0.0_f64
        } else {
            u64_to_f64_safe(numerator) / u64_to_f64_safe(denominator)
        }
    }
}

pub mod cache_strategy;
pub mod memory_cache;
pub mod redis_cache;

pub use cache_strategy::{CacheStrategy, CacheStrategyStats};
pub use memory_cache::MemoryCache;
pub use redis_cache::RedisCache;
pub use safe_cast::{safe_ratio, u64_to_f64_safe};

use crate::error::DataStorageResult;

/// Cache trait for different cache implementations
#[async_trait::async_trait]
pub trait Cache: Send + Sync {
    /// Get value from cache
    async fn get(&self, key: &str) -> DataStorageResult<Option<Vec<u8>>>;

    /// Set value in cache
    async fn set(&self, key: &str, value: &[u8], ttl_seconds: Option<u64>)
        -> DataStorageResult<()>;

    /// Delete value from cache
    async fn delete(&self, key: &str) -> DataStorageResult<bool>;

    /// Check if key exists in cache
    async fn exists(&self, key: &str) -> DataStorageResult<bool>;

    /// Clear all cache entries
    async fn clear(&self) -> DataStorageResult<()>;

    /// Get cache statistics
    async fn stats(&self) -> DataStorageResult<CacheStats>;
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits
    pub hits: u64,

    /// Number of cache misses
    pub misses: u64,

    /// Number of cache sets
    pub sets: u64,

    /// Number of cache deletes
    pub deletes: u64,

    /// Number of entries in cache
    pub entries: u64,

    /// Memory usage in bytes
    pub memory_usage: u64,

    /// Hit ratio (hits / (hits + misses))
    pub hit_ratio: f64,

    /// Average latency in microseconds
    pub avg_latency_us: u64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStats {
    /// Create new cache stats
    #[must_use]
    pub const fn new() -> Self {
        Self {
            hits: 0,
            misses: 0,
            sets: 0,
            deletes: 0,
            entries: 0,
            memory_usage: 0,
            hit_ratio: 0.0_f64,
            avg_latency_us: 0,
        }
    }

    /// Calculate hit ratio
    pub fn calculate_hit_ratio(&mut self) {
        let total = self.hits + self.misses;
        self.hit_ratio = safe_ratio(self.hits, total);
    }
}
