//! Storage implementations for different tiers
//!
//! This module provides storage implementations for hot, warm, and cold storage tiers.

use std::sync::Arc;
use uuid::Uuid;

use crate::{
    config::DataStorageConfig,
    error::{DataStorageError, DataStorageResult},
    types::{Block, Event, Opportunity, OpportunityFilter, StorageMetrics, Transaction},
};

pub mod cold_storage;
pub mod hot_storage;
pub mod hybrid_store;
pub mod ultra_hot_storage;
pub mod warm_storage;

pub use cold_storage::ColdStorage;
pub use hot_storage::HotStorage;
pub use hybrid_store::HybridStore;
pub use ultra_hot_storage::{UltraHotStorage, UltraHotStorageStats};
pub use warm_storage::WarmStorage;

/// Internal implementation of `DataStorage`
///
/// This struct coordinates between different storage tiers and provides
/// the actual implementation for the public `DataStorage` interface.
#[derive(Debug)]
pub struct DataStorageImpl {
    /// Hot storage for <1ms operations
    hot_storage: Arc<HotStorage>,

    /// Warm storage for analytical queries
    warm_storage: Arc<WarmStorage>,

    /// Cold storage for archival
    cold_storage: Arc<ColdStorage>,

    /// Hybrid store that coordinates between tiers
    hybrid_store: Arc<HybridStore>,
}

impl DataStorageImpl {
    /// Create a new data storage implementation
    ///
    /// # Arguments
    ///
    /// * `config` - Storage configuration
    ///
    /// # Errors
    ///
    /// Returns error if any storage tier initialization fails
    pub async fn new(config: DataStorageConfig) -> DataStorageResult<Self> {
        // Initialize hot storage
        let hot_storage = Arc::new(
            HotStorage::new(&config.hot_storage)
                .await
                .map_err(|e| DataStorageError::database("hot_storage_init", e.to_string()))?,
        );

        // Initialize warm storage
        let warm_storage = Arc::new(
            WarmStorage::new(&config.warm_storage)
                .map_err(|e| DataStorageError::database("warm_storage_init", e.to_string()))?,
        );

        // Initialize cold storage
        let cold_storage = Arc::new(
            ColdStorage::new(&config.cold_storage)
                .await
                .map_err(|e| DataStorageError::database("cold_storage_init", e.to_string()))?,
        );

        // Initialize hybrid store
        let hybrid_store = Arc::new(
            HybridStore::new(
                Arc::clone(&hot_storage),
                Arc::clone(&warm_storage),
                Arc::clone(&cold_storage),
            )
            .map_err(|e| DataStorageError::internal(e.to_string()))?,
        );

        Ok(Self {
            hot_storage,
            warm_storage,
            cold_storage,
            hybrid_store,
        })
    }

    /// Store an opportunity in hot storage (fast path)
    #[inline]
    /// Store opportunity in hot storage for fast access
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    pub fn store_opportunity_fast(&self, opportunity: &Opportunity) -> DataStorageResult<()> {
        self.hot_storage.store_opportunity(opportunity)
    }

    /// Get an opportunity from hot storage (fast path)
    #[inline]
    /// Get opportunity from hot storage
    ///
    /// # Errors
    ///
    /// Returns error if retrieval operation fails
    pub fn get_opportunity_fast(&self, id: &Uuid) -> DataStorageResult<Option<Opportunity>> {
        self.hot_storage.get_opportunity(id)
    }

    /// Store a transaction in warm storage
    /// Store transaction data
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    pub fn store_transaction(&self, transaction: &Transaction) -> DataStorageResult<()> {
        self.warm_storage.store_transaction(transaction)
    }

    /// Query opportunities using hybrid store
    /// Query opportunities with filter
    ///
    /// # Errors
    ///
    /// Returns error if query operation fails
    pub async fn query_opportunities(
        &self,
        filter: &OpportunityFilter,
    ) -> DataStorageResult<Vec<Opportunity>> {
        self.hybrid_store.query_opportunities(filter).await
    }

    /// Store a block in warm storage
    /// Store block data
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    pub fn store_block(&self, block: &Block) -> DataStorageResult<()> {
        self.warm_storage.store_block(block)
    }

    /// Store an event in warm storage
    /// Store event data
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    pub fn store_event(&self, event: &Event) -> DataStorageResult<()> {
        self.warm_storage.store_event(event)
    }

    /// Get storage metrics from all tiers
    /// Get storage metrics from all tiers
    ///
    /// # Errors
    ///
    /// Returns error if metrics collection fails
    pub fn get_metrics(&self) -> DataStorageResult<Vec<StorageMetrics>> {
        let mut metrics = Vec::new();

        // Get hot storage metrics
        if let Ok(hot_metrics) = self.hot_storage.get_metrics() {
            metrics.extend(hot_metrics);
        }

        // Get warm storage metrics
        if let Ok(warm_metrics) = self.warm_storage.get_metrics() {
            metrics.extend(warm_metrics);
        }

        // Get cold storage metrics
        if let Ok(cold_metrics) = self.cold_storage.get_metrics() {
            metrics.extend(cold_metrics);
        }

        Ok(metrics)
    }

    /// Perform health check on all storage tiers
    /// Perform health check on all storage tiers
    ///
    /// # Errors
    ///
    /// Returns error if any health check fails
    pub async fn health_check(&self) -> DataStorageResult<()> {
        // Check hot storage
        self.hot_storage
            .health_check()
            .map_err(|e| DataStorageError::database("hot_storage_health", e.to_string()))?;

        // Check warm storage
        self.warm_storage
            .health_check()
            .map_err(|e| DataStorageError::database("warm_storage_health", e.to_string()))?;

        // Check cold storage
        self.cold_storage
            .health_check()
            .await
            .map_err(|e| DataStorageError::database("cold_storage_health", e.to_string()))?;

        Ok(())
    }

    /// Shutdown all storage tiers gracefully
    /// Shutdown all storage tiers gracefully
    ///
    /// # Errors
    ///
    /// Returns error if shutdown fails
    pub fn shutdown(&self) -> DataStorageResult<()> {
        // Shutdown in reverse order of initialization

        // Shutdown hybrid store first
        if let Err(e) = self.hybrid_store.shutdown() {
            tracing::warn!("Failed to shutdown hybrid store: {e}");
        }

        // Shutdown cold storage
        if let Err(e) = self.cold_storage.shutdown() {
            tracing::warn!("Failed to shutdown cold storage: {e}");
        }

        // Shutdown warm storage
        if let Err(e) = self.warm_storage.shutdown() {
            tracing::warn!("Failed to shutdown warm storage: {e}");
        }

        // Shutdown hot storage last (most critical)
        self.hot_storage
            .shutdown()
            .map_err(|e| DataStorageError::database("hot_storage_shutdown", e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DataStorageConfig;

    fn create_unique_test_config() -> DataStorageResult<DataStorageConfig> {
        let mut config = DataStorageConfig::default();

        // Create unique paths to avoid database lock conflicts
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| DataStorageError::internal(format!("Failed to get timestamp: {e}")))?
            .as_nanos();

        let temp_dir = tempfile::tempdir()
            .map_err(|e| DataStorageError::internal(format!("Failed to create temp dir: {e}")))?;

        // Use in-memory storage for ultra-low latency testing
        config.hot_storage.use_memory_storage = true;
        config.hot_storage.database_path = None;
        config.cold_storage.storage_path = temp_dir.path().join(format!("cold_{timestamp}"));
        config.cold_storage.enable_encryption = false;

        // Use in-memory cache for testing
        config.cache.redis_url = None;
        config.cache.enable_memory_cache = true;

        Ok(config)
    }

    #[tokio::test]
    async fn test_storage_impl_creation() -> DataStorageResult<()> {
        let config = create_unique_test_config()?;
        let _storage = DataStorageImpl::new(config).await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_storage_impl_health_check() -> DataStorageResult<()> {
        let config = create_unique_test_config()?;
        let storage = DataStorageImpl::new(config).await?;
        storage.health_check().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_storage_impl_shutdown() -> DataStorageResult<()> {
        let config = create_unique_test_config()?;
        let storage = DataStorageImpl::new(config).await?;
        storage.shutdown()?;
        Ok(())
    }
}
