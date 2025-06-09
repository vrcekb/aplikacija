//! Hybrid Store Implementation
//!
//! Coordinates between hot, warm, and cold storage tiers to provide
//! optimal performance and cost characteristics for different data access patterns.

use std::sync::Arc;
use uuid::Uuid;

use crate::{
    error::{DataStorageError, DataStorageResult},
    storage::{ColdStorage, HotStorage, WarmStorage},
    types::{Opportunity, OpportunityFilter, StorageTier},
};

/// Hybrid store that coordinates between storage tiers
///
/// This store automatically routes operations to the appropriate storage tier
/// based on data characteristics and access patterns.
#[derive(Debug)]
pub struct HybridStore {
    /// Hot storage for <1ms operations
    hot: Arc<HotStorage>,

    /// Warm storage for analytical queries
    warm: Arc<WarmStorage>,

    /// Cold storage for archival
    cold: Arc<ColdStorage>,
}

impl HybridStore {
    /// Create a new hybrid store
    ///
    /// # Arguments
    ///
    /// * `hot_storage` - Hot storage instance
    /// * `warm_storage` - Warm storage instance
    /// * `cold_storage` - Cold storage instance
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub const fn new(
        hot_storage: Arc<HotStorage>,
        warm_storage: Arc<WarmStorage>,
        cold_storage: Arc<ColdStorage>,
    ) -> DataStorageResult<Self> {
        Ok(Self {
            hot: hot_storage,
            warm: warm_storage,
            cold: cold_storage,
        })
    }

    /// Store an opportunity with automatic tier selection
    ///
    /// # Arguments
    ///
    /// * `opportunity` - The opportunity to store
    ///
    /// # Errors
    ///
    /// Returns error if storage fails
    pub async fn store_opportunity(&self, opportunity: &Opportunity) -> DataStorageResult<()> {
        let tier = Self::select_storage_tier_for_opportunity(opportunity);

        match tier {
            StorageTier::Hot => {
                // Store in hot storage for fast access
                self.hot.store_opportunity(opportunity)?;

                // Also store in warm storage for persistence and analytics
                if let Err(e) = self.warm.store_opportunity(opportunity).await {
                    tracing::warn!("Failed to store opportunity in warm storage: {e}");
                }
            }
            StorageTier::Warm => {
                // Store only in warm storage
                self.warm.store_opportunity(opportunity).await?;
            }
            StorageTier::Cold => {
                // Serialize and store in cold storage
                let data =
                    serde_json::to_vec(opportunity).map_err(DataStorageError::Serialization)?;

                let key = format!("opportunity_{}", opportunity.id);
                self.cold.archive_data(&key, &data).await?;
            }
        }

        Ok(())
    }

    /// Get an opportunity with automatic tier lookup
    ///
    /// # Arguments
    ///
    /// * `id` - The opportunity ID
    ///
    /// # Errors
    ///
    /// Returns error if retrieval fails
    pub async fn get_opportunity(&self, id: &Uuid) -> DataStorageResult<Option<Opportunity>> {
        // Try hot storage first (fastest)
        if let Ok(Some(opportunity)) = self.hot.get_opportunity(id) {
            return Ok(Some(opportunity));
        }

        // Try warm storage next
        let filter = OpportunityFilter {
            limit: Some(1),
            offset: Some(0),
            ..Default::default()
        };

        // Create a more specific query for the ID
        // Note: This is a simplified implementation - in practice you'd want
        // a more efficient get_by_id method in warm storage
        if let Ok(opportunities) = self.warm.query_opportunities(&filter).await {
            if let Some(opportunity) = opportunities.into_iter().find(|o| o.id == *id) {
                // Cache in hot storage for future access
                if let Err(e) = self.hot.store_opportunity(&opportunity) {
                    tracing::warn!("Failed to cache opportunity in hot storage: {e}");
                }
                return Ok(Some(opportunity));
            }
        }

        // Finally try cold storage
        let key = format!("opportunity_{id}");
        if let Ok(Some(data)) = self.cold.retrieve_data(&key).await {
            let opportunity: Opportunity =
                serde_json::from_slice(&data).map_err(DataStorageError::Serialization)?;

            // Cache in hot storage for future access
            if let Err(e) = self.hot.store_opportunity(&opportunity) {
                tracing::warn!("Failed to cache opportunity in hot storage: {e}");
            }

            return Ok(Some(opportunity));
        }

        Ok(None)
    }

    /// Query opportunities across storage tiers
    ///
    /// # Arguments
    ///
    /// * `filter` - Query filter parameters
    ///
    /// # Errors
    ///
    /// Returns error if query fails
    pub async fn query_opportunities(
        &self,
        filter: &OpportunityFilter,
    ) -> DataStorageResult<Vec<Opportunity>> {
        // For complex queries, use warm storage primarily
        let opportunities = self.warm.query_opportunities(filter).await?;

        // If we need more results and have cold storage, query there too
        if opportunities.len() < filter.limit.unwrap_or(100) as usize {
            // TODO: Implement cold storage querying
            // This would require indexing in cold storage
        }

        Ok(opportunities)
    }

    /// Migrate data between storage tiers based on access patterns
    /// Migrate data between storage tiers
    ///
    /// # Errors
    ///
    /// Returns error if migration fails
    pub fn migrate_data(&self) -> DataStorageResult<()> {
        // TODO: Implement data migration logic
        // This would analyze access patterns and move data between tiers
        // For example:
        // - Move frequently accessed data from warm to hot
        // - Move old data from warm to cold
        // - Remove expired data from all tiers

        tracing::info!("Data migration not yet implemented");
        Ok(())
    }

    /// Cleanup expired data across all tiers
    /// Cleanup expired data from all tiers
    ///
    /// # Errors
    ///
    /// Returns error if cleanup fails
    pub fn cleanup_expired_data(&self) -> DataStorageResult<()> {
        // TODO: Implement cleanup logic
        // This would remove expired opportunities and old data

        tracing::info!("Data cleanup not yet implemented");
        Ok(())
    }

    /// Get storage tier statistics
    ///
    /// # Errors
    ///
    /// Returns error if statistics collection fails
    pub async fn get_tier_stats(&self) -> DataStorageResult<HybridStoreStats> {
        let hot_stats = self.hot.get_stats().map_err(|e| {
            DataStorageError::internal(format!("Failed to get hot storage stats: {e}"))
        })?;
        let cold_stats = self.cold.get_stats().await.map_err(|e| {
            DataStorageError::internal(format!("Failed to get cold storage stats: {e}"))
        })?;

        Ok(HybridStoreStats {
            hot_opportunity_count: hot_stats.opportunity_count,
            hot_database_size: hot_stats.database_size_bytes,
            cold_file_count: cold_stats.file_count,
            cold_total_size: cold_stats.total_size_bytes,
            warm_connection_count: 0, // TODO: Get from warm storage
        })
    }

    /// Shutdown hybrid store
    /// Shutdown all storage tiers gracefully
    ///
    /// # Errors
    ///
    /// Returns error if shutdown fails
    pub fn shutdown(&self) -> DataStorageResult<()> {
        tracing::info!("Hybrid store shutdown completed");
        Ok(())
    }

    /// Select appropriate storage tier for an opportunity
    fn select_storage_tier_for_opportunity(opportunity: &Opportunity) -> StorageTier {
        // Decision logic for tier selection

        // High-value or high-confidence opportunities go to hot storage
        #[allow(clippy::option_if_let_else)] // Explicit error handling is safer for financial data
        let profit_eth = if let Ok(value) = opportunity.profit_eth.parse::<f64>() {
            value
        } else {
            tracing::warn!(
                "Invalid profit_eth value: {}, defaulting to 0.0",
                opportunity.profit_eth
            );
            0.0_f64
        };
        if opportunity.confidence_score > 0.9_f64 || profit_eth > 1.0_f64 {
            return StorageTier::Hot;
        }

        // Recent opportunities go to warm storage
        let age = chrono::Utc::now().signed_duration_since(opportunity.created_at);
        if age.num_hours() < 24 {
            return StorageTier::Warm;
        }

        // Old opportunities go to cold storage
        StorageTier::Cold
    }

    /// Select storage tier for queries based on filter characteristics
    #[allow(dead_code)] // Will be used in future implementations
    fn select_query_tier(filter: &OpportunityFilter) -> StorageTier {
        // Recent data queries use warm storage
        if let Some(start_time) = filter.start_time {
            let age = chrono::Utc::now().signed_duration_since(start_time);
            if age.num_days() < 7 {
                return StorageTier::Warm;
            }
        }

        // High-confidence or high-value queries might check hot storage first
        if let Some(min_confidence) = filter.min_confidence {
            if min_confidence > 0.8_f64 {
                return StorageTier::Hot;
            }
        }

        // Default to warm storage for most queries
        StorageTier::Warm
    }
}

/// Hybrid store statistics
#[derive(Debug, Clone)]
pub struct HybridStoreStats {
    /// Number of opportunities in hot storage
    pub hot_opportunity_count: u64,

    /// Hot storage database size
    pub hot_database_size: u64,

    /// Number of files in cold storage
    pub cold_file_count: u64,

    /// Total cold storage size
    pub cold_total_size: u64,

    /// Number of warm storage connections
    pub warm_connection_count: u32,
}

// Default implementations for stats

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ColdStorageConfig, HotStorageConfig, WarmStorageConfig};
    use tempfile::TempDir;

    async fn create_test_hybrid_store(
    ) -> Result<(HybridStore, TempDir, TempDir), Box<dyn std::error::Error>> {
        let hot_temp_dir = TempDir::new()?;
        let cold_temp_dir = TempDir::new()?;

        let hot_config = HotStorageConfig {
            use_memory_storage: true,
            database_path: None,
            ..Default::default()
        };

        let cold_config = ColdStorageConfig {
            storage_path: cold_temp_dir.path().to_path_buf(),
            enable_encryption: false,
            ..Default::default()
        };

        let warm_config = WarmStorageConfig::default();

        let hot_storage = Arc::new(HotStorage::new(&hot_config).await?);
        let warm_storage = Arc::new(WarmStorage::new(&warm_config)?);
        let cold_storage = Arc::new(ColdStorage::new(&cold_config).await?);

        let hybrid_store = HybridStore::new(hot_storage, warm_storage, cold_storage)?;

        Ok((hybrid_store, hot_temp_dir, cold_temp_dir))
    }

    #[tokio::test]
    async fn test_hybrid_store_creation() -> DataStorageResult<()> {
        let (_store, _hot_temp, _cold_temp) = create_test_hybrid_store()
            .await
            .map_err(|e| DataStorageError::internal(format!("Test setup failed: {e}")))?;
        Ok(())
    }

    #[tokio::test]
    async fn test_tier_selection() -> DataStorageResult<()> {
        let (_store, _hot_temp, _cold_temp) = create_test_hybrid_store()
            .await
            .map_err(|e| DataStorageError::internal(format!("Test setup failed: {e}")))?;

        // High confidence opportunity should go to hot storage
        let high_conf_opp = Opportunity::new(
            "arbitrage".to_string(),
            1,
            "2.0".to_string(),
            "0.1".to_string(),
            "1.9".to_string(),
            0.95,
        );

        let tier = HybridStore::select_storage_tier_for_opportunity(&high_conf_opp);
        assert_eq!(tier, StorageTier::Hot);

        Ok(())
    }
}
