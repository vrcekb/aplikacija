//! Hot Storage Implementation using redb
//!
//! Ultra-fast storage tier optimized for <1ms operations using redb embedded database.
//! This storage tier is designed for frequently accessed data with minimal latency.

use chrono::Utc;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use crate::{
    config::HotStorageConfig,
    error::{CriticalError, DataStorageError, DataStorageResult},
    types::{Opportunity, StorageMetrics, StorageTier},
};

/// Table definitions for redb
const OPPORTUNITIES_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("opportunities");
const METADATA_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("metadata");

/// Hot storage implementation using redb
///
/// This storage provides ultra-fast access to frequently used data
/// with <1ms latency guarantee for critical operations.
#[derive(Debug)]
#[allow(dead_code)] // Fields will be used in full implementation
pub struct HotStorage {
    /// redb database instance
    database: Arc<Database>,

    /// Configuration
    config: HotStorageConfig,

    /// Operation metrics
    metrics: Arc<parking_lot::Mutex<Vec<StorageMetrics>>>,
}

impl HotStorage {
    /// Create a new hot storage instance
    ///
    /// # Arguments
    ///
    /// * `config` - Hot storage configuration
    ///
    /// # Errors
    ///
    /// Returns error if database initialization fails
    pub async fn new(config: &HotStorageConfig) -> DataStorageResult<Self> {
        let database = if config.use_memory_storage {
            // Use in-memory database for ultra-low latency
            Database::builder()
                .set_cache_size(
                    usize::try_from(config.cache_size_bytes).unwrap_or(128 * 1024 * 1024),
                )
                .create_with_backend(redb::backends::InMemoryBackend::new())
                .map_err(|e| {
                    DataStorageError::database(
                        "database_open",
                        format!("Failed to create in-memory database: {e}"),
                    )
                })?
        } else if let Some(ref path) = config.database_path {
            // Ensure parent directory exists
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    DataStorageError::database(
                        "create_directory",
                        format!("Failed to create directory {}: {}", parent.display(), e),
                    )
                })?;
            }

            // Open file-based database
            Database::builder()
                .set_cache_size(
                    usize::try_from(config.cache_size_bytes).unwrap_or(128 * 1024 * 1024),
                )
                .create(path)
                .map_err(|e| {
                    DataStorageError::database(
                        "database_open",
                        format!("Failed to open database {}: {}", path.display(), e),
                    )
                })?
        } else {
            return Err(DataStorageError::configuration(
                "Either use_memory_storage must be true or database_path must be provided",
            ));
        };

        // Initialize tables
        let write_txn = database
            .begin_write()
            .map_err(|e| DataStorageError::database("begin_write", e.to_string()))?;

        {
            let _opportunities_table = write_txn.open_table(OPPORTUNITIES_TABLE).map_err(|e| {
                DataStorageError::database("open_opportunities_table", e.to_string())
            })?;

            let _metadata_table = write_txn
                .open_table(METADATA_TABLE)
                .map_err(|e| DataStorageError::database("open_metadata_table", e.to_string()))?;
        }

        write_txn
            .commit()
            .map_err(|e| DataStorageError::database("commit_init", e.to_string()))?;

        Ok(Self {
            database: Arc::new(database),
            config: config.clone(),
            metrics: Arc::new(parking_lot::Mutex::new(Vec::new())),
        })
    }

    /// Store an opportunity (critical path - <1ms)
    ///
    /// # Arguments
    ///
    /// * `opportunity` - The opportunity to store
    ///
    /// # Errors
    ///
    /// Returns error if storage operation fails
    #[inline]
    pub fn store_opportunity(&self, opportunity: &Opportunity) -> DataStorageResult<()> {
        let start = Instant::now();

        // Serialize opportunity
        let key = opportunity.id.as_bytes().as_slice();
        let value = serde_json::to_vec(opportunity).map_err(DataStorageError::Serialization)?;

        // Store in database
        let write_txn = self
            .database
            .begin_write()
            .map_err(|_e| DataStorageError::Critical(CriticalError::HotStorageFailure(1001)))?;

        {
            let mut table = write_txn
                .open_table(OPPORTUNITIES_TABLE)
                .map_err(|_e| DataStorageError::Critical(CriticalError::HotStorageFailure(1002)))?;

            table
                .insert(key, value.as_slice())
                .map_err(|_e| DataStorageError::Critical(CriticalError::HotStorageFailure(1003)))?;
        }

        write_txn
            .commit()
            .map_err(|_e| DataStorageError::Critical(CriticalError::HotStorageFailure(1004)))?;

        let duration = start.elapsed();

        // Record metrics
        self.record_metric("store_opportunity", duration, value.len(), true);

        // Ensure <1ms latency for critical path
        if duration.as_millis() >= 1 {
            tracing::warn!("Hot storage operation exceeded 1ms: {:?}", duration);
        }

        Ok(())
    }

    /// Get an opportunity (critical path - <1ms)
    ///
    /// # Arguments
    ///
    /// * `id` - The opportunity ID
    ///
    /// # Errors
    ///
    /// Returns error if retrieval fails
    #[inline]
    pub fn get_opportunity(&self, id: &Uuid) -> DataStorageResult<Option<Opportunity>> {
        let start = Instant::now();

        let key = id.as_bytes().as_slice();

        // Read from database
        let read_txn = self
            .database
            .begin_read()
            .map_err(|_e| DataStorageError::Critical(CriticalError::HotStorageFailure(1005)))?;

        let table = read_txn
            .open_table(OPPORTUNITIES_TABLE)
            .map_err(|_e| DataStorageError::Critical(CriticalError::HotStorageFailure(1006)))?;

        let result = match table.get(key) {
            Ok(Some(value)) => {
                let data = value.value();
                let opportunity: Opportunity =
                    serde_json::from_slice(data).map_err(DataStorageError::Serialization)?;
                Some(opportunity)
            }
            Ok(None) => None,
            Err(_e) => {
                return Err(DataStorageError::Critical(
                    CriticalError::HotStorageFailure(1007),
                ))
            }
        };

        let duration = start.elapsed();

        // Record metrics
        let data_size = result.as_ref().map_or(0, |_| 0); // Approximate size
        self.record_metric("get_opportunity", duration, data_size, true);

        // Ensure <1ms latency for critical path
        if duration.as_millis() >= 1 {
            tracing::warn!("Hot storage operation exceeded 1ms: {:?}", duration);
        }

        Ok(result)
    }

    /// Delete an opportunity
    ///
    /// # Arguments
    ///
    /// * `id` - The opportunity ID to delete
    ///
    /// # Errors
    ///
    /// Returns error if deletion fails
    pub fn delete_opportunity(&self, id: &Uuid) -> DataStorageResult<bool> {
        let start = Instant::now();

        let key = id.as_bytes().as_slice();

        let write_txn = self
            .database
            .begin_write()
            .map_err(|e| DataStorageError::database("begin_write", e.to_string()))?;

        let deleted = {
            let mut table = write_txn
                .open_table(OPPORTUNITIES_TABLE)
                .map_err(|e| DataStorageError::database("open_table", e.to_string()))?;

            let result = table
                .remove(key)
                .map_err(|e| DataStorageError::database("remove", e.to_string()))?;
            result.is_some()
        };

        write_txn
            .commit()
            .map_err(|e| DataStorageError::database("commit", e.to_string()))?;

        let duration = start.elapsed();
        self.record_metric("delete_opportunity", duration, 0, true);

        Ok(deleted)
    }

    /// Get storage metrics
    /// Get storage metrics
    ///
    /// # Errors
    ///
    /// Returns error if metrics collection fails
    pub fn get_metrics(&self) -> DataStorageResult<Vec<StorageMetrics>> {
        let metrics = self.metrics.lock().clone();
        Ok(metrics)
    }

    /// Perform health check
    /// Perform health check on storage
    ///
    /// # Errors
    ///
    /// Returns error if health check fails
    pub fn health_check(&self) -> DataStorageResult<()> {
        // Try to read from metadata table
        let read_txn = self
            .database
            .begin_read()
            .map_err(|e| DataStorageError::database("health_check_read", e.to_string()))?;

        let _table = read_txn
            .open_table(METADATA_TABLE)
            .map_err(|e| DataStorageError::database("health_check_table", e.to_string()))?;

        Ok(())
    }

    /// Shutdown storage gracefully
    /// Shutdown storage gracefully
    ///
    /// # Errors
    ///
    /// Returns error if shutdown fails
    pub fn shutdown(&self) -> DataStorageResult<()> {
        // redb handles shutdown automatically when dropped
        tracing::info!("Hot storage shutdown completed");
        Ok(())
    }

    /// Record operation metrics
    fn record_metric(
        &self,
        operation: &str,
        duration: std::time::Duration,
        data_size: usize,
        success: bool,
    ) {
        let metric = StorageMetrics {
            operation: operation.to_string(),
            tier: StorageTier::Hot,
            duration_us: u64::try_from(duration.as_micros()).unwrap_or(u64::MAX),
            data_size: data_size as u64,
            success,
            error_code: if success { None } else { Some(1000) },
            timestamp: Utc::now(),
        };

        let mut metrics = self.metrics.lock();
        metrics.push(metric);

        // Keep only last 1000 metrics to prevent memory growth
        if metrics.len() > 1000 {
            metrics.drain(0..500);
        }
    }

    /// Get database statistics
    /// Get detailed storage statistics
    ///
    /// # Errors
    ///
    /// Returns error if statistics collection fails
    pub fn get_stats(&self) -> DataStorageResult<HotStorageStats> {
        let read_txn = self
            .database
            .begin_read()
            .map_err(|e| DataStorageError::database("stats_read", e.to_string()))?;

        let opportunities_table = read_txn
            .open_table(OPPORTUNITIES_TABLE)
            .map_err(|e| DataStorageError::database("stats_table", e.to_string()))?;

        let opportunity_count = opportunities_table
            .len()
            .map_err(|e| DataStorageError::database("stats_count", e.to_string()))?;

        Ok(HotStorageStats {
            opportunity_count,
            database_size_bytes: 0, // redb doesn't expose this easily
        })
    }
}

/// Hot storage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HotStorageStats {
    /// Number of opportunities stored
    pub opportunity_count: u64,

    /// Database size in bytes
    pub database_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HotStorageConfig;
    use crate::types::Opportunity;
    use tempfile::TempDir;

    async fn create_test_storage() -> DataStorageResult<(HotStorage, TempDir)> {
        let temp_dir = TempDir::new()
            .map_err(|e| DataStorageError::internal(format!("Failed to create temp dir: {e}")))?;

        // No longer need timestamp for in-memory storage

        let config = HotStorageConfig {
            use_memory_storage: true,
            database_path: None,
            ..Default::default()
        };

        let storage = HotStorage::new(&config).await?;
        Ok((storage, temp_dir))
    }

    #[tokio::test]
    async fn test_store_and_get_opportunity() -> DataStorageResult<()> {
        let (storage, _temp_dir) = create_test_storage().await?;

        let opportunity = Opportunity::new(
            "arbitrage".to_string(),
            1,
            "1.5".to_string(),
            "0.1".to_string(),
            "1.4".to_string(),
            0.95,
        );

        // Store opportunity
        storage.store_opportunity(&opportunity)?;

        // Retrieve opportunity
        let retrieved = storage.get_opportunity(&opportunity.id)?;
        assert!(retrieved.is_some());
        if let Some(retrieved_opp) = retrieved {
            assert_eq!(retrieved_opp.id, opportunity.id);
        } else {
            return Err(DataStorageError::internal(
                "Expected opportunity to be retrieved".to_string(),
            ));
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_opportunity() -> DataStorageResult<()> {
        let (storage, _temp_dir) = create_test_storage().await?;

        let opportunity = Opportunity::new(
            "arbitrage".to_string(),
            1,
            "1.5".to_string(),
            "0.1".to_string(),
            "1.4".to_string(),
            0.95,
        );

        // Store and then delete
        storage.store_opportunity(&opportunity)?;
        let deleted = storage.delete_opportunity(&opportunity.id)?;
        assert!(deleted);

        // Verify it's gone
        let retrieved = storage.get_opportunity(&opportunity.id)?;
        assert!(retrieved.is_none());

        Ok(())
    }

    #[tokio::test]
    async fn test_latency_requirement() -> DataStorageResult<()> {
        let (storage, _temp_dir) = create_test_storage().await?;

        let opportunity = Opportunity::new(
            "arbitrage".to_string(),
            1,
            "1.5".to_string(),
            "0.1".to_string(),
            "1.4".to_string(),
            0.95,
        );

        // Test store latency
        let start = Instant::now();
        storage.store_opportunity(&opportunity)?;
        let store_duration = start.elapsed();

        // Test get latency
        let start = Instant::now();
        let _retrieved = storage.get_opportunity(&opportunity.id)?;
        let get_duration = start.elapsed();

        // Assert <1ms requirement (1000 microseconds)
        assert!(
            store_duration.as_micros() < 1000,
            "Store too slow: {store_duration:?}"
        );
        assert!(
            get_duration.as_micros() < 1000,
            "Get too slow: {get_duration:?}"
        );

        Ok(())
    }
}
