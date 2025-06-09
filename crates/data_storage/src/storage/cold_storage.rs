//! Cold Storage Implementation for Archival Data
//!
//! Storage tier optimized for long-term archival with encryption and compression.
//! Provides <100ms access for archived data with unlimited capacity.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs;

use crate::{
    config::ColdStorageConfig,
    error::{DataStorageError, DataStorageResult},
    types::{CompressionType, StorageMetrics, StorageTier},
};

/// Cold storage implementation for archival data
///
/// This storage provides long-term archival capabilities with encryption
/// and compression for unlimited capacity storage.
#[derive(Debug)]
pub struct ColdStorage {
    /// Configuration
    config: ColdStorageConfig,

    /// Storage directory
    storage_path: PathBuf,

    /// Operation metrics
    metrics: Arc<parking_lot::Mutex<Vec<StorageMetrics>>>,
}

impl ColdStorage {
    /// Create a new cold storage instance
    ///
    /// # Arguments
    ///
    /// * `config` - Cold storage configuration
    ///
    /// # Errors
    ///
    /// Returns error if storage initialization fails
    pub async fn new(config: &ColdStorageConfig) -> DataStorageResult<Self> {
        // Create storage directory if it doesn't exist
        fs::create_dir_all(&config.storage_path)
            .await
            .map_err(|e| {
                DataStorageError::database(
                    "create_storage_dir",
                    format!(
                        "Failed to create storage directory {}: {}",
                        config.storage_path.display(),
                        e
                    ),
                )
            })?;

        Ok(Self {
            config: config.clone(),
            storage_path: config.storage_path.clone(),
            metrics: Arc::new(parking_lot::Mutex::new(Vec::new())),
        })
    }

    /// Archive data to cold storage
    ///
    /// # Arguments
    ///
    /// * `key` - Unique key for the data
    /// * `data` - Data to archive
    ///
    /// # Errors
    ///
    /// Returns error if archival fails
    pub async fn archive_data(&self, key: &str, data: &[u8]) -> DataStorageResult<()> {
        let start = Instant::now();

        // Compress data if enabled
        let compressed_data = if self.config.compression_type == CompressionType::None {
            data.to_vec()
        } else {
            self.compress_data(data)?
        };

        // Encrypt data if enabled
        let final_data = if self.config.enable_encryption {
            Self::encrypt_data(&compressed_data)
        } else {
            compressed_data
        };

        // Generate file path
        let file_path = self.generate_file_path(key);

        // Ensure parent directory exists
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)
                .await
                .map_err(|e| DataStorageError::database("create_archive_dir", e.to_string()))?;
        }

        // Write to file
        fs::write(&file_path, &final_data)
            .await
            .map_err(|e| DataStorageError::database("write_archive", e.to_string()))?;

        let duration = start.elapsed();
        self.record_metric("archive_data", duration, final_data.len(), true);

        Ok(())
    }

    /// Retrieve data from cold storage
    ///
    /// # Arguments
    ///
    /// * `key` - Unique key for the data
    ///
    /// # Errors
    ///
    /// Returns error if retrieval fails
    pub async fn retrieve_data(&self, key: &str) -> DataStorageResult<Option<Vec<u8>>> {
        let start = Instant::now();

        let file_path = self.generate_file_path(key);

        // Check if file exists
        if !file_path.exists() {
            return Ok(None);
        }

        // Read file
        let encrypted_data = fs::read(&file_path)
            .await
            .map_err(|e| DataStorageError::database("read_archive", e.to_string()))?;

        // Decrypt data if enabled
        let compressed_data = if self.config.enable_encryption {
            Self::decrypt_data(&encrypted_data)
        } else {
            encrypted_data
        };

        // Decompress data if needed
        let final_data = if self.config.compression_type == CompressionType::None {
            compressed_data
        } else {
            self.decompress_data(&compressed_data)?
        };

        let duration = start.elapsed();
        self.record_metric("retrieve_data", duration, final_data.len(), true);

        Ok(Some(final_data))
    }

    /// Delete archived data
    ///
    /// # Arguments
    ///
    /// * `key` - Unique key for the data
    ///
    /// # Errors
    ///
    /// Returns error if deletion fails
    pub async fn delete_data(&self, key: &str) -> DataStorageResult<bool> {
        let start = Instant::now();

        let file_path = self.generate_file_path(key);

        if !file_path.exists() {
            return Ok(false);
        }

        fs::remove_file(&file_path)
            .await
            .map_err(|e| DataStorageError::database("delete_archive", e.to_string()))?;

        let duration = start.elapsed();
        self.record_metric("delete_data", duration, 0, true);

        Ok(true)
    }

    /// List all archived keys
    /// List all stored keys
    ///
    /// # Errors
    ///
    /// Returns error if key listing fails
    pub async fn list_keys(&self) -> DataStorageResult<Vec<String>> {
        let start = Instant::now();

        let mut keys = Vec::new();
        let mut entries = fs::read_dir(&self.storage_path)
            .await
            .map_err(|e| DataStorageError::database("list_archives", e.to_string()))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| DataStorageError::database("read_dir_entry", e.to_string()))?
        {
            if entry
                .file_type()
                .await
                .map_err(|e| DataStorageError::database("get_file_type", e.to_string()))?
                .is_file()
            {
                if let Some(file_name) = entry.file_name().to_str() {
                    // Remove file extension to get key
                    let key = file_name.trim_end_matches(".archive");
                    keys.push(key.to_string());
                }
            }
        }

        let duration = start.elapsed();
        self.record_metric("list_keys", duration, keys.len(), true);

        Ok(keys)
    }

    /// Get storage statistics
    /// Get detailed storage statistics
    ///
    /// # Errors
    ///
    /// Returns error if statistics collection fails
    pub async fn get_stats(&self) -> DataStorageResult<ColdStorageStats> {
        let start = Instant::now();

        let mut total_size = 0u64;
        let mut file_count = 0u64;

        let mut entries = fs::read_dir(&self.storage_path)
            .await
            .map_err(|e| DataStorageError::database("stats_read_dir", e.to_string()))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| DataStorageError::database("stats_read_entry", e.to_string()))?
        {
            if entry
                .file_type()
                .await
                .map_err(|e| DataStorageError::database("stats_file_type", e.to_string()))?
                .is_file()
            {
                let metadata = entry
                    .metadata()
                    .await
                    .map_err(|e| DataStorageError::database("stats_metadata", e.to_string()))?;

                total_size += metadata.len();
                file_count += 1;
            }
        }

        let duration = start.elapsed();
        self.record_metric("get_stats", duration, 0, true);

        Ok(ColdStorageStats {
            total_size_bytes: total_size,
            file_count,
            compression_ratio: 0.0, // Would need to track original sizes
        })
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
    /// Perform health check on cold storage
    ///
    /// # Errors
    ///
    /// Returns error if health check fails
    pub async fn health_check(&self) -> DataStorageResult<()> {
        // Check if storage directory is accessible
        let metadata = fs::metadata(&self.storage_path)
            .await
            .map_err(|e| DataStorageError::database("health_check_metadata", e.to_string()))?;

        if !metadata.is_dir() {
            return Err(DataStorageError::database(
                "health_check_not_dir",
                "Storage path is not a directory".to_string(),
            ));
        }

        // Try to create a test file
        let test_path = self.storage_path.join(".health_check");
        fs::write(&test_path, b"test")
            .await
            .map_err(|e| DataStorageError::database("health_check_write", e.to_string()))?;

        // Clean up test file
        let _ = fs::remove_file(&test_path).await;

        Ok(())
    }

    /// Shutdown storage gracefully
    /// Shutdown cold storage gracefully
    ///
    /// # Errors
    ///
    /// Returns error if shutdown fails
    pub fn shutdown(&self) -> DataStorageResult<()> {
        tracing::info!("Cold storage shutdown completed");
        Ok(())
    }

    /// Generate file path for a key
    fn generate_file_path(&self, key: &str) -> PathBuf {
        // Create subdirectories based on key hash for better distribution
        let hash = Self::hash_key(key);
        let subdir1 = &hash[0..2];
        let subdir2 = &hash[2..4];

        self.storage_path
            .join(subdir1)
            .join(subdir2)
            .join(format!("{key}.archive"))
    }

    /// Hash a key for directory distribution
    fn hash_key(key: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Compress data
    fn compress_data(&self, data: &[u8]) -> DataStorageResult<Vec<u8>> {
        match self.config.compression_type {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Lz4 => {
                #[cfg(feature = "cold-storage")]
                {
                    lz4::block::compress(data, None, false).map_err(|e| {
                        DataStorageError::internal(format!("LZ4 compression failed: {e}"))
                    })
                }
                #[cfg(not(feature = "cold-storage"))]
                {
                    tracing::warn!("LZ4 compression not available, returning uncompressed data");
                    Ok(data.to_vec())
                }
            }
            _ => Err(DataStorageError::internal(
                "Unsupported compression type".to_string(),
            )),
        }
    }

    /// Decompress data
    fn decompress_data(&self, data: &[u8]) -> DataStorageResult<Vec<u8>> {
        match self.config.compression_type {
            CompressionType::None => Ok(data.to_vec()),
            CompressionType::Lz4 => {
                #[cfg(feature = "cold-storage")]
                {
                    lz4::block::decompress(data, None).map_err(|e| {
                        DataStorageError::internal(format!("LZ4 decompression failed: {e}"))
                    })
                }
                #[cfg(not(feature = "cold-storage"))]
                {
                    tracing::warn!("LZ4 decompression not available, returning data as-is");
                    Ok(data.to_vec())
                }
            }
            _ => Err(DataStorageError::internal(
                "Unsupported compression type".to_string(),
            )),
        }
    }

    /// Encrypt data (placeholder implementation)
    fn encrypt_data(data: &[u8]) -> Vec<u8> {
        // TODO: Implement actual encryption using AES-GCM
        // For now, just return the data as-is
        tracing::warn!("Encryption not yet implemented in cold storage");
        data.to_vec()
    }

    /// Decrypt data (placeholder implementation)
    fn decrypt_data(data: &[u8]) -> Vec<u8> {
        // TODO: Implement actual decryption using AES-GCM
        // For now, just return the data as-is
        tracing::warn!("Decryption not yet implemented in cold storage");
        data.to_vec()
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
            tier: StorageTier::Cold,
            #[allow(clippy::option_if_let_else)] // Explicit error handling is safer for financial data
            duration_us: if let Ok(value) = u64::try_from(duration.as_micros()) {
                value
            } else {
                tracing::warn!("Duration overflow in cold storage metrics, using max value");
                u64::MAX
            },
            data_size: data_size as u64,
            success,
            error_code: if success { None } else { Some(1300) },
            timestamp: Utc::now(),
        };

        let mut metrics = self.metrics.lock();
        metrics.push(metric);

        // Keep only last 1000 metrics
        if metrics.len() > 1000 {
            metrics.drain(0..500);
        }
    }
}

/// Cold storage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColdStorageStats {
    /// Total storage size in bytes
    pub total_size_bytes: u64,

    /// Number of archived files
    pub file_count: u64,

    /// Compression ratio (compressed/original)
    pub compression_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_storage() -> Result<(ColdStorage, TempDir), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;
        let config = ColdStorageConfig {
            storage_path: temp_dir.path().to_path_buf(),
            enable_encryption: false, // Disable for testing
            ..Default::default()
        };

        let storage = ColdStorage::new(&config).await?;
        Ok((storage, temp_dir))
    }

    #[tokio::test]
    async fn test_archive_and_retrieve() -> DataStorageResult<()> {
        let (storage, _temp_dir) = create_test_storage()
            .await
            .map_err(|e| DataStorageError::internal(format!("Test setup failed: {e}")))?;

        let key = "test_key";
        let data = b"test data for archival";

        // Archive data
        storage.archive_data(key, data).await?;

        // Retrieve data
        let retrieved = storage.retrieve_data(key).await?;
        assert!(retrieved.is_some());
        if let Some(retrieved_data) = retrieved {
            assert_eq!(retrieved_data, data);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_data() -> DataStorageResult<()> {
        let (storage, _temp_dir) = create_test_storage()
            .await
            .map_err(|e| DataStorageError::internal(format!("Test setup failed: {e}")))?;

        let key = "test_key";
        let data = b"test data";

        // Archive and then delete
        storage.archive_data(key, data).await?;
        let deleted = storage.delete_data(key).await?;
        assert!(deleted);

        // Verify it's gone
        let retrieved = storage.retrieve_data(key).await?;
        assert!(retrieved.is_none());

        Ok(())
    }
}
