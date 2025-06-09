//! Local SQLite-based vault implementation

use super::{HealthStatus, Vault, VaultHealth, VaultMetadata, VaultStats};
use crate::error::{SecureStorageError, SecureStorageResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Local vault implementation using `SQLite`
///
/// This module implements a high-performance local vault for `TallyIO` financial platform.
/// All database operations are optimized for <1ms latency requirements.
pub struct LocalVault {
    /// Database connection
    connection: Arc<Mutex<Connection>>,
    /// Database file path
    database_path: String,
    /// Statistics
    stats: Arc<Mutex<VaultStats>>,
}

impl LocalVault {
    /// Create a new local vault
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub async fn new(database_path: &str) -> SecureStorageResult<Self> {
        let path = Path::new(database_path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| SecureStorageError::Database {
                operation: "create_directories".to_string(),
                reason: format!("Failed to create directory {}: {}", parent.display(), e),
            })?;
        }

        // Open database connection
        let connection =
            Connection::open(database_path).map_err(|e| SecureStorageError::Database {
                operation: "open".to_string(),
                reason: format!("Failed to open database {database_path}: {e}"),
            })?;

        let vault = Self {
            connection: Arc::new(Mutex::new(connection)),
            database_path: database_path.to_string(),
            stats: Arc::new(Mutex::new(VaultStats::new())),
        };

        // Initialize database schema
        vault.initialize_schema().await?;

        info!("Local vault initialized at {}", database_path);
        Ok(vault)
    }

    /// Initialize database schema
    async fn initialize_schema(&self) -> SecureStorageResult<()> {
        let connection = self.connection.lock().await;

        // Enable WAL mode for better concurrency
        let _: String = connection
            .query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))
            .map_err(|e| SecureStorageError::Database {
                operation: "pragma_wal".to_string(),
                reason: e.to_string(),
            })?;

        // Enable foreign keys
        connection
            .execute("PRAGMA foreign_keys=ON", [])
            .map_err(|e| SecureStorageError::Database {
                operation: "pragma_foreign_keys".to_string(),
                reason: e.to_string(),
            })?;

        // Create vault_entries table
        connection
            .execute(
                r"
            CREATE TABLE IF NOT EXISTS vault_entries (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                accessed_at TEXT,
                expires_at TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                tags TEXT -- JSON encoded tags
            )
            ",
                [],
            )
            .map_err(|e| SecureStorageError::Database {
                operation: "create_table_vault_entries".to_string(),
                reason: e.to_string(),
            })?;

        // Create index on key prefix for efficient listing
        connection
            .execute(
                "CREATE INDEX IF NOT EXISTS idx_vault_entries_key_prefix ON vault_entries(key)",
                [],
            )
            .map_err(|e| SecureStorageError::Database {
                operation: "create_index_key_prefix".to_string(),
                reason: e.to_string(),
            })?;

        // Create index on expiration for cleanup
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_vault_entries_expires_at ON vault_entries(expires_at)",
            [],
        ).map_err(|e| {
            SecureStorageError::Database {
                operation: "create_index_expires_at".to_string(),
                reason: e.to_string(),
            }
        })?;
        drop(connection);

        debug!("Database schema initialized");
        Ok(())
    }

    /// Convert database row to `VaultMetadata`
    fn row_to_metadata(row: &Row<'_>) -> rusqlite::Result<VaultMetadata> {
        let tags_json: Option<String> = row.get("tags")?;
        let tags = tags_json.map_or_else(HashMap::new, |json| {
            serde_json::from_str(&json).unwrap_or_else(|_| HashMap::new())
        });

        Ok(VaultMetadata {
            created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>("created_at")?)
                .map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        0,
                        "created_at".to_string(),
                        rusqlite::types::Type::Text,
                    )
                })?
                .with_timezone(&Utc),
            modified_at: DateTime::parse_from_rfc3339(&row.get::<_, String>("modified_at")?)
                .map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        0,
                        "modified_at".to_string(),
                        rusqlite::types::Type::Text,
                    )
                })?
                .with_timezone(&Utc),
            accessed_at: row
                .get::<_, Option<String>>("accessed_at")?
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            content_type: row.get("content_type")?,
            size: usize::try_from(row.get::<_, i64>("size")?).map_err(|_| {
                rusqlite::Error::InvalidColumnType(
                    151,
                    "size".to_string(),
                    rusqlite::types::Type::Integer,
                )
            })?,
            tags,
            expires_at: row
                .get::<_, Option<String>>("expires_at")?
                .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| dt.with_timezone(&Utc)),
            version: u64::try_from(row.get::<_, i64>("version")?).map_err(|_| {
                rusqlite::Error::InvalidColumnType(
                    159,
                    "version".to_string(),
                    rusqlite::types::Type::Integer,
                )
            })?,
        })
    }

    /// Update statistics
    async fn update_stats(&self, operation: &str, duration: std::time::Duration, success: bool) {
        let mut stats = self.stats.lock().await;
        stats.operations_count += 1;

        // Update average response time - use saturating conversion for safety
        let new_time = f64::from(
            u32::try_from(duration.as_millis().min(u128::from(u32::MAX))).unwrap_or(u32::MAX),
        );
        let ops_count_f64 = f64::from(
            u32::try_from(stats.operations_count.min(u64::from(u32::MAX))).unwrap_or(u32::MAX),
        );
        let prev_ops_f64 = f64::from(
            u32::try_from(
                stats
                    .operations_count
                    .saturating_sub(1)
                    .min(u64::from(u32::MAX)),
            )
            .unwrap_or(u32::MAX),
        );

        stats.avg_response_time_ms =
            stats.avg_response_time_ms.mul_add(prev_ops_f64, new_time) / ops_count_f64;

        // Update error rate
        if success {
            stats.error_rate = (stats.error_rate * prev_ops_f64) / ops_count_f64;
        } else {
            let errors = stats.error_rate.mul_add(prev_ops_f64, 1.0_f64);
            stats.error_rate = errors / ops_count_f64;
        }

        stats.timestamp = Utc::now();
        drop(stats);
        debug!(
            "Updated stats for operation {}: {}ms, success: {}",
            operation, new_time, success
        );
    }

    /// Clean up expired entries
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub async fn cleanup_expired(&self) -> SecureStorageResult<u64> {
        let start = std::time::Instant::now();

        let now = Utc::now().to_rfc3339();
        let deleted = {
            let connection = self.connection.lock().await;
            connection
                .execute(
                    "DELETE FROM vault_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
                    params![now],
                )
                .map_err(|e| SecureStorageError::Database {
                    operation: "cleanup_expired".to_string(),
                    reason: e.to_string(),
                })?
        };
        let deleted = u64::try_from(deleted).unwrap_or(0);

        self.update_stats("cleanup_expired", start.elapsed(), true)
            .await;

        if deleted > 0 {
            info!("Cleaned up {} expired entries", deleted);
        }

        Ok(deleted)
    }
}

#[async_trait]
impl Vault for LocalVault {
    async fn store(&self, key: &str, value: &[u8]) -> SecureStorageResult<()> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            super::utils::validate_key(key)?;
            super::utils::validate_value_size(value, 100 * 1024 * 1024)?; // 100MB limit

            let now = Utc::now().to_rfc3339();

            {
                let connection = self.connection.lock().await;
                connection
                    .execute(
                        r"
                    INSERT OR REPLACE INTO vault_entries
                    (key, value, content_type, size, created_at, modified_at, version)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    ",
                        params![
                            key,
                            value,
                            "application/octet-stream",
                            i64::try_from(value.len()).unwrap_or(i64::MAX),
                            now,
                            now,
                        ],
                    )
                    .map_err(|e| SecureStorageError::Database {
                        operation: "store".to_string(),
                        reason: e.to_string(),
                    })?;
            }

            success = true;
            debug!("Stored key '{}' ({} bytes)", key, value.len());
            Ok(())
        }
        .await;

        self.update_stats("store", start.elapsed(), success).await;
        result
    }

    async fn retrieve(&self, key: &str) -> SecureStorageResult<Vec<u8>> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            super::utils::validate_key(key)?;

            // Update access time and get value
            let now = Utc::now().to_rfc3339();
            let value: Vec<u8> = {
                let connection = self.connection.lock().await;
                connection.execute(
                    "UPDATE vault_entries SET accessed_at = ? WHERE key = ?",
                    params![now, key],
                ).map_err(|e| {
                    SecureStorageError::Database {
                        operation: "update_access_time".to_string(),
                        reason: e.to_string(),
                    }
                })?;

                connection.query_row(
                    "SELECT value FROM vault_entries WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                    params![key, now],
                    |row| row.get(0),
                ).optional().map_err(|e| {
                    SecureStorageError::Database {
                        operation: "retrieve".to_string(),
                        reason: e.to_string(),
                    }
                })?.ok_or_else(|| {
                    SecureStorageError::NotFound {
                        resource: "vault_entry".to_string(),
                        identifier: key.to_string(),
                    }
                })?
            };

            success = true;
            debug!("Retrieved key '{}' ({} bytes)", key, value.len());
            Ok(value)
        }.await;

        self.update_stats("retrieve", start.elapsed(), success)
            .await;
        result
    }

    async fn delete(&self, key: &str) -> SecureStorageResult<()> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            super::utils::validate_key(key)?;

            let deleted = {
                let connection = self.connection.lock().await;
                connection
                    .execute("DELETE FROM vault_entries WHERE key = ?", params![key])
                    .map_err(|e| SecureStorageError::Database {
                        operation: "delete".to_string(),
                        reason: e.to_string(),
                    })?
            };

            if deleted == 0 {
                return Err(SecureStorageError::NotFound {
                    resource: "vault_entry".to_string(),
                    identifier: key.to_string(),
                });
            }

            success = true;
            debug!("Deleted key '{}'", key);
            Ok(())
        }
        .await;

        self.update_stats("delete", start.elapsed(), success).await;
        result
    }

    async fn list_keys(&self, prefix: &str) -> SecureStorageResult<Vec<String>> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            let now = Utc::now().to_rfc3339();

            let pattern = format!("{prefix}%");
            let connection = self.connection.lock().await;
            let mut stmt = connection.prepare(
                "SELECT key FROM vault_entries WHERE key LIKE ? AND (expires_at IS NULL OR expires_at > ?) ORDER BY key"
            ).map_err(|e| {
                SecureStorageError::Database {
                    operation: "prepare_list_keys".to_string(),
                    reason: e.to_string(),
                }
            })?;

            let keys: Result<Vec<String>, rusqlite::Error> = stmt.query_map(
                params![pattern, now],
                |row| row.get(0),
            )?.collect();

            let keys = keys.map_err(|e| {
                SecureStorageError::Database {
                    operation: "list_keys".to_string(),
                    reason: e.to_string(),
                }
            })?;
            drop(stmt);
            drop(connection);

            success = true;
            debug!("Listed {} keys with prefix '{}'", keys.len(), prefix);
            Ok(keys)
        }.await;

        self.update_stats("list_keys", start.elapsed(), success)
            .await;
        result
    }

    async fn exists(&self, key: &str) -> SecureStorageResult<bool> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            super::utils::validate_key(key)?;

            let now = Utc::now().to_rfc3339();

            let exists: bool = {
                let connection = self.connection.lock().await;
                connection.query_row(
                    "SELECT 1 FROM vault_entries WHERE key = ? AND (expires_at IS NULL OR expires_at > ?) LIMIT 1",
                    params![key, now],
                    |_| Ok(true),
                ).optional().map_err(|e| {
                    SecureStorageError::Database {
                        operation: "exists".to_string(),
                        reason: e.to_string(),
                    }
                })?.unwrap_or(false)
            };

            success = true;
            debug!("Key '{}' exists: {}", key, exists);
            Ok(exists)
        }.await;

        self.update_stats("exists", start.elapsed(), success).await;
        result
    }

    async fn get_metadata(&self, key: &str) -> SecureStorageResult<VaultMetadata> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            super::utils::validate_key(key)?;

            let now = Utc::now().to_rfc3339();

            let metadata = {
                let connection = self.connection.lock().await;
                connection.query_row(
                    r"
                    SELECT content_type, size, created_at, modified_at, accessed_at, expires_at, version, tags
                    FROM vault_entries
                    WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)
                    ",
                    params![key, now],
                    Self::row_to_metadata,
                ).optional().map_err(|e| {
                    SecureStorageError::Database {
                        operation: "get_metadata".to_string(),
                        reason: e.to_string(),
                    }
                })?.ok_or_else(|| {
                    SecureStorageError::NotFound {
                        resource: "vault_entry".to_string(),
                        identifier: key.to_string(),
                    }
                })?
            };

            success = true;
            debug!("Retrieved metadata for key '{}'", key);
            Ok(metadata)
        }.await;

        self.update_stats("get_metadata", start.elapsed(), success)
            .await;
        result
    }

    async fn store_with_metadata(
        &self,
        key: &str,
        value: &[u8],
        metadata: VaultMetadata,
    ) -> SecureStorageResult<()> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            super::utils::validate_key(key)?;
            super::utils::validate_value_size(value, 100 * 1024 * 1024)?;

            let tags_json = serde_json::to_string(&metadata.tags).unwrap_or_else(|_| String::new());

            {
                let connection = self.connection.lock().await;
                connection.execute(
                    r"
                    INSERT OR REPLACE INTO vault_entries
                    (key, value, content_type, size, created_at, modified_at, accessed_at, expires_at, version, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ",
                    params![
                        key,
                        value,
                        metadata.content_type,
                        i64::try_from(value.len()).unwrap_or(i64::MAX),
                        metadata.created_at.to_rfc3339(),
                        metadata.modified_at.to_rfc3339(),
                        metadata.accessed_at.map(|dt| dt.to_rfc3339()),
                        metadata.expires_at.map(|dt| dt.to_rfc3339()),
                        i64::try_from(metadata.version).unwrap_or(i64::MAX),
                        tags_json,
                    ],
                ).map_err(|e| {
                    SecureStorageError::Database {
                        operation: "store_with_metadata".to_string(),
                        reason: e.to_string(),
                    }
                })?;
            }

            success = true;
            debug!("Stored key '{}' with metadata ({} bytes)", key, value.len());
            Ok(())
        }.await;

        self.update_stats("store_with_metadata", start.elapsed(), success)
            .await;
        result
    }

    async fn batch_store(&self, items: Vec<(String, Vec<u8>)>) -> SecureStorageResult<()> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            let now = Utc::now().to_rfc3339();

            let connection = self.connection.lock().await;
            let tx =
                connection
                    .unchecked_transaction()
                    .map_err(|e| SecureStorageError::Database {
                        operation: "begin_transaction".to_string(),
                        reason: e.to_string(),
                    })?;

            for (key, value) in &items {
                super::utils::validate_key(key)?;
                super::utils::validate_value_size(value, 100 * 1024 * 1024)?;

                tx.execute(
                    r"
                    INSERT OR REPLACE INTO vault_entries
                    (key, value, content_type, size, created_at, modified_at, version)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    ",
                    params![
                        key,
                        value,
                        "application/octet-stream",
                        i64::try_from(value.len()).unwrap_or(i64::MAX),
                        now,
                        now,
                    ],
                )
                .map_err(|e| SecureStorageError::Database {
                    operation: "batch_store_item".to_string(),
                    reason: e.to_string(),
                })?;
            }

            tx.commit().map_err(|e| SecureStorageError::Database {
                operation: "commit_transaction".to_string(),
                reason: e.to_string(),
            })?;
            drop(connection);

            success = true;
            debug!("Batch stored {} items", items.len());
            Ok(())
        }
        .await;

        self.update_stats("batch_store", start.elapsed(), success)
            .await;
        result
    }

    async fn batch_retrieve(
        &self,
        keys: Vec<String>,
    ) -> SecureStorageResult<HashMap<String, Vec<u8>>> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            let now = Utc::now().to_rfc3339();
            let mut results = HashMap::new();

            let connection = self.connection.lock().await;
            for key in keys {
                super::utils::validate_key(&key)?;

                let value: Option<Vec<u8>> = connection.query_row(
                    "SELECT value FROM vault_entries WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                    params![key, now],
                    |row| row.get(0),
                ).optional().map_err(|e| {
                    SecureStorageError::Database {
                        operation: "batch_retrieve_item".to_string(),
                        reason: e.to_string(),
                    }
                })?;

                if let Some(value) = value {
                    results.insert(key, value);
                }
            }
            drop(connection);

            success = true;
            debug!("Batch retrieved {} items", results.len());
            Ok(results)
        }.await;

        self.update_stats("batch_retrieve", start.elapsed(), success)
            .await;
        result
    }

    async fn batch_delete(&self, keys: Vec<String>) -> SecureStorageResult<()> {
        let start = std::time::Instant::now();
        let mut success = false;

        let result = async {
            let connection = self.connection.lock().await;
            let tx =
                connection
                    .unchecked_transaction()
                    .map_err(|e| SecureStorageError::Database {
                        operation: "begin_transaction".to_string(),
                        reason: e.to_string(),
                    })?;

            for key in &keys {
                super::utils::validate_key(key)?;

                tx.execute("DELETE FROM vault_entries WHERE key = ?", params![key])
                    .map_err(|e| SecureStorageError::Database {
                        operation: "batch_delete_item".to_string(),
                        reason: e.to_string(),
                    })?;
            }

            tx.commit().map_err(|e| SecureStorageError::Database {
                operation: "commit_transaction".to_string(),
                reason: e.to_string(),
            })?;
            drop(connection);

            success = true;
            debug!("Batch deleted {} items", keys.len());
            Ok(())
        }
        .await;

        self.update_stats("batch_delete", start.elapsed(), success)
            .await;
        result
    }

    async fn health_check(&self) -> SecureStorageResult<VaultHealth> {
        let start = std::time::Instant::now();

        let connection_result = {
            let connection = self.connection.lock().await;
            connection.query_row("SELECT 1", [], |_| Ok(()))
        };
        let status = match connection_result {
            Ok(()) => HealthStatus::Healthy,
            Err(_) => HealthStatus::Critical,
        };

        let response_time_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        let mut details = HashMap::new();
        details.insert("database_path".to_string(), self.database_path.clone());
        details.insert("vault_type".to_string(), "local_sqlite".to_string());

        Ok(VaultHealth {
            status,
            checked_at: Utc::now(),
            response_time_ms,
            details,
        })
    }

    async fn get_stats(&self) -> SecureStorageResult<VaultStats> {
        // Get total entries and size
        let (total_entries, total_size): (i64, i64) = {
            let connection = self.connection.lock().await;
            connection
                .query_row(
                    "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM vault_entries",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(|e| SecureStorageError::Database {
                    operation: "get_stats".to_string(),
                    reason: e.to_string(),
                })?
        };

        let mut stats = self.stats.lock().await.clone();
        stats.total_entries = u64::try_from(total_entries).unwrap_or(0);
        stats.total_size_bytes = u64::try_from(total_size).unwrap_or(0);

        Ok(stats)
    }
}
