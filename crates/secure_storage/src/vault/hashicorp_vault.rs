//! `HashiCorp` Vault implementation

#[cfg(feature = "vault")]
use super::{HealthStatus, Vault, VaultHealth, VaultMetadata, VaultStats};
#[cfg(feature = "vault")]
use crate::error::{SecureStorageError, SecureStorageResult};
#[cfg(feature = "vault")]
use async_trait::async_trait;
#[cfg(feature = "vault")]
use std::collections::HashMap;
#[cfg(feature = "vault")]
use vaultrs::{
    client::{Client, VaultClient, VaultClientSettings},
    kv2,
};

/// `HashiCorp` Vault implementation
#[cfg(feature = "vault")]
/// TODO: Add documentation
pub struct HashiCorpVault {
    client: VaultClient,
    mount_path: String,
}

#[cfg(feature = "vault")]
impl HashiCorpVault {
    /// Create a new `HashiCorp` Vault instance
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn new(url: &str, token: &str, mount_path: &str) -> SecureStorageResult<Self> {
        use url::Url;

        let address = Url::parse(url).map_err(|e| SecureStorageError::Network {
            reason: format!("Invalid vault URL: {e}"),
        })?;

        let settings = VaultClientSettings {
            address,
            token: token.to_string(),
            ca_certs: Vec::with_capacity(0),
            verify: true,
            timeout: None,
            version: 1,
            namespace: None,
            wrapping: false,
            identity: None,
        };

        let client = VaultClient::new(settings).map_err(|e| SecureStorageError::Network {
            reason: format!("Failed to create vault client: {e}"),
        })?;

        Ok(Self {
            client,
            mount_path: mount_path.to_string(),
        })
    }
}

#[cfg(feature = "vault")]
#[async_trait]
impl Vault for HashiCorpVault {
    async fn store(&self, key: &str, value: &[u8]) -> SecureStorageResult<()> {
        use base64::{engine::general_purpose, Engine as _};
        let data = HashMap::from([("data".to_string(), general_purpose::STANDARD.encode(value))]);

        kv2::set(&self.client, &self.mount_path, key, &data)
            .await
            .map_err(|e| SecureStorageError::Vault {
                operation: "store".to_string(),
                reason: e.to_string(),
            })?;

        Ok(())
    }

    async fn retrieve(&self, key: &str) -> SecureStorageResult<Vec<u8>> {
        use base64::{engine::general_purpose, Engine as _};

        let secret: HashMap<String, String> = kv2::read(&self.client, &self.mount_path, key)
            .await
            .map_err(|e| SecureStorageError::Vault {
                operation: "retrieve".to_string(),
                reason: e.to_string(),
            })?;

        let data_str = secret
            .get("data")
            .ok_or_else(|| SecureStorageError::NotFound {
                resource: "vault_entry".to_string(),
                identifier: key.to_string(),
            })?;

        let data =
            general_purpose::STANDARD
                .decode(data_str)
                .map_err(|e| SecureStorageError::Vault {
                    operation: "retrieve".to_string(),
                    reason: format!("Base64 decode error: {e}"),
                })?;

        Ok(data)
    }

    async fn delete(&self, key: &str) -> SecureStorageResult<()> {
        kv2::delete_latest(&self.client, &self.mount_path, key)
            .await
            .map_err(|e| SecureStorageError::Vault {
                operation: "delete".to_string(),
                reason: e.to_string(),
            })?;

        Ok(())
    }

    async fn list_keys(&self, prefix: &str) -> SecureStorageResult<Vec<String>> {
        let keys = kv2::list(&self.client, &self.mount_path, prefix)
            .await
            .map_err(|e| SecureStorageError::Vault {
                operation: "list_keys".to_string(),
                reason: e.to_string(),
            })?;

        Ok(keys)
    }

    async fn exists(&self, key: &str) -> SecureStorageResult<bool> {
        let result: Result<HashMap<String, String>, _> =
            kv2::read(&self.client, &self.mount_path, key).await;
        match result {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn get_metadata(&self, _key: &str) -> SecureStorageResult<VaultMetadata> {
        // HashiCorp Vault doesn't provide detailed metadata in the same way
        // This would need to be implemented based on specific requirements
        Err(SecureStorageError::Vault {
            operation: "get_metadata".to_string(),
            reason: "Metadata not supported for HashiCorp Vault".to_string(),
        })
    }

    async fn store_with_metadata(
        &self,
        key: &str,
        value: &[u8],
        _metadata: VaultMetadata,
    ) -> SecureStorageResult<()> {
        // For now, just store the value without metadata
        self.store(key, value).await
    }

    async fn batch_store(&self, items: Vec<(String, Vec<u8>)>) -> SecureStorageResult<()> {
        for (key, value) in items {
            self.store(&key, &value).await?;
        }
        Ok(())
    }

    async fn batch_retrieve(
        &self,
        keys: Vec<String>,
    ) -> SecureStorageResult<HashMap<String, Vec<u8>>> {
        let mut results = HashMap::new();
        for key in keys {
            if let Ok(value) = self.retrieve(&key).await {
                results.insert(key, value);
            }
        }
        Ok(results)
    }

    async fn batch_delete(&self, keys: Vec<String>) -> SecureStorageResult<()> {
        for key in keys {
            self.delete(&key).await?;
        }
        Ok(())
    }

    async fn health_check(&self) -> SecureStorageResult<VaultHealth> {
        let start = std::time::Instant::now();

        // Try to read vault health
        let status = match Client::status(&self.client).await {
            Ok(_) => HealthStatus::Healthy,
            Err(_) => HealthStatus::Critical,
        };

        let response_time_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        let mut details = HashMap::new();
        details.insert("vault_type".to_string(), "hashicorp".to_string());
        details.insert("mount_path".to_string(), self.mount_path.clone());

        Ok(VaultHealth {
            status,
            checked_at: chrono::Utc::now(),
            response_time_ms,
            details,
        })
    }

    async fn get_stats(&self) -> SecureStorageResult<VaultStats> {
        // HashiCorp Vault doesn't provide detailed statistics in the same way
        // This would need to be implemented based on specific requirements
        Ok(VaultStats::new())
    }
}

#[cfg(not(feature = "vault"))]
/// TODO: Add documentation
pub struct HashiCorpVault;

#[cfg(not(feature = "vault"))]
impl HashiCorpVault {
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub async fn new(_url: &str, _token: &str, _mount_path: &str) -> SecureStorageResult<Self> {
        Err(SecureStorageError::Configuration {
            field: "vault_feature".to_string(),
            reason: "HashiCorp Vault support not enabled (missing 'vault' feature)".to_string(),
        })
    }
}
