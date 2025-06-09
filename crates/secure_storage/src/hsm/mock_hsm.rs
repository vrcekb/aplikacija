//! # Mock HSM Implementation
//!
//! Mock HSM for testing and development with realistic performance characteristics.

use super::{
    EncryptionAlgorithm, HsmCredentials, HsmHealth, HsmKeyInfo, HsmKeyType, HsmMetrics,
    HsmOperationResult, HsmProvider, SigningAlgorithm,
};
use crate::error::{SecureStorageError, SecureStorageResult};
use crate::memory::secure_buffer::SecureBuffer;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Mock HSM implementation for testing
pub struct MockHsm {
    initialized: bool,
    keys: Arc<RwLock<HashMap<String, MockKey>>>,
    session_id: String,
    operation_count: Arc<RwLock<u64>>,
    start_time: Instant,
}

#[derive(Debug, Clone)]
struct MockKey {
    info: HsmKeyInfo,
    key_material: Vec<u8>,
    #[allow(dead_code)] // Used for HSM compliance tracking
    extractable: bool,
}

impl MockHsm {
    /// Create a new mock HSM instance
    #[must_use]
    /// TODO: Add documentation
    pub fn new() -> Self {
        Self {
            initialized: false,
            keys: Arc::new(RwLock::new(HashMap::new())),
            session_id: Uuid::new_v4().to_string(),
            operation_count: Arc::new(RwLock::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Simulate HSM operation latency
    async fn simulate_latency(&self, operation: &str) {
        let delay = match operation {
            "sign" => Duration::from_millis(25), // Realistic HSM signing latency
            "verify" => Duration::from_millis(15),
            "encrypt" => Duration::from_millis(8),
            "decrypt" => Duration::from_millis(10),
            "generate_key" => Duration::from_millis(100),
            "import_key" => Duration::from_millis(50),
            _ => Duration::from_millis(5),
        };

        tokio::time::sleep(delay).await;
    }

    /// Increment operation counter
    async fn increment_operations(&self) {
        let mut count = self.operation_count.write().await;
        *count += 1;
    }

    /// Generate mock key material
    fn generate_key_material(&self, key_type: &HsmKeyType) -> Vec<u8> {
        let size = match key_type {
            HsmKeyType::Rsa2048 => 256, // 2048 bits = 256 bytes
            HsmKeyType::Rsa4096 => 512, // 4096 bits = 512 bytes
            HsmKeyType::EcdsaSecp256k1
            | HsmKeyType::EcdsaSecp256r1
            | HsmKeyType::Aes256
            | HsmKeyType::Ed25519 => 32, // 256 bits = 32 bytes
        };

        // Generate deterministic but unique key material

        let mut hasher = DefaultHasher::new();
        key_type.to_string().hash(&mut hasher);
        self.session_id.hash(&mut hasher);
        Instant::now().elapsed().as_nanos().hash(&mut hasher);

        let hash = hasher.finish();
        let hash_bytes = hash.to_le_bytes();
        let mut key_material = vec![0u8; size];

        // Fill with repeated hash
        for (i, byte) in key_material.iter_mut().enumerate() {
            *byte = hash_bytes[i % hash_bytes.len()];
        }

        key_material
    }
}

impl Default for MockHsm {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl HsmProvider for MockHsm {
    async fn initialize(&mut self, credentials: &HsmCredentials) -> SecureStorageResult<()> {
        // Simulate authentication delay
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Basic credential validation
        if credentials.username.is_empty() {
            return Err(SecureStorageError::Authentication {
                reason: "Username cannot be empty".to_string(),
            });
        }

        if credentials.password.is_empty() {
            return Err(SecureStorageError::Authentication {
                reason: "Password cannot be empty".to_string(),
            });
        }

        self.initialized = true;
        tracing::info!("Mock HSM initialized successfully");

        Ok(())
    }

    async fn generate_key(
        &self,
        key_type: HsmKeyType,
        label: &str,
        extractable: bool,
    ) -> SecureStorageResult<HsmOperationResult<HsmKeyInfo>> {
        if !self.initialized {
            return Err(SecureStorageError::Hsm {
                operation: "generate_key".to_string(),
                reason: "HSM not initialized".to_string(),
            });
        }

        let start = Instant::now();
        self.simulate_latency("generate_key").await;
        self.increment_operations().await;

        let key_id = Uuid::new_v4().to_string();
        let key_material = self.generate_key_material(&key_type);

        let key_info = HsmKeyInfo {
            key_id: key_id.clone(),
            key_type: key_type.clone(),
            created_at: Instant::now(),
            usage_count: 0,
            max_usage: None,
            expires_at: None,
            label: label.to_string(),
        };

        let mock_key = MockKey {
            info: key_info.clone(),
            key_material,
            extractable,
        };

        {
            let mut keys = self.keys.write().await;
            keys.insert(key_id, mock_key);
        }

        Ok(HsmOperationResult {
            result: key_info,
            duration: start.elapsed(),
            hsm_session_id: self.session_id.clone(),
        })
    }

    async fn import_key(
        &self,
        key_material: &SecureBuffer,
        key_type: HsmKeyType,
        label: &str,
    ) -> SecureStorageResult<HsmOperationResult<HsmKeyInfo>> {
        if !self.initialized {
            return Err(SecureStorageError::Hsm {
                operation: "import_key".to_string(),
                reason: "HSM not initialized".to_string(),
            });
        }

        let start = Instant::now();
        self.simulate_latency("import_key").await;
        self.increment_operations().await;

        let key_id = Uuid::new_v4().to_string();

        let key_info = HsmKeyInfo {
            key_id: key_id.clone(),
            key_type: key_type.clone(),
            created_at: Instant::now(),
            usage_count: 0,
            max_usage: None,
            expires_at: None,
            label: label.to_string(),
        };

        let mock_key = MockKey {
            info: key_info.clone(),
            key_material: key_material.read().to_vec(),
            extractable: false, // Imported keys are typically non-extractable
        };

        {
            let mut keys = self.keys.write().await;
            keys.insert(key_id, mock_key);
        }

        Ok(HsmOperationResult {
            result: key_info,
            duration: start.elapsed(),
            hsm_session_id: self.session_id.clone(),
        })
    }

    async fn sign(
        &self,
        key_id: &str,
        data: &[u8],
        _algorithm: SigningAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<Vec<u8>>> {
        if !self.initialized {
            return Err(SecureStorageError::Hsm {
                operation: "sign".to_string(),
                reason: "HSM not initialized".to_string(),
            });
        }

        let start = Instant::now();
        self.simulate_latency("sign").await;
        self.increment_operations().await;

        let key = {
            let keys = self.keys.read().await;
            keys.get(key_id)
                .cloned()
                .ok_or_else(|| SecureStorageError::NotFound {
                    resource: "hsm_key".to_string(),
                    identifier: key_id.to_string(),
                })?
        };

        // Mock signature: simple hash of key_material + data

        let mut hasher = DefaultHasher::new();
        key.key_material.hash(&mut hasher);
        data.hash(&mut hasher);
        let hash = hasher.finish();
        let signature = hash.to_le_bytes().to_vec();

        Ok(HsmOperationResult {
            result: signature,
            duration: start.elapsed(),
            hsm_session_id: self.session_id.clone(),
        })
    }

    async fn verify(
        &self,
        key_id: &str,
        data: &[u8],
        signature: &[u8],
        _algorithm: SigningAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<bool>> {
        if !self.initialized {
            return Err(SecureStorageError::Hsm {
                operation: "verify".to_string(),
                reason: "HSM not initialized".to_string(),
            });
        }

        let start = Instant::now();
        self.simulate_latency("verify").await;
        self.increment_operations().await;

        let key = {
            let keys = self.keys.read().await;
            keys.get(key_id)
                .cloned()
                .ok_or_else(|| SecureStorageError::NotFound {
                    resource: "hsm_key".to_string(),
                    identifier: key_id.to_string(),
                })?
        };

        // Mock verification: regenerate signature and compare

        let mut hasher = DefaultHasher::new();
        key.key_material.hash(&mut hasher);
        data.hash(&mut hasher);
        let hash = hasher.finish();
        let expected_signature = hash.to_le_bytes();

        let is_valid = expected_signature.as_slice() == signature;

        Ok(HsmOperationResult {
            result: is_valid,
            duration: start.elapsed(),
            hsm_session_id: self.session_id.clone(),
        })
    }

    async fn encrypt(
        &self,
        key_id: &str,
        data: &[u8],
        _algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<Vec<u8>>> {
        if !self.initialized {
            return Err(SecureStorageError::Hsm {
                operation: "encrypt".to_string(),
                reason: "HSM not initialized".to_string(),
            });
        }

        let start = Instant::now();
        self.simulate_latency("encrypt").await;
        self.increment_operations().await;

        let key = {
            let keys = self.keys.read().await;
            keys.get(key_id)
                .cloned()
                .ok_or_else(|| SecureStorageError::NotFound {
                    resource: "hsm_key".to_string(),
                    identifier: key_id.to_string(),
                })?
        };

        // Mock encryption: XOR with key material (repeated)
        let mut encrypted = Vec::with_capacity(data.len());
        for (i, byte) in data.iter().enumerate() {
            let key_byte = key.key_material[i % key.key_material.len()];
            encrypted.push(byte ^ key_byte);
        }

        Ok(HsmOperationResult {
            result: encrypted,
            duration: start.elapsed(),
            hsm_session_id: self.session_id.clone(),
        })
    }

    async fn decrypt(
        &self,
        key_id: &str,
        encrypted_data: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<Vec<u8>>> {
        // For mock implementation, decryption is same as encryption (XOR is reversible)
        self.encrypt(key_id, encrypted_data, algorithm).await
    }

    async fn delete_key(&self, key_id: &str) -> SecureStorageResult<HsmOperationResult<()>> {
        if !self.initialized {
            return Err(SecureStorageError::Hsm {
                operation: "delete_key".to_string(),
                reason: "HSM not initialized".to_string(),
            });
        }

        let start = Instant::now();
        self.simulate_latency("delete_key").await;
        self.increment_operations().await;

        {
            let mut keys = self.keys.write().await;
            keys.remove(key_id)
                .ok_or_else(|| SecureStorageError::NotFound {
                    resource: "hsm_key".to_string(),
                    identifier: key_id.to_string(),
                })?;
        }

        Ok(HsmOperationResult {
            result: (),
            duration: start.elapsed(),
            hsm_session_id: self.session_id.clone(),
        })
    }

    async fn list_keys(&self) -> SecureStorageResult<HsmOperationResult<Vec<HsmKeyInfo>>> {
        if !self.initialized {
            return Err(SecureStorageError::Hsm {
                operation: "list_keys".to_string(),
                reason: "HSM not initialized".to_string(),
            });
        }

        let start = Instant::now();
        self.simulate_latency("list_keys").await;
        self.increment_operations().await;

        let key_infos: Vec<HsmKeyInfo> = {
            let keys = self.keys.read().await;
            keys.values().map(|k| k.info.clone()).collect()
        };

        Ok(HsmOperationResult {
            result: key_infos,
            duration: start.elapsed(),
            hsm_session_id: self.session_id.clone(),
        })
    }

    async fn health_check(&self) -> SecureStorageResult<HsmHealth> {
        Ok(HsmHealth {
            is_available: self.initialized,
            session_count: 1,
            free_memory: 1024 * 1024 * 1024, // 1GB mock free memory
            temperature: Some(45.0),         // Mock temperature
            last_error: None,
        })
    }

    async fn get_metrics(&self) -> SecureStorageResult<HsmMetrics> {
        let operation_count = *self.operation_count.read().await;
        let elapsed = self.start_time.elapsed();
        let ops_per_second = if elapsed.as_secs() > 0 {
            let count_f64 = f64::from(
                u32::try_from(operation_count.min(u64::from(u32::MAX))).unwrap_or(u32::MAX),
            );
            count_f64 / elapsed.as_secs_f64()
        } else {
            0.0_f64
        };

        Ok(HsmMetrics {
            operations_per_second: ops_per_second,
            average_latency_ms: 25.0, // Mock average latency
            error_rate: 0.0,          // Mock HSM has no errors
            active_sessions: 1,
            total_operations: operation_count,
        })
    }
}
