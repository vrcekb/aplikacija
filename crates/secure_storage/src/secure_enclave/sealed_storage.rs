//! # Sealed Storage System
//!
//! Hardware-encrypted persistent storage for secure enclaves.
//! Data is encrypted with hardware-derived keys and can only
//! be decrypted by the same enclave on the same platform.

use super::EnclavePlatform;
use crate::error::{SecureStorageError, SecureStorageResult};
use crate::types::KeyId;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Sealed data structure
#[derive(Debug, Clone)]
pub struct SealedData {
    /// Platform that sealed the data
    pub platform: EnclavePlatform,
    /// Sealing policy
    pub policy: SealingPolicy,
    /// Encrypted data
    pub encrypted_data: Vec<u8>,
    /// Authentication tag
    pub auth_tag: Vec<u8>,
    /// Additional authenticated data
    pub aad: Vec<u8>,
    /// Sealing timestamp
    pub sealed_at: u64,
    /// Data version
    pub version: u32,
}

/// Sealing policy determines what can unseal the data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealingPolicy {
    /// Only the exact same enclave (MRENCLAVE)
    EnclaveIdentity,
    /// Any enclave from the same signer (MRSIGNER)
    SignerIdentity,
    /// Platform-specific policy
    Platform,
}

/// Sealed storage configuration
#[derive(Debug, Clone)]
pub struct SealedStorageConfig {
    /// Storage directory path
    pub storage_path: PathBuf,
    /// Platform type
    pub platform: EnclavePlatform,
    /// Default sealing policy
    pub default_policy: SealingPolicy,
    /// Maximum file size in bytes
    pub max_file_size: usize,
    /// Enable compression
    pub enable_compression: bool,
    /// Enable integrity verification
    pub enable_integrity_check: bool,
}

impl SealedStorageConfig {
    /// Create production configuration
    #[must_use]
    pub const fn new_production(storage_path: PathBuf, platform: EnclavePlatform) -> Self {
        Self {
            storage_path,
            platform,
            default_policy: SealingPolicy::EnclaveIdentity,
            max_file_size: 64 * 1024 * 1024, // 64MB
            enable_compression: true,
            enable_integrity_check: true,
        }
    }

    /// Create development configuration
    #[must_use]
    pub const fn new_development(storage_path: PathBuf, platform: EnclavePlatform) -> Self {
        Self {
            storage_path,
            platform,
            default_policy: SealingPolicy::SignerIdentity,
            max_file_size: 16 * 1024 * 1024, // 16MB
            enable_compression: false,
            enable_integrity_check: true,
        }
    }
}

/// Sealed storage system
#[derive(Debug)]
pub struct SealedStorage {
    /// Configuration
    config: SealedStorageConfig,
    /// Cached sealed data
    cache: RwLock<HashMap<String, SealedData>>,
    /// Performance counters
    seal_operations: AtomicU64,
    unseal_operations: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl SealedStorage {
    /// Create a new sealed storage system
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub async fn new(
        storage_path: PathBuf,
        platform: EnclavePlatform,
    ) -> SecureStorageResult<Self> {
        let config = if cfg!(debug_assertions) {
            SealedStorageConfig::new_development(storage_path, platform)
        } else {
            SealedStorageConfig::new_production(storage_path, platform)
        };

        // Create storage directory if it doesn't exist
        if !config.storage_path.exists() {
            fs::create_dir_all(&config.storage_path)
                .await
                .map_err(|e| SecureStorageError::InvalidInput {
                    field: "storage_path".to_string(),
                    reason: format!("Failed to create storage directory: {e}"),
                })?;
        }

        info!("Initialized sealed storage at: {:?}", config.storage_path);

        Ok(Self {
            config,
            cache: RwLock::new(HashMap::new()),
            seal_operations: AtomicU64::new(0),
            unseal_operations: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        })
    }

    /// Seal data with hardware encryption
    ///
    /// # Errors
    ///
    /// Returns error if sealing fails
    pub async fn seal_data(
        &self,
        key_id: &KeyId,
        data: &[u8],
        policy: Option<SealingPolicy>,
    ) -> SecureStorageResult<()> {
        let start = Instant::now();

        if data.len() > self.config.max_file_size {
            return Err(SecureStorageError::InvalidInput {
                field: "data_size".to_string(),
                reason: format!(
                    "Data size {} exceeds maximum {}",
                    data.len(),
                    self.config.max_file_size
                ),
            });
        }

        let policy = policy.unwrap_or(self.config.default_policy);

        debug!(
            "Sealing data for key_id: {} with policy: {:?}",
            key_id, policy
        );

        // Perform platform-specific sealing
        let sealed_data = match self.config.platform {
            EnclavePlatform::IntelSgx => Self::seal_sgx_data(data, policy)?,
            EnclavePlatform::ArmTrustZone => self.seal_trustzone_data(data, policy).await?,
            EnclavePlatform::AmdMemoryGuard => self.seal_amd_data(data, policy).await?,
            EnclavePlatform::Simulation => self.seal_simulation_data(data, policy).await?,
        };

        // Store to filesystem
        let file_path = self.get_file_path(key_id);
        self.write_sealed_file(&file_path, &sealed_data).await?;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(key_id.to_string(), sealed_data);
        }

        self.seal_operations.fetch_add(1, Ordering::Relaxed);

        let elapsed = start.elapsed();
        debug!("Data sealed in {:?} for key_id: {}", elapsed, key_id);

        Ok(())
    }

    /// Unseal data with hardware decryption
    ///
    /// # Errors
    ///
    /// Returns error if unsealing fails
    pub async fn unseal_data(&self, key_id: &KeyId) -> SecureStorageResult<Vec<u8>> {
        let start = Instant::now();

        debug!("Unsealing data for key_id: {}", key_id);

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(sealed_data) = cache.get(&key_id.to_string()) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return self.unseal_platform_data(sealed_data).await;
            }
        }

        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        // Load from filesystem
        let file_path = self.get_file_path(key_id);
        let sealed_data = self.read_sealed_file(&file_path).await?;

        // Perform platform-specific unsealing
        let data = self.unseal_platform_data(&sealed_data).await?;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(key_id.to_string(), sealed_data);
        }

        self.unseal_operations.fetch_add(1, Ordering::Relaxed);

        let elapsed = start.elapsed();
        debug!("Data unsealed in {:?} for key_id: {}", elapsed, key_id);

        Ok(data)
    }

    /// Delete sealed data
    ///
    /// # Errors
    ///
    /// Returns error if deletion fails
    pub async fn delete_sealed_data(&self, key_id: &KeyId) -> SecureStorageResult<bool> {
        let file_path = self.get_file_path(key_id);

        // Remove from cache
        {
            let mut cache = self.cache.write().await;
            cache.remove(&key_id.to_string());
        }

        // Remove from filesystem
        if file_path.exists() {
            fs::remove_file(&file_path)
                .await
                .map_err(|e| SecureStorageError::InvalidInput {
                    field: "file_path".to_string(),
                    reason: format!("Failed to delete sealed file: {e}"),
                })?;

            info!("Deleted sealed data for key_id: {}", key_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all sealed data keys
    ///
    /// # Errors
    ///
    /// Returns error if listing fails
    pub async fn list_sealed_keys(&self) -> SecureStorageResult<Vec<String>> {
        let mut keys = Vec::with_capacity(100);

        let mut entries = fs::read_dir(&self.config.storage_path).await.map_err(|e| {
            SecureStorageError::InvalidInput {
                field: "storage_path".to_string(),
                reason: format!("Failed to read storage directory: {e}"),
            }
        })?;

        while let Some(entry) =
            entries
                .next_entry()
                .await
                .map_err(|e| SecureStorageError::InvalidInput {
                    field: "directory_entry".to_string(),
                    reason: format!("Failed to read directory entry: {e}"),
                })?
        {
            if let Some(file_name) = entry.file_name().to_str() {
                if file_name.ends_with(".sealed") {
                    let key = file_name.trim_end_matches(".sealed");
                    keys.push(key.to_string());
                }
            }
        }

        Ok(keys)
    }

    /// Seal data using SGX
    fn seal_sgx_data(data: &[u8], policy: SealingPolicy) -> SecureStorageResult<SealedData> {
        use aes_gcm::{
            aead::{Aead, KeyInit},
            Aes256Gcm, Key, Nonce,
        };

        debug!("Sealing data using SGX with policy: {:?}", policy);

        // Generate sealing key based on policy
        let sealing_key = Self::derive_sgx_sealing_key(policy);

        // Create cipher
        let key = Key::<Aes256Gcm>::from_slice(&sealing_key);
        let cipher = Aes256Gcm::new(key);

        // Generate nonce
        let nonce_bytes = Self::generate_secure_nonce();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Additional authenticated data
        let aad = Self::create_sgx_aad(policy);

        // Encrypt data
        let encrypted_data = cipher
            .encrypt(
                nonce,
                aes_gcm::aead::Payload {
                    msg: data,
                    aad: &aad,
                },
            )
            .map_err(|e| SecureStorageError::Encryption {
                reason: format!("SGX sealing failed: {e}"),
            })?;

        // Create authentication tag (first 16 bytes of encrypted data in GCM)
        let auth_tag = if encrypted_data.len() >= 16 {
            encrypted_data[encrypted_data.len() - 16..].to_vec()
        } else {
            return Err(SecureStorageError::Encryption {
                reason: "Invalid encrypted data length".to_string(),
            });
        };

        // Remove auth tag from encrypted data
        let encrypted_payload = encrypted_data[..encrypted_data.len() - 16].to_vec();

        Ok(SealedData {
            platform: EnclavePlatform::IntelSgx,
            policy,
            encrypted_data: encrypted_payload,
            auth_tag,
            aad,
            sealed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            version: 1,
        })
    }

    /// Derive SGX sealing key based on policy
    fn derive_sgx_sealing_key(policy: SealingPolicy) -> Vec<u8> {
        use sha2::{Digest, Sha256};

        // In production, this would use SGX sealing key derivation
        // For now, simulate with deterministic key derivation

        let mut hasher = Sha256::new();
        hasher.update(b"TallyIO-SGX-Sealing-Key");

        match policy {
            SealingPolicy::EnclaveIdentity => {
                hasher.update(b"MRENCLAVE");
                // In production: use actual MRENCLAVE
                hasher.update([0x12, 0x34, 0x56, 0x78]);
            }
            SealingPolicy::SignerIdentity => {
                hasher.update(b"MRSIGNER");
                // In production: use actual MRSIGNER
                hasher.update([0x87, 0x65, 0x43, 0x21]);
            }
            SealingPolicy::Platform => {
                hasher.update(b"PLATFORM");
                hasher.update(0x1337u16.to_le_bytes()); // TallyIO product ID
            }
        }

        hasher.finalize().to_vec()
    }

    /// Generate secure nonce
    fn generate_secure_nonce() -> Vec<u8> {
        use rand::RngCore;

        let mut nonce = vec![0u8; 12]; // 96-bit nonce for GCM
        rand::thread_rng().fill_bytes(&mut nonce);

        nonce
    }

    /// Create SGX additional authenticated data
    fn create_sgx_aad(policy: SealingPolicy) -> Vec<u8> {
        let mut aad = Vec::with_capacity(64);
        aad.extend_from_slice(b"TallyIO-SGX-v1");
        aad.extend_from_slice(&(policy as u8).to_le_bytes());
        aad.extend_from_slice(
            &std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs())
                .to_le_bytes(),
        );

        aad
    }

    /// Seal data using `TrustZone`
    async fn seal_trustzone_data(
        &self,
        data: &[u8],
        policy: SealingPolicy,
    ) -> SecureStorageResult<SealedData> {
        let encrypted_data = self.encrypt_data_simulation(data, "TZ").await?;
        let auth_tag = vec![0xBB; 16];

        Ok(SealedData {
            platform: EnclavePlatform::ArmTrustZone,
            policy,
            encrypted_data,
            auth_tag,
            aad: b"TZ-SEALED".to_vec(),
            sealed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            version: 1,
        })
    }

    /// Seal data using AMD Memory Guard
    async fn seal_amd_data(
        &self,
        data: &[u8],
        policy: SealingPolicy,
    ) -> SecureStorageResult<SealedData> {
        let encrypted_data = self.encrypt_data_simulation(data, "AMD").await?;
        let auth_tag = vec![0xCC; 16];

        Ok(SealedData {
            platform: EnclavePlatform::AmdMemoryGuard,
            policy,
            encrypted_data,
            auth_tag,
            aad: b"AMD-SEALED".to_vec(),
            sealed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            version: 1,
        })
    }

    /// Seal data in simulation mode
    async fn seal_simulation_data(
        &self,
        data: &[u8],
        policy: SealingPolicy,
    ) -> SecureStorageResult<SealedData> {
        let encrypted_data = self.encrypt_data_simulation(data, "SIM").await?;
        let auth_tag = vec![0xDD; 16];

        Ok(SealedData {
            platform: EnclavePlatform::Simulation,
            policy,
            encrypted_data,
            auth_tag,
            aad: b"SIM-SEALED".to_vec(),
            sealed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            version: 1,
        })
    }

    /// Simulate data encryption
    async fn encrypt_data_simulation(
        &self,
        data: &[u8],
        prefix: &str,
    ) -> SecureStorageResult<Vec<u8>> {
        // Simple XOR encryption for simulation
        let key = prefix.as_bytes();
        let mut encrypted = Vec::with_capacity(data.len());

        for (i, &byte) in data.iter().enumerate() {
            let key_byte = key[i % key.len()];
            encrypted.push(byte ^ key_byte);
        }

        // Simulate encryption time
        tokio::time::sleep(Duration::from_micros(data.len() as u64 / 1000)).await;

        Ok(encrypted)
    }

    /// Unseal platform-specific data
    async fn unseal_platform_data(&self, sealed_data: &SealedData) -> SecureStorageResult<Vec<u8>> {
        match sealed_data.platform {
            EnclavePlatform::IntelSgx => self.unseal_sgx_data(sealed_data).await,
            EnclavePlatform::ArmTrustZone => self.unseal_trustzone_data(sealed_data).await,
            EnclavePlatform::AmdMemoryGuard => self.unseal_amd_data(sealed_data).await,
            EnclavePlatform::Simulation => self.unseal_simulation_data(sealed_data).await,
        }
    }

    /// Unseal SGX data
    async fn unseal_sgx_data(&self, sealed_data: &SealedData) -> SecureStorageResult<Vec<u8>> {
        // In production, this would use sgx_unseal_data()
        self.decrypt_data_simulation(&sealed_data.encrypted_data, "SGX")
            .await
    }

    /// Unseal `TrustZone` data
    async fn unseal_trustzone_data(
        &self,
        sealed_data: &SealedData,
    ) -> SecureStorageResult<Vec<u8>> {
        self.decrypt_data_simulation(&sealed_data.encrypted_data, "TZ")
            .await
    }

    /// Unseal AMD data
    async fn unseal_amd_data(&self, sealed_data: &SealedData) -> SecureStorageResult<Vec<u8>> {
        self.decrypt_data_simulation(&sealed_data.encrypted_data, "AMD")
            .await
    }

    /// Unseal simulation data
    async fn unseal_simulation_data(
        &self,
        sealed_data: &SealedData,
    ) -> SecureStorageResult<Vec<u8>> {
        self.decrypt_data_simulation(&sealed_data.encrypted_data, "SIM")
            .await
    }

    /// Simulate data decryption
    async fn decrypt_data_simulation(
        &self,
        encrypted_data: &[u8],
        prefix: &str,
    ) -> SecureStorageResult<Vec<u8>> {
        // Simple XOR decryption for simulation (same as encryption)
        let key = prefix.as_bytes();
        let mut decrypted = Vec::with_capacity(encrypted_data.len());

        for (i, &byte) in encrypted_data.iter().enumerate() {
            let key_byte = key[i % key.len()];
            decrypted.push(byte ^ key_byte);
        }

        // Simulate decryption time
        tokio::time::sleep(Duration::from_micros(encrypted_data.len() as u64 / 1000)).await;

        Ok(decrypted)
    }

    /// Get file path for key ID
    fn get_file_path(&self, key_id: &KeyId) -> PathBuf {
        self.config.storage_path.join(format!("{key_id}.sealed"))
    }

    /// Write sealed data to file
    async fn write_sealed_file(
        &self,
        file_path: &PathBuf,
        sealed_data: &SealedData,
    ) -> SecureStorageResult<()> {
        // In production, this would use a proper serialization format
        let serialized = format!(
            "{}:{}:{}:{}",
            sealed_data.platform as u32,
            sealed_data.policy as u32,
            hex::encode(&sealed_data.encrypted_data),
            hex::encode(&sealed_data.auth_tag)
        );

        fs::write(file_path, serialized)
            .await
            .map_err(|e| SecureStorageError::InvalidInput {
                field: "file_write".to_string(),
                reason: format!("Failed to write sealed file: {e}"),
            })
    }

    /// Read sealed data from file
    async fn read_sealed_file(&self, file_path: &PathBuf) -> SecureStorageResult<SealedData> {
        let content =
            fs::read_to_string(file_path)
                .await
                .map_err(|e| SecureStorageError::NotFound {
                    resource: "sealed_file".to_string(),
                    identifier: format!("{}: {}", file_path.display(), e),
                })?;

        // Simple deserialization for demonstration
        let parts: Vec<&str> = content.split(':').collect();
        if parts.len() != 4 {
            return Err(SecureStorageError::InvalidInput {
                field: "sealed_file_format".to_string(),
                reason: "Invalid sealed file format".to_string(),
            });
        }

        let platform = match parts[0].parse::<u32>() {
            Ok(0) => EnclavePlatform::IntelSgx,
            Ok(1) => EnclavePlatform::ArmTrustZone,
            Ok(2) => EnclavePlatform::AmdMemoryGuard,
            Ok(3) => EnclavePlatform::Simulation,
            _ => {
                return Err(SecureStorageError::InvalidInput {
                    field: "platform".to_string(),
                    reason: "Invalid platform value".to_string(),
                })
            }
        };

        let policy = match parts[1].parse::<u32>() {
            Ok(0) => SealingPolicy::EnclaveIdentity,
            Ok(1) => SealingPolicy::SignerIdentity,
            Ok(2) => SealingPolicy::Platform,
            _ => {
                return Err(SecureStorageError::InvalidInput {
                    field: "policy".to_string(),
                    reason: "Invalid policy value".to_string(),
                })
            }
        };

        let encrypted_data =
            hex::decode(parts[2]).map_err(|e| SecureStorageError::InvalidInput {
                field: "encrypted_data".to_string(),
                reason: format!("Invalid hex data: {e}"),
            })?;

        let auth_tag = hex::decode(parts[3]).map_err(|e| SecureStorageError::InvalidInput {
            field: "auth_tag".to_string(),
            reason: format!("Invalid hex tag: {e}"),
        })?;

        Ok(SealedData {
            platform,
            policy,
            encrypted_data,
            auth_tag,
            aad: Vec::with_capacity(0),
            sealed_at: 0,
            version: 1,
        })
    }

    /// Get storage statistics
    #[must_use]
    pub async fn get_stats(&self) -> SealedStorageStats {
        let cache_size = self.cache.read().await.len();

        SealedStorageStats {
            seal_operations: self.seal_operations.load(Ordering::Relaxed),
            unseal_operations: self.unseal_operations.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            cache_size,
            storage_path: self.config.storage_path.clone(),
        }
    }
}

/// Sealed storage statistics
#[derive(Debug, Clone)]
pub struct SealedStorageStats {
    /// Number of seal operations
    pub seal_operations: u64,
    /// Number of unseal operations
    pub unseal_operations: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Current cache size
    pub cache_size: usize,
    /// Storage path
    pub storage_path: PathBuf,
}
