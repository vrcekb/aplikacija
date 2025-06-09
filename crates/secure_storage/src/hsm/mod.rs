//! # HSM (Hardware Security Module) Integration
//!
//! Production-grade HSM integration for `TallyIO` financial platform.
//! Provides secure key storage, cryptographic operations, and hardware-backed security.

use crate::error::SecureStorageResult;
use crate::memory::secure_buffer::SecureBuffer;
use std::time::{Duration, Instant};

// pub mod pkcs11;
// pub mod aws_cloudhsm;
// pub mod azure_hsm;
pub mod integration;
pub mod mock_hsm;

/// HSM key types supported by the secure storage system
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HsmKeyType {
    /// RSA 2048-bit key for signing/encryption
    Rsa2048,
    /// RSA 4096-bit key for enhanced security
    Rsa4096,
    /// ECDSA secp256k1 key for blockchain operations
    EcdsaSecp256k1,
    /// ECDSA secp256r1 key for NIST compliance
    EcdsaSecp256r1,
    /// AES key for symmetric encryption
    Aes256,
    /// Ed25519 for high-performance signing
    Ed25519,
}

impl std::fmt::Display for HsmKeyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rsa2048 => write!(f, "RSA-2048"),
            Self::Rsa4096 => write!(f, "RSA-4096"),
            Self::EcdsaSecp256k1 => write!(f, "ECDSA-secp256k1"),
            Self::EcdsaSecp256r1 => write!(f, "ECDSA-secp256r1"),
            Self::Aes256 => write!(f, "AES-256"),
            Self::Ed25519 => write!(f, "Ed25519"),
        }
    }
}

/// HSM credentials for authentication
#[derive(Debug, Clone)]
pub struct HsmCredentials {
    /// Username for HSM authentication
    pub username: String,
    /// Secure password buffer for HSM authentication
    pub password: SecureBuffer,
    /// HSM slot identifier for PKCS#11 operations
    pub slot_id: Option<u32>,
    /// Token label for HSM identification
    pub token_label: Option<String>,
    /// PIN for additional HSM authentication
    pub pin: Option<SecureBuffer>,
}

/// Signing algorithms supported by HSM
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SigningAlgorithm {
    /// RSA PKCS#1 v1.5 with SHA-256 for legacy compatibility
    RsaPkcs1Sha256,
    /// RSA PSS with SHA-256 for enhanced security
    RsaPssSha256,
    /// ECDSA with SHA-256 for elliptic curve signing
    EcdsaSha256,
    /// Ed25519 for high-performance digital signatures
    Ed25519,
}

/// Encryption algorithms supported by HSM
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncryptionAlgorithm {
    /// RSA OAEP encryption for secure data protection
    RsaOaep,
    /// RSA PKCS#1 v1.5 encryption for legacy compatibility
    RsaPkcs1,
    /// AES-GCM for authenticated encryption
    AesGcm,
    /// AES-CBC for block cipher encryption
    AesCbc,
}

/// HSM key metadata
#[derive(Debug, Clone)]
pub struct HsmKeyInfo {
    /// Unique identifier for the HSM key
    pub key_id: String,
    /// Type of the HSM key (RSA, ECDSA, AES, etc.)
    pub key_type: HsmKeyType,
    /// Timestamp when the key was created
    pub created_at: Instant,
    /// Number of times this key has been used
    pub usage_count: u64,
    /// Maximum allowed usage count (None for unlimited)
    pub max_usage: Option<u64>,
    /// Optional expiration time for the key
    pub expires_at: Option<Instant>,
    /// Human-readable label for the key
    pub label: String,
}

/// HSM operation result with performance metrics
#[derive(Debug)]
pub struct HsmOperationResult<T> {
    /// The actual result of the HSM operation
    pub result: T,
    /// Duration of the HSM operation for performance monitoring
    pub duration: Duration,
    /// HSM session identifier for tracking operations
    pub hsm_session_id: String,
}

/// HSM health status
#[derive(Debug, Clone)]
pub struct HsmHealth {
    /// Whether the HSM is available and responding
    pub is_available: bool,
    /// Number of active HSM sessions
    pub session_count: u32,
    /// Available memory in bytes on the HSM device
    pub free_memory: u64,
    /// HSM device temperature in Celsius (if supported)
    pub temperature: Option<f32>,
    /// Last error message from HSM operations
    pub last_error: Option<String>,
}

/// HSM performance metrics
#[derive(Debug, Clone)]
pub struct HsmMetrics {
    /// Number of HSM operations per second
    pub operations_per_second: f64,
    /// Average latency of HSM operations in milliseconds
    pub average_latency_ms: f64,
    /// Error rate as a percentage (0.0 to 1.0)
    pub error_rate: f64,
    /// Number of currently active HSM sessions
    pub active_sessions: u32,
    /// Total number of operations performed since startup
    pub total_operations: u64,
}

/// HSM provider trait for production systems
#[async_trait::async_trait]
pub trait HsmProvider: Send + Sync {
    /// Initialize HSM connection with authentication
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if initialization fails
    /// Returns `SecureStorageError::Authentication` if credentials are invalid
    async fn initialize(&mut self, credentials: &HsmCredentials) -> SecureStorageResult<()>;

    /// Generate a new key with specified parameters
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if key generation fails
    /// Returns `SecureStorageError::InvalidInput` if parameters are invalid
    async fn generate_key(
        &self,
        key_type: HsmKeyType,
        label: &str,
        extractable: bool,
    ) -> SecureStorageResult<HsmOperationResult<HsmKeyInfo>>;

    /// Import existing key material
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if import fails
    /// Returns `SecureStorageError::InvalidInput` if key material is invalid
    async fn import_key(
        &self,
        key_material: &SecureBuffer,
        key_type: HsmKeyType,
        label: &str,
    ) -> SecureStorageResult<HsmOperationResult<HsmKeyInfo>>;

    /// Sign data with specified key (< 50ms for production)
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if signing fails
    /// Returns `SecureStorageError::NotFound` if key doesn't exist
    async fn sign(
        &self,
        key_id: &str,
        data: &[u8],
        algorithm: SigningAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<Vec<u8>>>;

    /// Verify signature
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if verification fails
    /// Returns `SecureStorageError::NotFound` if key doesn't exist
    async fn verify(
        &self,
        key_id: &str,
        data: &[u8],
        signature: &[u8],
        algorithm: SigningAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<bool>>;

    /// Encrypt data (< 10ms for 1KB)
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if encryption fails
    /// Returns `SecureStorageError::NotFound` if key doesn't exist
    async fn encrypt(
        &self,
        key_id: &str,
        data: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<Vec<u8>>>;

    /// Decrypt data (< 10ms for 1KB)
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if decryption fails
    /// Returns `SecureStorageError::NotFound` if key doesn't exist
    async fn decrypt(
        &self,
        key_id: &str,
        encrypted_data: &[u8],
        algorithm: EncryptionAlgorithm,
    ) -> SecureStorageResult<HsmOperationResult<Vec<u8>>>;

    /// Delete key from HSM
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if deletion fails
    /// Returns `SecureStorageError::NotFound` if key doesn't exist
    async fn delete_key(&self, key_id: &str) -> SecureStorageResult<HsmOperationResult<()>>;

    /// List all keys
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if listing fails
    async fn list_keys(&self) -> SecureStorageResult<HsmOperationResult<Vec<HsmKeyInfo>>>;

    /// Get HSM health status
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if health check fails
    async fn health_check(&self) -> SecureStorageResult<HsmHealth>;

    /// Get HSM performance metrics
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::HsmError` if metrics retrieval fails
    async fn get_metrics(&self) -> SecureStorageResult<HsmMetrics>;
}
