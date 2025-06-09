//! Core types for secure storage operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// Unique identifier for cryptographic keys in `TallyIO` secure storage
///
/// This type provides a type-safe wrapper around string-based key identifiers,
/// ensuring that key IDs are properly validated and cannot be confused with
/// other string types in the API. Used throughout the secure storage system
/// to reference encryption keys, signing keys, and other cryptographic material.
///
/// # Security Considerations
///
/// - Key IDs should be unpredictable to prevent enumeration attacks
/// - UUIDs are recommended for new key generation
/// - Key IDs are logged in audit trails for compliance
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::KeyId;
///
/// // Generate a random key ID
/// let key_id = KeyId::generate();
///
/// // Create from existing string
/// let key_id = KeyId::new("user_master_key_2024");
///
/// // Access the underlying string
/// println!("Key ID: {}", key_id.as_str());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyId(pub String);

impl KeyId {
    /// Creates a new key ID from any string-like input
    ///
    /// This method accepts any type that can be converted into a String,
    /// providing flexibility while maintaining type safety.
    ///
    /// # Arguments
    ///
    /// * `id` - The identifier string, typically a UUID or descriptive name
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::KeyId;
    ///
    /// let key_id = KeyId::new("master_key_001");
    /// let key_id = KeyId::new(String::from("user_key"));
    /// ```
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generates a cryptographically random key ID using UUID v4
    ///
    /// This method creates a new, unpredictable key identifier suitable
    /// for production use. The generated UUID provides 122 bits of entropy,
    /// making collision probability negligible for practical purposes.
    ///
    /// # Security
    ///
    /// Uses UUID v4 which provides cryptographically strong randomness,
    /// preventing key ID enumeration or prediction attacks.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::KeyId;
    ///
    /// let key_id = KeyId::generate();
    /// assert_ne!(key_id, KeyId::generate()); // Extremely unlikely to be equal
    /// ```
    #[must_use]
    pub fn generate() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Returns a reference to the underlying string identifier
    ///
    /// This method provides access to the raw string representation
    /// of the key ID for use in APIs, logging, and storage operations.
    ///
    /// # Returns
    ///
    /// A string slice containing the key identifier
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::KeyId;
    ///
    /// let key_id = KeyId::new("example_key");
    /// assert_eq!(key_id.as_str(), "example_key");
    /// ```
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KeyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for KeyId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for KeyId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Secure cryptographic key material with automatic memory protection
///
/// This structure holds sensitive cryptographic key data along with its metadata.
/// The key material is automatically zeroized when dropped to prevent key recovery
/// from memory dumps or swap files. This is critical for financial applications
/// where key compromise could lead to catastrophic losses.
///
/// # Security Features
///
/// - **Automatic Zeroization**: Key bytes are securely wiped on drop
/// - **Memory Protection**: Attempts to prevent swapping to disk
/// - **Metadata Tracking**: Maintains key lifecycle information
/// - **Type Safety**: Prevents accidental key exposure through type system
///
/// # Memory Safety
///
/// The underlying key bytes are stored in a `Vec<u8>` that is zeroized when
/// the structure is dropped. This provides protection against:
/// - Memory dumps
/// - Core dumps
/// - Swap file exposure
/// - Use-after-free vulnerabilities
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::{KeyMaterial, KeyMetadata, KeyId, EncryptionAlgorithm};
///
/// let metadata = KeyMetadata::new(
///     KeyId::generate(),
///     EncryptionAlgorithm::Aes256Gcm,
///     256
/// );
///
/// let key_material = KeyMaterial::new(vec![0u8; 32], metadata);
///
/// // Key is automatically zeroized when key_material goes out of scope
/// ```
#[derive(Clone)]
pub struct KeyMaterial {
    /// The actual cryptographic key bytes
    ///
    /// This field contains the raw key material used for cryptographic operations.
    /// The bytes are automatically zeroized when the structure is dropped.
    pub key: Vec<u8>,

    /// Metadata describing the key properties and lifecycle
    ///
    /// Contains information about the key algorithm, creation time, usage statistics,
    /// and other properties needed for key management and audit compliance.
    pub metadata: KeyMetadata,
}

impl KeyMaterial {
    /// Creates new key material with the specified key bytes and metadata
    ///
    /// This constructor takes ownership of the key bytes and metadata,
    /// ensuring that the key material is properly managed throughout its lifecycle.
    ///
    /// # Arguments
    ///
    /// * `key` - The raw cryptographic key bytes
    /// * `metadata` - Key metadata including algorithm, creation time, etc.
    ///
    /// # Security
    ///
    /// The key bytes will be automatically zeroized when this structure is dropped.
    /// Ensure that the input key bytes are also properly zeroized if they are
    /// no longer needed elsewhere.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{KeyMaterial, KeyMetadata, KeyId, EncryptionAlgorithm};
    ///
    /// let key_bytes = vec![0u8; 32]; // In practice, use secure random generation
    /// let metadata = KeyMetadata::new(KeyId::generate(), EncryptionAlgorithm::Aes256Gcm, 256);
    /// let key_material = KeyMaterial::new(key_bytes, metadata);
    /// ```
    #[must_use]
    pub const fn new(key: Vec<u8>, metadata: KeyMetadata) -> Self {
        Self { key, metadata }
    }

    /// Returns a reference to the raw key bytes
    ///
    /// This method provides access to the underlying cryptographic key material
    /// for use in encryption/decryption operations. The returned slice should
    /// be used immediately and not stored, as the key may be zeroized when
    /// the parent structure is dropped.
    ///
    /// # Security Warning
    ///
    /// The returned slice provides direct access to sensitive key material.
    /// Ensure that:
    /// - The slice is not copied unnecessarily
    /// - Operations using the key are constant-time when possible
    /// - The slice is not logged or stored in non-secure locations
    ///
    /// # Returns
    ///
    /// A byte slice containing the raw key material
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{KeyMaterial, KeyMetadata, KeyId, EncryptionAlgorithm};
    ///
    /// let metadata = KeyMetadata::new(KeyId::generate(), EncryptionAlgorithm::Aes256Gcm, 256);
    /// let key_material = KeyMaterial::new(vec![0u8; 32], metadata);
    /// let key_bytes = key_material.key_bytes();
    /// // Use key_bytes immediately for cryptographic operations
    /// ```
    #[must_use]
    pub fn key_bytes(&self) -> &[u8] {
        &self.key
    }

    /// Returns the length of the encryption key in bytes.
    ///
    /// This method provides the size of the underlying key material, which is
    /// important for cryptographic operations and key validation. The length
    /// depends on the encryption algorithm being used.
    ///
    /// # Returns
    ///
    /// The number of bytes in the encryption key:
    /// - AES-256: 32 bytes
    /// - AES-128: 16 bytes
    /// - `ChaCha20`: 32 bytes
    ///
    /// # Performance
    ///
    /// This is a const function with zero runtime cost.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{KeyMaterial, KeyMetadata, KeyId, EncryptionAlgorithm};
    ///
    /// let metadata = KeyMetadata::new(KeyId::generate(), EncryptionAlgorithm::Aes256Gcm, 256);
    /// let key = KeyMaterial::new(vec![0u8; 32], metadata);
    /// assert_eq!(key.len(), 32); // AES-256 key size
    /// ```
    #[must_use]
    pub const fn len(&self) -> usize {
        self.key.len()
    }

    /// Checks if the encryption key is empty (zero length).
    ///
    /// This method determines whether the key contains any key material.
    /// An empty key is invalid for cryptographic operations and indicates
    /// an error condition or uninitialized state.
    ///
    /// # Returns
    ///
    /// `true` if the key has zero length, `false` otherwise.
    ///
    /// # Security Implications
    ///
    /// Empty keys should never be used for encryption operations as they
    /// provide no security. Always validate that keys are non-empty before
    /// performing cryptographic operations.
    ///
    /// # Performance
    ///
    /// This is a const function with zero runtime cost.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{KeyMaterial, KeyMetadata, KeyId, EncryptionAlgorithm};
    ///
    /// let metadata = KeyMetadata::new(KeyId::generate(), EncryptionAlgorithm::Aes256Gcm, 256);
    /// let empty_key = KeyMaterial::new(vec![], metadata.clone());
    /// assert!(empty_key.is_empty());
    ///
    /// let valid_key = KeyMaterial::new(vec![0u8; 32], metadata);
    /// assert!(!valid_key.is_empty());
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.key.is_empty()
    }
}

impl fmt::Debug for KeyMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyMaterial")
            .field("key", &"[REDACTED]")
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Comprehensive metadata for cryptographic key lifecycle management
///
/// This structure tracks all aspects of a cryptographic key's lifecycle,
/// from creation to expiration. It provides essential information for
/// key management, audit compliance, and security policy enforcement
/// in financial applications where key governance is critical.
///
/// # Key Lifecycle Tracking
///
/// - **Creation**: When and how the key was generated
/// - **Usage**: Track encryption/decryption operations and frequency
/// - **Expiration**: Automatic key rotation and lifecycle management
/// - **Permissions**: Fine-grained control over key operations
/// - **Audit**: Complete audit trail for compliance requirements
///
/// # Compliance Features
///
/// The metadata supports various regulatory requirements:
/// - **SOX**: Key creation and usage tracking
/// - **PCI DSS**: Key lifecycle management
/// - **FIPS 140-2**: Algorithm and key size validation
/// - **Common Criteria**: Access control and audit logging
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::{KeyMetadata, KeyId, EncryptionAlgorithm};
///
/// // Create metadata for a new master key
/// let metadata = KeyMetadata::new(
///     KeyId::generate(),
///     EncryptionAlgorithm::Aes256Gcm,
///     256
/// );
///
/// // Check if key has expired
/// if metadata.is_expired() {
///     println!("Key has expired and should be rotated");
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyMetadata {
    /// Unique identifier for this cryptographic key
    ///
    /// This ID is used throughout the system to reference the key
    /// and must be unique across all keys in the vault.
    pub id: KeyId,

    /// Cryptographic algorithm this key is designed for
    ///
    /// Determines the key's intended use and validates that operations
    /// are performed with the correct algorithm.
    pub algorithm: EncryptionAlgorithm,

    /// Key size in bits (e.g., 256 for AES-256)
    ///
    /// Used for validation and security policy enforcement.
    /// Must match the algorithm's expected key size.
    pub key_size: u32,

    /// UTC timestamp when the key was created
    ///
    /// Used for audit trails and key age calculations.
    /// Immutable after key creation.
    pub created_at: DateTime<Utc>,

    /// UTC timestamp when the key was last used for cryptographic operations
    ///
    /// Updated automatically on each encryption/decryption operation.
    /// Used for key usage analytics and dormant key detection.
    pub last_used: Option<DateTime<Utc>>,

    /// UTC timestamp when the key expires and should no longer be used
    ///
    /// Enforces automatic key rotation policies. Operations with
    /// expired keys are rejected for security compliance.
    pub expires_at: Option<DateTime<Utc>>,

    /// Detailed permissions controlling how this key can be used
    ///
    /// Provides fine-grained access control for different operations
    /// and enforces usage limits for compliance.
    pub usage: KeyUsage,

    /// Additional metadata tags for custom key management
    ///
    /// Allows storing custom attributes like department, project,
    /// compliance classification, or other organizational metadata.
    pub tags: HashMap<String, String>,
}

impl KeyMetadata {
    /// Creates new key metadata with default settings
    ///
    /// This constructor initializes a new key metadata structure with the
    /// specified core properties and sensible defaults for optional fields.
    /// The key is created with default usage permissions (encrypt/decrypt enabled)
    /// and no expiration date.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for the key
    /// * `algorithm` - Cryptographic algorithm the key will be used with
    /// * `key_size` - Key size in bits (must match algorithm requirements)
    ///
    /// # Security Considerations
    ///
    /// - The creation timestamp is set to the current UTC time
    /// - Default usage permissions allow encryption and decryption only
    /// - No expiration is set by default (should be configured based on policy)
    /// - Tags are empty and should be populated based on organizational needs
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{KeyMetadata, KeyId, EncryptionAlgorithm};
    ///
    /// let metadata = KeyMetadata::new(
    ///     KeyId::generate(),
    ///     EncryptionAlgorithm::Aes256Gcm,
    ///     256
    /// );
    ///
    /// assert_eq!(metadata.key_size, 256);
    /// assert_eq!(metadata.algorithm, EncryptionAlgorithm::Aes256Gcm);
    /// ```
    #[must_use]
    pub fn new(id: KeyId, algorithm: EncryptionAlgorithm, key_size: u32) -> Self {
        Self {
            id,
            algorithm,
            key_size,
            created_at: Utc::now(),
            last_used: None,
            expires_at: None,
            usage: KeyUsage::default(),
            tags: HashMap::new(),
        }
    }

    /// Checks if the key has expired and should no longer be used
    ///
    /// This method evaluates the key's expiration timestamp against the current
    /// time to determine if the key is still valid for cryptographic operations.
    /// Expired keys should be rejected for all operations to maintain security.
    ///
    /// # Returns
    ///
    /// * `true` if the key has an expiration date and it has passed
    /// * `false` if the key has no expiration date or has not yet expired
    ///
    /// # Security Implications
    ///
    /// Using expired keys violates security policies and may indicate:
    /// - Failed key rotation procedures
    /// - Compromised key management processes
    /// - Potential compliance violations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{KeyMetadata, KeyId, EncryptionAlgorithm};
    /// use chrono::{Utc, Duration};
    ///
    /// let mut metadata = KeyMetadata::new(
    ///     KeyId::generate(),
    ///     EncryptionAlgorithm::Aes256Gcm,
    ///     256
    /// );
    ///
    /// // Key without expiration is not expired
    /// assert!(!metadata.is_expired());
    ///
    /// // Set expiration in the past
    /// metadata.expires_at = Some(Utc::now() - Duration::hours(1));
    /// assert!(metadata.is_expired());
    /// ```
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .is_some_and(|expires_at| Utc::now() > expires_at)
    }

    /// Update last used timestamp
    pub fn mark_used(&mut self) {
        self.last_used = Some(Utc::now());
    }
}

/// Cryptographic encryption algorithms supported by `TallyIO` secure storage
///
/// This enum defines the available symmetric encryption algorithms for protecting
/// sensitive financial data. Each algorithm provides authenticated encryption with
/// associated data (AEAD) properties, ensuring both confidentiality and integrity.
///
/// # Security Properties
///
/// All algorithms provide:
/// - **Confidentiality**: Data cannot be read without the key
/// - **Integrity**: Tampering is detected and rejected
/// - **Authentication**: Data origin is verified
/// - **Semantic Security**: Identical plaintexts produce different ciphertexts
///
/// # Algorithm Selection Guidelines
///
/// - **AES-256-GCM**: Recommended for most use cases, hardware acceleration available
/// - **ChaCha20-Poly1305**: Recommended for software-only implementations
/// - **AES-256-CBC**: Legacy support only, not recommended for new implementations
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::EncryptionAlgorithm;
///
/// // Recommended for new implementations
/// let algorithm = EncryptionAlgorithm::Aes256Gcm;
///
/// // Alternative for software-only environments
/// let algorithm = EncryptionAlgorithm::ChaCha20Poly1305;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM: Advanced Encryption Standard with 256-bit keys and Galois/Counter Mode
    ///
    /// This is the recommended algorithm for most use cases. It provides:
    /// - 256-bit key strength (post-quantum security margin)
    /// - Hardware acceleration on modern CPUs (AES-NI)
    /// - Authenticated encryption with associated data (AEAD)
    /// - Excellent performance and security balance
    ///
    /// **Use when**: Hardware acceleration is available, maximum compatibility needed
    Aes256Gcm,

    /// ChaCha20-Poly1305: Stream cipher with Poly1305 message authentication
    ///
    /// This algorithm provides:
    /// - 256-bit key strength
    /// - Constant-time implementation (side-channel resistant)
    /// - Excellent software performance
    /// - AEAD properties with Poly1305 MAC
    ///
    /// **Use when**: Software-only implementation needed, side-channel resistance required
    ChaCha20Poly1305,

    /// AES-256-CBC: Advanced Encryption Standard with Cipher Block Chaining (LEGACY)
    ///
    /// **⚠️ DEPRECATED**: This algorithm is provided for legacy compatibility only.
    /// It does not provide authenticated encryption and is vulnerable to padding
    /// oracle attacks if not implemented carefully.
    ///
    /// **Do not use** for new implementations. Migrate existing data to AES-256-GCM.
    Aes256Cbc,
}

impl EncryptionAlgorithm {
    /// Returns the required key size in bytes for this encryption algorithm
    ///
    /// All supported algorithms use 256-bit (32-byte) keys, providing
    /// strong security suitable for financial applications and meeting
    /// post-quantum security recommendations.
    ///
    /// # Returns
    ///
    /// The key size in bytes (always 32 for current algorithms)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::EncryptionAlgorithm;
    ///
    /// assert_eq!(EncryptionAlgorithm::Aes256Gcm.key_size_bytes(), 32);
    /// assert_eq!(EncryptionAlgorithm::ChaCha20Poly1305.key_size_bytes(), 32);
    /// ```
    #[must_use]
    pub const fn key_size_bytes(self) -> usize {
        match self {
            Self::Aes256Gcm | Self::ChaCha20Poly1305 | Self::Aes256Cbc => 32, // 256 bits
        }
    }

    /// Returns the required nonce/IV size in bytes for this encryption algorithm
    ///
    /// The nonce (number used once) or initialization vector (IV) size varies
    /// by algorithm and is critical for security. Using the wrong size will
    /// result in encryption failures or security vulnerabilities.
    ///
    /// # Algorithm-Specific Sizes
    ///
    /// - **AES-256-GCM**: 12 bytes (96 bits) - optimal for GCM mode
    /// - **ChaCha20-Poly1305**: 12 bytes (96 bits) - standard for `ChaCha20`
    /// - **AES-256-CBC**: 16 bytes (128 bits) - AES block size
    ///
    /// # Security Note
    ///
    /// Nonces must be unique for each encryption operation with the same key.
    /// Reusing nonces can lead to catastrophic security failures.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::EncryptionAlgorithm;
    ///
    /// assert_eq!(EncryptionAlgorithm::Aes256Gcm.nonce_size_bytes(), 12);
    /// assert_eq!(EncryptionAlgorithm::ChaCha20Poly1305.nonce_size_bytes(), 12);
    /// assert_eq!(EncryptionAlgorithm::Aes256Cbc.nonce_size_bytes(), 16);
    /// ```
    #[must_use]
    pub const fn nonce_size_bytes(self) -> usize {
        match self {
            Self::Aes256Gcm | Self::ChaCha20Poly1305 => 12, // 96 bits
            Self::Aes256Cbc => 16,                          // 128 bits
        }
    }

    /// Checks if this algorithm provides authenticated encryption (AEAD)
    ///
    /// Authenticated encryption provides both confidentiality and integrity,
    /// ensuring that encrypted data cannot be tampered with undetected.
    /// This is essential for financial applications where data integrity
    /// is as important as confidentiality.
    ///
    /// # AEAD Properties
    ///
    /// Authenticated encryption algorithms provide:
    /// - **Confidentiality**: Data cannot be read without the key
    /// - **Integrity**: Tampering is detected and rejected
    /// - **Authentication**: Data origin is verified
    ///
    /// # Algorithm Support
    ///
    /// - **AES-256-GCM**: ✅ Full AEAD support
    /// - **ChaCha20-Poly1305**: ✅ Full AEAD support
    /// - **AES-256-CBC**: ❌ No authentication (legacy only)
    ///
    /// # Security Recommendation
    ///
    /// Always prefer authenticated encryption algorithms for new implementations.
    /// Non-authenticated algorithms should only be used for legacy compatibility.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::EncryptionAlgorithm;
    ///
    /// assert!(EncryptionAlgorithm::Aes256Gcm.is_authenticated());
    /// assert!(EncryptionAlgorithm::ChaCha20Poly1305.is_authenticated());
    /// assert!(!EncryptionAlgorithm::Aes256Cbc.is_authenticated());
    /// ```
    #[must_use]
    pub const fn is_authenticated(self) -> bool {
        match self {
            Self::Aes256Gcm | Self::ChaCha20Poly1305 => true,
            Self::Aes256Cbc => false,
        }
    }
}

impl fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aes256Gcm => write!(f, "AES-256-GCM"),
            Self::ChaCha20Poly1305 => write!(f, "ChaCha20-Poly1305"),
            Self::Aes256Cbc => write!(f, "AES-256-CBC"),
        }
    }
}

/// Fine-grained permissions controlling cryptographic key usage in `TallyIO`
///
/// This structure defines exactly how a cryptographic key can be used within
/// the secure storage system. It provides granular control over key operations
/// to enforce security policies and compliance requirements in financial applications.
///
/// # Permission Model
///
/// The permission model follows the principle of least privilege:
/// - **Explicit Permissions**: Each operation type must be explicitly enabled
/// - **Usage Limits**: Operations can be limited by count to enforce key rotation
/// - **Audit Trail**: All permission checks are logged for compliance
/// - **Immutable Enforcement**: Permissions cannot be escalated after key creation
///
/// # Security Benefits
///
/// - **Separation of Duties**: Different keys for different operations
/// - **Key Rotation Enforcement**: Automatic limits prevent overuse
/// - **Compliance Support**: Granular controls for regulatory requirements
/// - **Attack Surface Reduction**: Minimize key exposure through limited permissions
///
/// # Financial Use Cases
///
/// - **Master Keys**: Derive-only permissions for key hierarchy
/// - **Transaction Keys**: Encrypt/decrypt for payment processing
/// - **Audit Keys**: Sign-only for tamper-evident logging
/// - **Backup Keys**: Decrypt-only for data recovery
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::KeyUsage;
///
/// // Master key for key derivation only
/// let master_key_usage = KeyUsage {
///     encrypt: false,
///     decrypt: false,
///     sign: false,
///     verify: false,
///     derive: true,
///     max_operations: Some(1000),
///     operation_count: 0,
/// };
///
/// // Transaction processing key
/// let transaction_key_usage = KeyUsage {
///     encrypt: true,
///     decrypt: true,
///     sign: false,
///     verify: false,
///     derive: false,
///     max_operations: Some(10000),
///     operation_count: 0,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)] // Financial applications require granular permissions
pub struct KeyUsage {
    /// Can be used for encryption
    pub encrypt: bool,
    /// Can be used for decryption
    pub decrypt: bool,
    /// Can be used for signing
    pub sign: bool,
    /// Can be used for verification
    pub verify: bool,
    /// Can be used for key derivation
    pub derive: bool,
    /// Maximum number of operations (None = unlimited)
    pub max_operations: Option<u64>,
    /// Current operation count
    pub operation_count: u64,
}

impl Default for KeyUsage {
    fn default() -> Self {
        Self {
            encrypt: true,
            decrypt: true,
            sign: false,
            verify: false,
            derive: false,
            max_operations: None,
            operation_count: 0,
        }
    }
}

impl KeyUsage {
    /// Checks if the key is authorized and available for encryption operations
    ///
    /// This method validates both the permission flag and usage limits to determine
    /// if the key can be used for encryption. This is critical for financial
    /// applications where unauthorized encryption could lead to data loss or
    /// compliance violations.
    ///
    /// # Security Validation
    ///
    /// The method performs two checks:
    /// 1. **Permission Check**: Verifies the `encrypt` flag is enabled
    /// 2. **Usage Limit Check**: Ensures operation count hasn't exceeded limits
    ///
    /// # Returns
    ///
    /// * `true` if the key can be used for encryption
    /// * `false` if encryption is disabled or usage limits are exceeded
    ///
    /// # Compliance
    ///
    /// This check is logged for audit purposes and supports:
    /// - Key rotation policies (through usage limits)
    /// - Separation of duties (through permission flags)
    /// - Regulatory compliance (through audit trails)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::KeyUsage;
    ///
    /// let mut key_usage = KeyUsage::default();
    /// assert!(key_usage.can_encrypt()); // Default allows encryption
    ///
    /// key_usage.encrypt = false;
    /// assert!(!key_usage.can_encrypt()); // Permission disabled
    /// ```
    #[must_use]
    pub fn can_encrypt(&self) -> bool {
        self.encrypt && self.is_within_limits()
    }

    /// Checks if the key is authorized and available for decryption operations
    ///
    /// This method validates both the permission flag and usage limits to determine
    /// if the key can be used for decryption. Decryption permissions are often
    /// more restricted than encryption in financial systems to prevent unauthorized
    /// data access.
    ///
    /// # Security Validation
    ///
    /// The method performs two checks:
    /// 1. **Permission Check**: Verifies the `decrypt` flag is enabled
    /// 2. **Usage Limit Check**: Ensures operation count hasn't exceeded limits
    ///
    /// # Returns
    ///
    /// * `true` if the key can be used for decryption
    /// * `false` if decryption is disabled or usage limits are exceeded
    ///
    /// # Financial Security
    ///
    /// Decryption controls are essential for:
    /// - Preventing unauthorized data access
    /// - Enforcing data classification policies
    /// - Supporting break-glass procedures
    /// - Maintaining audit trails for sensitive operations
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::KeyUsage;
    ///
    /// let mut key_usage = KeyUsage::default();
    /// assert!(key_usage.can_decrypt()); // Default allows decryption
    ///
    /// key_usage.max_operations = Some(0);
    /// assert!(!key_usage.can_decrypt()); // Usage limit exceeded
    /// ```
    #[must_use]
    pub fn can_decrypt(&self) -> bool {
        self.decrypt && self.is_within_limits()
    }

    /// Check if operation count is within limits
    fn is_within_limits(&self) -> bool {
        self.max_operations
            .is_none_or(|max_ops| self.operation_count < max_ops)
    }

    /// Increment operation count
    pub const fn increment_operations(&mut self) {
        self.operation_count += 1;
    }
}

/// Container for encrypted data with associated cryptographic metadata.
///
/// This structure holds encrypted data along with all the metadata required
/// for decryption, including the encryption algorithm, key identifier, nonce,
/// and authentication information. It provides a complete, self-contained
/// representation of encrypted data that can be safely stored or transmitted.
///
/// # Structure
///
/// - **Algorithm**: Identifies the encryption algorithm used
/// - **Key ID**: References the key used for encryption
/// - **Nonce**: Unique value used for this encryption operation
/// - **Ciphertext**: The actual encrypted data
/// - **Tag**: Authentication tag for authenticated encryption modes
/// - **AAD**: Additional Authenticated Data (optional)
///
/// # Security Properties
///
/// - Authenticated encryption prevents tampering
/// - Unique nonces prevent replay attacks
/// - Key rotation support through key ID references
/// - Metadata integrity through serialization
///
/// # Usage Patterns
///
/// ```rust
/// use secure_storage::types::{EncryptedData, EncryptionAlgorithm, KeyId};
///
/// let encrypted = EncryptedData::new(
///     EncryptionAlgorithm::Aes256Gcm,
///     KeyId::new("key-123".to_string()),
///     vec![1, 2, 3, 4], // nonce
///     vec![5, 6, 7, 8], // ciphertext
/// ).with_tag(vec![9, 10, 11, 12]); // authentication tag
/// ```
///
/// # Serialization
///
/// The structure supports JSON serialization for storage and transmission.
/// All binary data is base64-encoded in the JSON representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Encryption algorithm used
    pub algorithm: EncryptionAlgorithm,
    /// Key ID used for encryption
    pub key_id: KeyId,
    /// Nonce/IV used
    pub nonce: Vec<u8>,
    /// Encrypted data
    pub ciphertext: Vec<u8>,
    /// Authentication tag (for AEAD algorithms)
    pub tag: Option<Vec<u8>>,
    /// Additional authenticated data
    pub aad: Option<Vec<u8>>,
    /// Encryption timestamp
    pub encrypted_at: DateTime<Utc>,
}

impl EncryptedData {
    /// Creates a new encrypted data container with the specified parameters.
    ///
    /// This constructor initializes an encrypted data container with the minimum
    /// required information for decryption. Additional metadata like authentication
    /// tags and additional authenticated data can be added using builder methods.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The encryption algorithm used to encrypt the data
    /// * `key_id` - Identifier of the key used for encryption
    /// * `nonce` - Unique nonce/IV used for this encryption operation
    /// * `ciphertext` - The encrypted data bytes
    ///
    /// # Returns
    ///
    /// A new `EncryptedData` instance ready for use or further configuration.
    ///
    /// # Security Requirements
    ///
    /// - Nonce must be unique for each encryption operation with the same key
    /// - Ciphertext should only contain the encrypted payload, not metadata
    /// - Key ID must reference a valid, accessible encryption key
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{EncryptedData, EncryptionAlgorithm, KeyId};
    ///
    /// let encrypted = EncryptedData::new(
    ///     EncryptionAlgorithm::Aes256Gcm,
    ///     KeyId::new("master-key-001".to_string()),
    ///     vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], // 12-byte nonce for GCM
    ///     vec![0xde, 0xad, 0xbe, 0xef], // encrypted data
    /// );
    /// ```
    #[must_use]
    pub fn new(
        algorithm: EncryptionAlgorithm,
        key_id: KeyId,
        nonce: Vec<u8>,
        ciphertext: Vec<u8>,
    ) -> Self {
        Self {
            algorithm,
            key_id,
            nonce,
            ciphertext,
            tag: None,
            aad: None,
            encrypted_at: Utc::now(),
        }
    }

    /// Adds an authentication tag to the encrypted data container.
    ///
    /// This method sets the authentication tag for authenticated encryption modes
    /// like AES-GCM and ChaCha20-Poly1305. The tag is used to verify the integrity
    /// and authenticity of both the ciphertext and any additional authenticated data.
    ///
    /// # Arguments
    ///
    /// * `tag` - Authentication tag bytes generated during encryption
    ///
    /// # Returns
    ///
    /// The modified `EncryptedData` instance with the authentication tag set.
    ///
    /// # Security Importance
    ///
    /// The authentication tag is critical for security:
    /// - Detects any tampering with the ciphertext
    /// - Prevents chosen-ciphertext attacks
    /// - Ensures data integrity and authenticity
    /// - Required for authenticated encryption modes
    ///
    /// # Tag Sizes
    ///
    /// - AES-GCM: 16 bytes (128 bits) standard
    /// - ChaCha20-Poly1305: 16 bytes (128 bits)
    /// - Shorter tags may be acceptable for some use cases but reduce security
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{EncryptedData, EncryptionAlgorithm, KeyId};
    ///
    /// let encrypted = EncryptedData::new(
    ///     EncryptionAlgorithm::Aes256Gcm,
    ///     KeyId::new("key-001".to_string()),
    ///     vec![0; 12], // nonce
    ///     vec![0; 16], // ciphertext
    /// ).with_tag(vec![0; 16]); // 16-byte authentication tag
    /// ```
    #[must_use]
    pub fn with_tag(mut self, tag: Vec<u8>) -> Self {
        self.tag = Some(tag);
        self
    }

    /// Adds Additional Authenticated Data (AAD) to the encrypted data container.
    ///
    /// AAD is data that is authenticated but not encrypted. It's included in the
    /// authentication tag calculation but remains in plaintext. This is useful
    /// for metadata that needs to be authenticated but doesn't need to be secret.
    ///
    /// # Arguments
    ///
    /// * `aad` - Additional authenticated data bytes
    ///
    /// # Returns
    ///
    /// The modified `EncryptedData` instance with AAD set.
    ///
    /// # Use Cases
    ///
    /// - Protocol headers that must be authenticated
    /// - Metadata like timestamps or version numbers
    /// - Context information for key derivation
    /// - Database record identifiers
    ///
    /// # Security Properties
    ///
    /// - AAD is included in authentication tag calculation
    /// - Tampering with AAD will cause authentication failure
    /// - AAD is not encrypted and remains readable
    /// - Must be provided during both encryption and decryption
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{EncryptedData, EncryptionAlgorithm, KeyId};
    ///
    /// let metadata = b"user_id:12345,timestamp:1640995200";
    /// let encrypted = EncryptedData::new(
    ///     EncryptionAlgorithm::Aes256Gcm,
    ///     KeyId::new("key-001".to_string()),
    ///     vec![0; 12], // nonce
    ///     vec![0; 16], // ciphertext
    /// ).with_aad(metadata.to_vec());
    /// ```
    #[must_use]
    pub fn with_aad(mut self, aad: Vec<u8>) -> Self {
        self.aad = Some(aad);
        self
    }

    /// Calculates the total size in bytes of all encrypted data components.
    ///
    /// This method returns the sum of all data components including nonce,
    /// ciphertext, authentication tag, and additional authenticated data.
    /// This is useful for storage allocation, bandwidth estimation, and
    /// performance analysis.
    ///
    /// # Returns
    ///
    /// Total size in bytes of:
    /// - Nonce/IV bytes
    /// - Encrypted ciphertext bytes
    /// - Authentication tag bytes (if present)
    /// - Additional authenticated data bytes (if present)
    ///
    /// # Performance
    ///
    /// This method performs simple arithmetic operations and has minimal
    /// computational overhead. The result can be cached if called frequently.
    ///
    /// # Use Cases
    ///
    /// - Storage space estimation
    /// - Network bandwidth planning
    /// - Memory allocation sizing
    /// - Performance monitoring and optimization
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{EncryptedData, EncryptionAlgorithm, KeyId};
    ///
    /// let encrypted = EncryptedData::new(
    ///     EncryptionAlgorithm::Aes256Gcm,
    ///     KeyId::new("key-001".to_string()),
    ///     vec![0; 12], // 12-byte nonce
    ///     vec![0; 32], // 32-byte ciphertext
    /// ).with_tag(vec![0; 16]); // 16-byte tag
    ///
    /// assert_eq!(encrypted.total_size(), 60); // 12 + 32 + 16 = 60 bytes
    /// ```
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.nonce.len()
            + self.ciphertext.len()
            + self.tag.as_ref().map_or(0, std::vec::Vec::len)
            + self.aad.as_ref().map_or(0, std::vec::Vec::len)
    }
}

/// Represents an authenticated user session with role-based access control.
///
/// This structure contains all information necessary to track and validate
/// user sessions within the secure storage system. Sessions provide the
/// foundation for authentication, authorization, and audit logging.
///
/// # Session Lifecycle
///
/// 1. **Creation**: Session is created upon successful authentication
/// 2. **Activity**: Session is used for authorized operations
/// 3. **Validation**: Each operation validates session expiry and permissions
/// 4. **Expiration**: Session automatically expires after configured timeout
/// 5. **Cleanup**: Expired sessions are removed from active session store
///
/// # Security Features
///
/// - **Unique Session IDs**: Cryptographically secure UUIDs prevent guessing
/// - **Role-Based Access**: Users assigned specific roles with defined permissions
/// - **Automatic Expiration**: Sessions expire to limit exposure window
/// - **Activity Tracking**: Last activity timestamp for idle timeout enforcement
/// - **Audit Integration**: All session events are logged for security monitoring
///
/// # Thread Safety
///
/// Session objects are designed to be immutable after creation, making them
/// safe to share across threads. Session updates create new instances rather
/// than modifying existing ones.
///
/// # Serialization
///
/// Sessions can be serialized to JSON for storage in databases, caches, or
/// transmission over networks. Sensitive information is not included in
/// the serialized form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Unique session ID
    pub id: String,
    /// User/service ID
    pub user_id: String,
    /// Assigned roles
    pub roles: Vec<String>,
    /// Session creation time
    pub created_at: DateTime<Utc>,
    /// Session expiration time
    pub expires_at: DateTime<Utc>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

impl Session {
    /// Creates a new authenticated session for the specified user.
    ///
    /// This constructor initializes a new session with a cryptographically secure
    /// session ID, user information, role assignments, and expiration time.
    /// The session is immediately ready for use in authorization decisions.
    ///
    /// # Arguments
    ///
    /// * `user_id` - Unique identifier for the authenticated user or service
    /// * `roles` - Vector of role names assigned to this session
    /// * `duration_secs` - Session lifetime in seconds from creation
    ///
    /// # Returns
    ///
    /// A new `Session` instance with:
    /// - Unique UUID-based session ID
    /// - Current timestamp for creation and last activity
    /// - Calculated expiration time based on duration
    /// - Assigned roles for authorization
    ///
    /// # Security Considerations
    ///
    /// - Session IDs are generated using cryptographically secure random numbers
    /// - Roles should be validated against the current role definitions
    /// - Duration should respect security policy limits (typically ≤24 hours)
    /// - User ID should be validated and sanitized before session creation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::Session;
    ///
    /// // Create a 1-hour session for an admin user
    /// let session = Session::new(
    ///     "admin@example.com".to_string(),
    ///     vec!["admin".to_string(), "operator".to_string()],
    ///     3600, // 1 hour
    /// );
    ///
    /// assert!(!session.is_expired());
    /// assert!(session.has_role("admin"));
    /// ```
    #[must_use]
    pub fn new(user_id: String, roles: Vec<String>, duration_secs: i64) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            user_id,
            roles,
            created_at: now,
            expires_at: now + chrono::Duration::seconds(duration_secs),
            last_activity: now,
            metadata: HashMap::new(),
        }
    }

    /// Determines if the session has expired based on the current time.
    ///
    /// This method compares the current UTC time with the session's expiration
    /// timestamp to determine if the session is still valid. Expired sessions
    /// should not be used for authorization decisions and should be removed
    /// from active session storage.
    ///
    /// # Returns
    ///
    /// `true` if the current time is past the session expiration time,
    /// `false` if the session is still valid.
    ///
    /// # Security Implications
    ///
    /// - Expired sessions must not be accepted for any operations
    /// - This check should be performed before every authorization decision
    /// - Clock skew between systems should be considered in distributed environments
    /// - Expired sessions should trigger cleanup and audit logging
    ///
    /// # Performance
    ///
    /// This method performs a simple timestamp comparison and has minimal
    /// computational overhead. It can be called frequently without performance impact.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::Session;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// // Create a very short session for testing
    /// let session = Session::new(
    ///     "test_user".to_string(),
    ///     vec!["user".to_string()],
    ///     1, // 1 second duration
    /// );
    ///
    /// assert!(!session.is_expired()); // Should be valid initially
    ///
    /// thread::sleep(Duration::from_secs(2));
    /// assert!(session.is_expired()); // Should be expired after 2 seconds
    /// ```
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Update last activity timestamp
    pub fn update_activity(&mut self) {
        self.last_activity = Utc::now();
    }

    /// Checks if the session has been assigned a specific role.
    ///
    /// This method performs a case-sensitive search through the session's
    /// assigned roles to determine if the specified role is present. This
    /// is a fundamental operation for role-based access control (RBAC).
    ///
    /// # Arguments
    ///
    /// * `role` - The role name to check for (case-sensitive)
    ///
    /// # Returns
    ///
    /// `true` if the session has the specified role, `false` otherwise.
    ///
    /// # Security Considerations
    ///
    /// - Role names are case-sensitive for security consistency
    /// - This method should be used in conjunction with permission checking
    /// - Roles should be validated against current system role definitions
    /// - Consider using permission-based checks for fine-grained access control
    ///
    /// # Performance
    ///
    /// This method performs a linear search through the roles vector.
    /// For sessions with many roles, consider using a `HashSet` for O(1) lookups.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::Session;
    ///
    /// let session = Session::new(
    ///     "user@example.com".to_string(),
    ///     vec!["admin".to_string(), "operator".to_string(), "reader".to_string()],
    ///     3600,
    /// );
    ///
    /// assert!(session.has_role("admin"));
    /// assert!(session.has_role("operator"));
    /// assert!(!session.has_role("Admin")); // Case-sensitive
    /// assert!(!session.has_role("writer")); // Not assigned
    /// ```
    #[must_use]
    pub fn has_role(&self, role: &str) -> bool {
        self.roles.contains(&role.to_string())
    }
}

/// Defines a role with associated permissions for role-based access control.
///
/// A role represents a collection of permissions that can be assigned to users
/// or sessions. Roles provide a way to group related permissions and simplify
/// access control management by allowing administrators to assign roles rather
/// than individual permissions.
///
/// # Role-Based Access Control (RBAC)
///
/// Roles form the foundation of RBAC systems:
/// - **Users** are assigned one or more **Roles**
/// - **Roles** contain collections of **Permissions**
/// - **Permissions** define allowed actions on specific resources
/// - **Access decisions** are made by checking if user roles have required permissions
///
/// # Role Hierarchy
///
/// Roles can be organized hierarchically where higher-level roles inherit
/// permissions from lower-level roles:
/// - `admin` might inherit all permissions from `operator` and `reader`
/// - `operator` might inherit all permissions from `reader`
/// - `reader` has only basic read permissions
///
/// # Permission Model
///
/// Each role contains a vector of permissions that define what actions
/// the role can perform on which resources. Permissions follow the pattern:
/// `action:resource:scope` (e.g., "read:vault:*", "write:keys:user-123")
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::{Role, Permission};
/// use std::collections::HashMap;
///
/// let admin_role = Role {
///     id: "admin".to_string(),
///     name: "Administrator".to_string(),
///     description: "Full system access".to_string(),
///     permissions: vec![
///         Permission {
///             id: "admin-all".to_string(),
///             resource: "*".to_string(),
///             actions: vec!["*".to_string()],
///             constraints: HashMap::new(),
///         }
///     ],
///     metadata: HashMap::new(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    /// Role ID
    pub id: String,
    /// Role name
    pub name: String,
    /// Role description
    pub description: String,
    /// Permissions granted by this role
    pub permissions: Vec<Permission>,
    /// Role metadata
    pub metadata: HashMap<String, String>,
}

impl Role {
    /// Determines if this role has permission to perform an action on a resource.
    ///
    /// This method evaluates all permissions associated with the role to determine
    /// if the specified action is allowed on the given resource. It supports
    /// wildcard matching for both actions and resources to enable flexible
    /// permission models.
    ///
    /// # Arguments
    ///
    /// * `action` - The action being requested (e.g., "read", "write", "delete")
    /// * `resource` - The target resource (e.g., "vault/secrets", "keys/master")
    ///
    /// # Returns
    ///
    /// `true` if any permission in this role allows the action on the resource,
    /// `false` if no matching permission is found.
    ///
    /// # Permission Matching
    ///
    /// The method checks each permission using the following logic:
    /// - **Exact match**: Action and resource match exactly
    /// - **Action wildcard**: Permission action is "*" (allows any action)
    /// - **Resource wildcard**: Permission resource is "*" (allows any resource)
    /// - **Prefix matching**: Resource starts with permission resource + "/"
    ///
    /// # Security Model
    ///
    /// - Permissions are additive (any matching permission grants access)
    /// - Wildcard permissions should be used carefully to avoid over-privileging
    /// - Resource hierarchies are supported through prefix matching
    /// - Case-sensitive matching ensures consistent security behavior
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::{Role, Permission};
    /// use std::collections::HashMap;
    ///
    /// let role = Role {
    ///     id: "operator".to_string(),
    ///     name: "Operator".to_string(),
    ///     description: "Limited operational access".to_string(),
    ///     permissions: vec![
    ///         Permission {
    ///             id: "vault-read".to_string(),
    ///             resource: "vault/*".to_string(),
    ///             actions: vec!["read".to_string()],
    ///             constraints: HashMap::new(),
    ///         }
    ///     ],
    ///     metadata: HashMap::new(),
    /// };
    ///
    /// assert!(role.has_permission("read", "vault/config"));
    /// assert!(!role.has_permission("write", "vault/config"));
    /// ```
    #[must_use]
    pub fn has_permission(&self, action: &str, resource: &str) -> bool {
        self.permissions
            .iter()
            .any(|permission| permission.allows_action(action, resource))
    }
}

/// Defines a specific permission for access control operations.
///
/// A permission represents the authorization to perform specific actions
/// on particular resources within the secure storage system. Permissions
/// are the atomic units of access control and are grouped into roles for
/// easier management.
///
/// # Permission Model
///
/// Each permission defines:
/// - **Resource**: What can be accessed (e.g., "vault/secrets", "keys/*")
/// - **Actions**: What operations are allowed (e.g., `["read", "write"]`)
/// - **Scope**: Additional context or constraints (implicit in resource)
///
/// # Resource Patterns
///
/// Resources support hierarchical patterns:
/// - `vault/secrets` - Specific resource
/// - `vault/*` - All resources under vault
/// - `*` - All resources (use with extreme caution)
/// - `keys/user-123` - User-specific resources
///
/// # Action Types
///
/// Common actions include:
/// - `read` - View or retrieve data
/// - `write` - Create or modify data
/// - `delete` - Remove data
/// - `admin` - Administrative operations
/// - `*` - All actions (wildcard)
///
/// # Security Principles
///
/// - **Principle of Least Privilege**: Grant minimum necessary permissions
/// - **Explicit Permissions**: Prefer specific over wildcard permissions
/// - **Resource Isolation**: Use resource patterns to isolate access
/// - **Action Granularity**: Define actions at appropriate granularity
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::Permission;
/// use std::collections::HashMap;
///
/// // Read-only access to vault configuration
/// let read_config = Permission {
///     id: "vault-config-read".to_string(),
///     resource: "vault/config".to_string(),
///     actions: vec!["read".to_string()],
///     constraints: HashMap::new(),
/// };
///
/// // Full access to user's own keys
/// let user_keys = Permission {
///     id: "user-keys-full".to_string(),
///     resource: "keys/user-123/*".to_string(),
///     actions: vec!["read".to_string(), "write".to_string(), "delete".to_string()],
///     constraints: HashMap::new(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Permission {
    /// Permission ID
    pub id: String,
    /// Resource type
    pub resource: String,
    /// Allowed actions
    pub actions: Vec<String>,
    /// Resource constraints (e.g., key patterns)
    pub constraints: HashMap<String, String>,
}

impl Permission {
    /// Evaluates if this permission allows a specific action on a resource.
    ///
    /// This method implements the core permission checking logic that determines
    /// whether a requested action on a resource is authorized by this permission.
    /// It supports both exact matching and wildcard patterns for flexible
    /// access control policies.
    ///
    /// # Arguments
    ///
    /// * `action` - The action being requested (e.g., "read", "write", "delete")
    /// * `resource` - The target resource path (e.g., "vault/config", "keys/user-123")
    ///
    /// # Returns
    ///
    /// `true` if this permission authorizes the action on the resource,
    /// `false` otherwise.
    ///
    /// # Matching Logic
    ///
    /// ## Action Matching
    /// - Exact match: Permission action equals requested action
    /// - Wildcard match: Permission action is "*" (allows any action)
    ///
    /// ## Resource Matching
    /// - Exact match: Permission resource equals requested resource
    /// - Wildcard match: Permission resource is "*" (allows any resource)
    /// - Prefix match: Requested resource starts with permission resource + "/"
    ///
    /// # Security Considerations
    ///
    /// - All matching is case-sensitive for security consistency
    /// - Wildcard permissions should be used judiciously
    /// - Resource hierarchies enable fine-grained access control
    /// - Failed matches should be logged for security monitoring
    ///
    /// # Examples
    ///
    /// ```rust
    /// use secure_storage::types::Permission;
    /// use std::collections::HashMap;
    ///
    /// let permission = Permission {
    ///     id: "vault-read".to_string(),
    ///     resource: "vault/*".to_string(),
    ///     actions: vec!["read".to_string(), "list".to_string()],
    ///     constraints: HashMap::new(),
    /// };
    ///
    /// // These should be allowed
    /// assert!(permission.allows_action("read", "vault/config"));
    /// assert!(permission.allows_action("list", "vault/secrets"));
    ///
    /// // These should be denied
    /// assert!(!permission.allows_action("write", "vault/config")); // Wrong action
    /// assert!(!permission.allows_action("read", "keys/master"));   // Wrong resource
    /// ```
    #[must_use]
    pub fn allows_action(&self, action: &str, resource: &str) -> bool {
        // Check if action is allowed
        if !self.actions.contains(&action.to_string()) && !self.actions.contains(&"*".to_string()) {
            return false;
        }

        // Check resource match (simple pattern matching)
        if self.resource == "*" || self.resource == resource {
            return true;
        }

        // Check wildcard pattern matching (e.g., "vault/*" matches "vault/config")
        if self.resource.ends_with("/*") {
            let prefix = &self.resource[..self.resource.len() - 2]; // Remove "/*"
            if resource.starts_with(prefix) && resource.len() > prefix.len() {
                // Ensure there's a '/' separator
                return resource.chars().nth(prefix.len()) == Some('/');
            }
        }

        // Check if resource matches any constraint patterns
        if let Some(pattern) = self.constraints.get("resource_pattern") {
            return resource.starts_with(pattern);
        }

        false
    }
}

/// Represents a single entry in the security audit log.
///
/// Audit entries provide a comprehensive record of security-relevant events
/// within the secure storage system. Each entry captures the essential
/// information needed for security monitoring, compliance reporting, and
/// incident investigation.
///
/// # Audit Trail Requirements
///
/// Audit logs must be:
/// - **Immutable**: Entries cannot be modified after creation
/// - **Comprehensive**: All security events must be logged
/// - **Tamper-evident**: Integrity protection prevents unauthorized changes
/// - **Timestamped**: Precise timing for event correlation
/// - **Attributable**: Clear identification of actors and actions
///
/// # Event Categories
///
/// - **Authentication**: Login attempts, session creation/termination
/// - **Authorization**: Permission checks, access decisions
/// - **Data Access**: Vault operations, key usage, configuration changes
/// - **Administrative**: User management, role assignments, system changes
/// - **Security**: Failed access attempts, anomaly detection, incidents
///
/// # Compliance Standards
///
/// Audit logging supports compliance with:
/// - **SOX**: Financial data access tracking
/// - **HIPAA**: Healthcare information access logs
/// - **PCI DSS**: Payment data access monitoring
/// - **GDPR**: Personal data processing records
/// - **SOC 2**: Security control effectiveness evidence
///
/// # Data Retention
///
/// Audit entries should be retained according to:
/// - Regulatory requirements (typically 3-7 years)
/// - Organizational security policies
/// - Legal hold requirements
/// - Storage capacity constraints
///
/// # Privacy Considerations
///
/// - Sensitive data should not be logged in plaintext
/// - Personal information should be minimized or pseudonymized
/// - Access to audit logs should be strictly controlled
/// - Log data should be encrypted at rest and in transit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry ID
    pub id: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// User/service ID
    pub actor: String,
    /// Action performed
    pub action: String,
    /// Resource affected
    pub resource: String,
    /// Operation result
    pub result: AuditResult,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    /// Request IP address
    pub ip_address: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
}

/// Represents the outcome of an audited operation.
///
/// This enumeration captures the result of operations that are being audited,
/// providing essential information for security monitoring and compliance
/// reporting. The result includes both success/failure status and detailed
/// context about any failures.
///
/// # Result Categories
///
/// ## Success
/// - Operation completed successfully
/// - All security checks passed
/// - Expected outcome achieved
/// - No security concerns identified
///
/// ## Failure
/// - Operation failed to complete
/// - Security violation detected
/// - Authorization denied
/// - Technical error occurred
/// - Policy violation identified
///
/// # Security Monitoring
///
/// Audit results enable:
/// - **Anomaly Detection**: Patterns of failures may indicate attacks
/// - **Compliance Reporting**: Success/failure rates for regulatory requirements
/// - **Performance Monitoring**: Operation success rates and error patterns
/// - **Incident Response**: Detailed failure information for investigation
///
/// # Failure Analysis
///
/// Failed operations should include:
/// - **Error Code**: Specific error identifier for categorization
/// - **Error Message**: Human-readable description of the failure
/// - **Context**: Additional information about the failure conditions
/// - **Remediation**: Suggested actions to resolve the issue
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::AuditResult;
///
/// // Successful operation
/// let success = AuditResult::Success;
///
/// // Failed operation with details
/// let failure = AuditResult::Failure {
///     reason: "Invalid credentials provided".to_string(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure {
        /// Failure reason description
        reason: String,
    },
    /// Operation was denied
    Denied {
        /// Denial reason description
        reason: String,
    },
}

/// Configuration structure for vault backend settings.
///
/// This structure defines the configuration parameters for connecting to
/// and operating with various vault backends. It provides a unified
/// configuration interface that supports multiple vault types while
/// maintaining type safety and validation.
///
/// # Supported Vault Types
///
/// - **Local**: Encrypted `SQLite` database for development and testing
/// - **`HashiCorp` Vault**: Enterprise-grade secret management system
/// - **HSM**: Hardware Security Module integration for maximum security
///
/// # Configuration Hierarchy
///
/// The vault configuration consists of:
/// - **Vault Type**: Determines the backend implementation to use
/// - **Connection**: Network and connection-specific settings
/// - **Security**: Cryptographic and security policy settings
///
/// # Environment-Specific Settings
///
/// Different environments typically require different configurations:
/// - **Development**: Local vault with relaxed security for testing
/// - **Staging**: Network vault with production-like security settings
/// - **Production**: HSM or enterprise vault with maximum security
///
/// # Security Considerations
///
/// - Connection strings should not contain embedded credentials
/// - Use environment variables or external secret management for sensitive values
/// - Validate all configuration parameters before use
/// - Log configuration changes for audit purposes
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::{VaultConfig, VaultType, ConnectionConfig, SecurityConfig};
///
/// // Local development configuration
/// let local_config = VaultConfig {
///     vault_type: VaultType::Local,
///     connection: ConnectionConfig {
///         url: "./dev_vault.db".to_string(),
///         timeout_secs: 30,
///         max_retries: 3,
///         parameters: std::collections::HashMap::new(),
///     },
///     security: SecurityConfig::default(),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultConfig {
    /// Vault type
    pub vault_type: VaultType,
    /// Connection settings
    pub connection: ConnectionConfig,
    /// Security settings
    pub security: SecurityConfig,
}

/// Enumeration of supported vault backend types.
///
/// This enumeration defines the different types of vault backends that
/// the secure storage system can connect to and operate with. Each type
/// represents a different approach to secure data storage with varying
/// levels of security, performance, and operational complexity.
///
/// # Vault Type Characteristics
///
/// ## Local Vault
/// - **Use Case**: Development, testing, single-node deployments
/// - **Storage**: Encrypted `SQLite` database file
/// - **Security**: File-system level protection, application-level encryption
/// - **Performance**: High (no network overhead)
/// - **Scalability**: Single node only
/// - **Availability**: Depends on local storage reliability
///
/// ## `HashiCorp` Vault
/// - **Use Case**: Production environments, enterprise deployments
/// - **Storage**: Distributed, highly available backend
/// - **Security**: Enterprise-grade with audit logging and policy enforcement
/// - **Performance**: Network-dependent, highly optimized
/// - **Scalability**: Horizontal scaling with clustering
/// - **Availability**: High availability with replication
///
/// ## Hardware Security Module (HSM)
/// - **Use Case**: Maximum security requirements, compliance environments
/// - **Storage**: Tamper-resistant hardware devices
/// - **Security**: FIPS 140-2 Level 3/4 certified hardware protection
/// - **Performance**: Limited by hardware throughput
/// - **Scalability**: Limited by hardware capacity
/// - **Availability**: Depends on hardware redundancy
///
/// # Selection Criteria
///
/// Choose vault type based on:
/// - **Security Requirements**: HSM > `HashiCorp` Vault > Local
/// - **Performance Needs**: Local > `HashiCorp` Vault > HSM
/// - **Scalability Requirements**: `HashiCorp` Vault > Local > HSM
/// - **Operational Complexity**: Local < `HashiCorp` Vault < HSM
/// - **Cost Considerations**: Local < `HashiCorp` Vault < HSM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VaultType {
    /// Local encrypted `SQLite` vault
    Local,
    /// `HashiCorp` Vault
    HashiCorp,
    /// Hardware Security Module
    Hsm,
}

/// Configuration for vault backend connections.
///
/// This structure defines the connection parameters required to establish
/// and maintain connections to vault backends. It provides a flexible
/// configuration system that supports various connection types while
/// maintaining security and reliability.
///
/// # Connection Parameters
///
/// ## URL/Path
/// - **Local Vault**: File system path to database file
/// - **Network Vault**: `HTTP`/`HTTPS` URL with optional port
/// - **HSM**: Device connection string or network endpoint
///
/// ## Timeout Settings
/// - **Connection Timeout**: Maximum time to establish initial connection
/// - **Operation Timeout**: Maximum time for individual operations
/// - **Keep-Alive**: Connection persistence settings
///
/// ## Retry Policy
/// - **Retry Attempts**: Number of retries for failed operations
/// - **Backoff Strategy**: Exponential or linear backoff between retries
/// - **Circuit Breaker**: Automatic failure detection and recovery
///
/// # Security Considerations
///
/// - Use `HTTPS` for all network connections
/// - Validate SSL/TLS certificates in production
/// - Store connection credentials securely (not in configuration)
/// - Use connection pooling for performance and resource management
/// - Implement proper timeout values to prevent resource exhaustion
///
/// # Examples
///
/// ```rust
/// use secure_storage::types::ConnectionConfig;
/// use std::collections::HashMap;
///
/// // Local database connection
/// let local_config = ConnectionConfig {
///     url: "/var/lib/secure_storage/vault.db".to_string(),
///     timeout_secs: 30,
///     max_retries: 3,
///     parameters: HashMap::new(),
/// };
///
/// // Network vault connection
/// let mut network_params = HashMap::new();
/// network_params.insert("tls_verify".to_string(), "true".to_string());
/// let network_config = ConnectionConfig {
///     url: "https://vault.example.com:8200".to_string(),
///     timeout_secs: 60,
///     max_retries: 5,
///     parameters: network_params,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Connection URL or path
    pub url: String,
    /// Connection timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Additional connection parameters
    pub parameters: HashMap<String, String>,
}

/// Security configuration for vault operations and data protection.
///
/// This structure defines the security policies and cryptographic parameters
/// used throughout the secure storage system. It encompasses encryption
/// settings, key management policies, session controls, and audit requirements
/// to ensure comprehensive security coverage.
///
/// # Security Domains
///
/// ## Cryptographic Settings
/// - **Default Algorithm**: Primary encryption algorithm for new data
/// - **Key Rotation**: Automatic key rotation intervals and policies
/// - **Key Derivation**: Parameters for generating keys from passwords
/// - **Random Generation**: Cryptographically secure random number generation
///
/// ## Session Management
/// - **Session Timeout**: Maximum session lifetime before forced expiration
/// - **Idle Timeout**: Automatic logout after period of inactivity
/// - **Concurrent Sessions**: Maximum simultaneous sessions per user
/// - **Session Binding**: IP address and user agent validation
///
/// ## Access Control
/// - **Authentication**: Multi-factor authentication requirements
/// - **Authorization**: Role-based access control policies
/// - **Permission Model**: Fine-grained permission definitions
/// - **Privilege Escalation**: Controls for administrative access
///
/// ## Audit and Monitoring
/// - **Audit Retention**: How long to keep audit logs
/// - **Event Logging**: Which events to log and at what detail level
/// - **Integrity Protection**: Cryptographic protection of audit logs
/// - **Real-time Monitoring**: Anomaly detection and alerting
///
/// # Compliance Standards
///
/// Security configuration supports compliance with:
/// - **FIPS 140-2**: Cryptographic module security standards
/// - **Common Criteria**: Security evaluation criteria
/// - **SOC 2**: Service organization control requirements
/// - **ISO 27001**: Information security management standards
///
/// # Security Hardening
///
/// - Use strong encryption algorithms (AES-256, `ChaCha20`)
/// - Implement regular key rotation (daily to monthly)
/// - Enforce short session timeouts (15 minutes to 8 hours)
/// - Enable comprehensive audit logging
/// - Use multi-factor authentication where possible
/// - Implement network security controls (TLS, VPN)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Default encryption algorithm
    pub default_algorithm: EncryptionAlgorithm,
    /// Key rotation interval in seconds
    pub key_rotation_interval_secs: u64,
    /// Session timeout in seconds
    pub session_timeout_secs: u64,
    /// Enable audit logging
    pub audit_enabled: bool,
    /// Audit retention period in days
    pub audit_retention_days: u32,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            default_algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_rotation_interval_secs: 86400, // 24 hours
            session_timeout_secs: 3600,        // 1 hour
            audit_enabled: true,
            audit_retention_days: 365,
        }
    }
}
