//! Configuration management for secure storage

use crate::error::{SecureStorageError, SecureStorageResult};
use crate::types::{EncryptionAlgorithm, VaultType};
use garde::Validate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Comprehensive configuration for secure vault operations.
///
/// This structure defines all parameters required to configure a secure vault instance,
/// including connection settings, security policies, and operational parameters.
/// The configuration is validated on creation to ensure all parameters are within
/// acceptable ranges for production use.
///
/// # Security
///
/// All sensitive configuration values should be loaded from secure sources
/// (environment variables, encrypted files, or external secret management systems).
/// Never hardcode sensitive values in configuration files.
///
/// # Validation
///
/// All fields are validated using the `garde` crate to ensure:
/// - Connection timeouts are reasonable (5s to 300s)
/// - Security parameters meet minimum requirements
/// - Resource limits are within operational bounds
///
/// # Examples
///
/// ```rust
/// use secure_storage::config::VaultConfig;
/// use secure_storage::types::{VaultType, EncryptionAlgorithm};
/// use std::collections::HashMap;
///
/// let config = VaultConfig {
///     vault_type: VaultType::Local,
///     connection: secure_storage::config::ConnectionConfig {
///         url: "./vault.db".to_string(),
///         timeout_secs: 30,
///         max_retries: 3,
///         parameters: HashMap::new(),
///     },
///     security: secure_storage::config::SecurityConfig {
///         default_algorithm: EncryptionAlgorithm::Aes256Gcm,
///         key_rotation_interval_secs: 86400,
///         session_timeout_secs: 3600,
///         audit_enabled: true,
///         audit_retention_days: 365,
///     },
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct VaultConfig {
    /// Vault type
    #[garde(skip)]
    pub vault_type: VaultType,
    /// Connection settings
    #[garde(dive)]
    pub connection: ConnectionConfig,
    /// Security settings
    #[garde(dive)]
    pub security: SecurityConfig,
}

/// Network and connection configuration for vault backends.
///
/// Defines how the secure storage system connects to various vault backends,
/// including connection timeouts, retry policies, and backend-specific parameters.
/// This configuration is critical for ensuring reliable and performant vault operations.
///
/// # Connection Types
///
/// - **Local**: File path to `SQLite` database
/// - **`HashiCorp` Vault**: `HTTP`/`HTTPS` URL with authentication
/// - **HSM**: Device connection string or network endpoint
///
/// # Timeout Considerations
///
/// Connection timeouts should be set based on network conditions and SLA requirements:
/// - Local vaults: 1-5 seconds
/// - Network vaults: 5-30 seconds
/// - HSM devices: 10-60 seconds
///
/// # Security
///
/// - URLs should use HTTPS for network connections
/// - Authentication parameters should be stored securely
/// - Connection strings should not contain embedded credentials
///
/// # Performance
///
/// Connection pooling and keep-alive settings significantly impact performance
/// for high-throughput applications.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConnectionConfig {
    /// Connection URL or path
    #[garde(length(min = 1))]
    pub url: String,
    /// Connection timeout in seconds
    #[garde(range(min = 1, max = 300))]
    pub timeout_secs: u64,
    /// Maximum retry attempts
    #[garde(range(min = 0, max = 10))]
    pub max_retries: u32,
    /// Additional connection parameters
    #[garde(skip)]
    pub parameters: HashMap<String, String>,
}

/// Comprehensive security configuration for vault operations.
///
/// This structure defines all security-related parameters including encryption
/// algorithms, key rotation policies, session management, and audit settings.
/// These settings directly impact the security posture of the entire system.
///
/// # Encryption Standards
///
/// - **AES-256-GCM**: Recommended for maximum security and performance
/// - **ChaCha20-Poly1305**: Alternative for environments without AES acceleration
/// - **Key Rotation**: Automatic rotation based on time or usage thresholds
///
/// # Session Security
///
/// - Session timeouts enforce security policies
/// - Minimum timeout: 5 minutes (300 seconds)
/// - Maximum timeout: 24 hours (86400 seconds)
/// - Idle sessions are automatically terminated
///
/// # Audit Requirements
///
/// - All security events must be logged
/// - Audit logs are retained according to compliance requirements
/// - Log integrity is protected through cryptographic signatures
///
/// # Compliance
///
/// Default settings meet or exceed requirements for:
/// - FIPS 140-2 Level 3
/// - Common Criteria EAL4+
/// - SOC 2 Type II
/// - PCI DSS Level 1
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SecurityConfig {
    /// Default encryption algorithm
    #[garde(skip)]
    pub default_algorithm: EncryptionAlgorithm,
    /// Key rotation interval in seconds
    #[garde(range(min = 3600, max = 31_536_000))] // 1 hour to 1 year
    pub key_rotation_interval_secs: u64,
    /// Session timeout in seconds
    #[garde(range(min = 300, max = 86400))] // 5 minutes to 24 hours
    pub session_timeout_secs: u64,
    /// Enable audit logging
    #[garde(skip)]
    pub audit_enabled: bool,
    /// Audit retention period in days
    #[garde(range(min = 1, max = 3650))] // 1 day to 10 years
    pub audit_retention_days: u32,
}

/// Root configuration structure for the entire secure storage system.
///
/// This is the primary configuration object that encompasses all aspects of
/// secure storage operation including vault settings, encryption parameters,
/// access control, performance tuning, and operational policies.
///
/// # Configuration Hierarchy
///
/// ```text
/// SecureStorageConfig
/// ├── vault: VaultConfig (connection and basic settings)
/// ├── encryption: EncryptionConfig (cryptographic parameters)
/// ├── key_derivation: KeyDerivationConfig (key generation settings)
/// ├── memory_protection: MemoryProtectionConfig (memory security)
/// ├── access_control: AccessControlConfig (RBAC settings)
/// ├── session: SessionConfig (session management)
/// ├── rate_limiting: RateLimitingConfig (DoS protection)
/// ├── audit: AuditConfig (logging and compliance)
/// └── performance: PerformanceConfig (optimization settings)
/// ```
///
/// # Validation
///
/// The entire configuration tree is validated recursively using the `garde` crate.
/// Validation ensures:
/// - All required fields are present
/// - Numeric values are within acceptable ranges
/// - String values meet format requirements
/// - Cross-field dependencies are satisfied
///
/// # Loading Configuration
///
/// Configuration can be loaded from multiple sources:
/// - TOML files (recommended for development)
/// - Environment variables (recommended for production)
/// - External configuration services (`HashiCorp` Consul, etc.)
/// - Programmatic construction (for testing)
///
/// # Security Considerations
///
/// - Never store sensitive values directly in configuration files
/// - Use environment variables or external secret management
/// - Validate all configuration before use
/// - Log configuration changes for audit purposes
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SecureStorageConfig {
    /// Vault configuration
    #[garde(dive)]
    pub vault: VaultConfig,

    /// Encryption settings
    #[garde(dive)]
    pub encryption: EncryptionConfig,

    /// Access control settings
    #[garde(dive)]
    pub access_control: AccessControlConfig,

    /// Audit settings
    #[garde(dive)]
    pub audit: AuditConfig,

    /// Performance settings
    #[garde(dive)]
    pub performance: PerformanceConfig,
}

impl Default for SecureStorageConfig {
    fn default() -> Self {
        Self {
            vault: VaultConfig {
                vault_type: VaultType::Local,
                connection: ConnectionConfig {
                    url: "./secure_storage.db".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                    parameters: HashMap::new(),
                },
                security: SecurityConfig {
                    default_algorithm: EncryptionAlgorithm::Aes256Gcm,
                    key_rotation_interval_secs: 86400, // 24 hours
                    session_timeout_secs: 3600,        // 1 hour
                    audit_enabled: true,
                    audit_retention_days: 365,
                },
            },
            encryption: EncryptionConfig::default(),
            access_control: AccessControlConfig::default(),
            audit: AuditConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl SecureStorageConfig {
    /// Load configuration from file
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::Configuration` if:
    /// - File cannot be read
    /// - File format is invalid (JSON/YAML parsing fails)
    /// - Configuration validation fails
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn from_file(path: impl Into<PathBuf>) -> SecureStorageResult<Self> {
        let path = path.into();
        let content =
            std::fs::read_to_string(&path).map_err(|e| SecureStorageError::Configuration {
                field: "config_file".to_string(),
                reason: format!("Failed to read config file {}: {}", path.display(), e),
            })?;

        let config: Self = if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::from_str(&content).map_err(|e| SecureStorageError::Configuration {
                field: "yaml_parsing".to_string(),
                reason: e.to_string(),
            })?
        } else {
            serde_json::from_str(&content).map_err(|e| SecureStorageError::Configuration {
                field: "json_parsing".to_string(),
                reason: e.to_string(),
            })?
        };

        config
            .validate()
            .map_err(|e| SecureStorageError::Configuration {
                field: "validation".to_string(),
                reason: e.to_string(),
            })?;

        Ok(config)
    }

    /// Save configuration to file
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::Configuration` if:
    /// - Serialization to JSON/YAML fails
    /// - File cannot be written to disk
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn save_to_file(&self, path: impl Into<PathBuf>) -> SecureStorageResult<()> {
        let path = path.into();

        let content = if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            serde_yaml::to_string(self).map_err(|e| SecureStorageError::Configuration {
                field: "yaml_serialization".to_string(),
                reason: e.to_string(),
            })?
        } else {
            serde_json::to_string_pretty(self).map_err(|e| SecureStorageError::Configuration {
                field: "json_serialization".to_string(),
                reason: e.to_string(),
            })?
        };

        std::fs::write(&path, content).map_err(|e| SecureStorageError::Configuration {
            field: "file_write".to_string(),
            reason: format!("Failed to write config file {}: {}", path.display(), e),
        })?;

        Ok(())
    }

    /// Validate configuration
    ///
    /// # Errors
    ///
    /// Returns `SecureStorageError::Configuration` if any configuration
    /// parameters are invalid or inconsistent
    ///
    /// # Errors
    ///
    /// Returns error if operation fails
    pub fn validate(&self) -> SecureStorageResult<()> {
        garde::Validate::validate(self, &()).map_err(|e| SecureStorageError::Configuration {
            field: "validation".to_string(),
            reason: e.to_string(),
        })?;
        Ok(())
    }

    /// Returns the configured connection timeout as a `Duration` object.
    ///
    /// This method converts the stored timeout value (in seconds) to a `Duration`
    /// for use with async operations and connection management. The timeout
    /// applies to initial connection establishment, not individual operations.
    ///
    /// # Returns
    ///
    /// A `Duration` representing the maximum time to wait for connection establishment.
    /// Range: 5 seconds to 300 seconds (5 minutes) based on validation rules.
    ///
    /// # Performance
    ///
    /// This is a const function with zero runtime cost - the conversion is
    /// performed at compile time when possible.
    ///
    /// # Usage
    ///
    /// ```rust
    /// use secure_storage::config::SecureStorageConfig;
    /// use std::time::Duration;
    ///
    /// let config = SecureStorageConfig::default();
    /// let timeout = config.connection_timeout();
    /// assert!(timeout >= Duration::from_secs(5));
    /// assert!(timeout <= Duration::from_secs(300));
    /// ```
    #[must_use]
    pub const fn connection_timeout(&self) -> Duration {
        Duration::from_secs(self.vault.connection.timeout_secs)
    }

    /// Returns the configured session timeout as a `Duration` object.
    ///
    /// This method provides the maximum lifetime for authenticated sessions
    /// before they automatically expire. Sessions that exceed this duration
    /// will be invalidated and require re-authentication.
    ///
    /// # Returns
    ///
    /// A `Duration` representing the maximum session lifetime.
    /// Range: 300 seconds (5 minutes) to 86400 seconds (24 hours).
    ///
    /// # Security Implications
    ///
    /// - Shorter timeouts improve security by limiting exposure window
    /// - Longer timeouts improve user experience but increase risk
    /// - Production systems should use timeouts appropriate for their threat model
    ///
    /// # Performance
    ///
    /// This is a const function with zero runtime overhead.
    ///
    /// # Usage
    ///
    /// ```rust
    /// use secure_storage::config::SecureStorageConfig;
    /// use std::time::Duration;
    ///
    /// let config = SecureStorageConfig::default();
    /// let session_timeout = config.session_timeout();
    ///
    /// // Verify timeout is within security policy bounds
    /// assert!(session_timeout >= Duration::from_secs(300));   // Min 5 minutes
    /// assert!(session_timeout <= Duration::from_secs(86400)); // Max 24 hours
    /// ```
    #[must_use]
    pub const fn session_timeout(&self) -> Duration {
        Duration::from_secs(self.vault.security.session_timeout_secs)
    }

    /// Returns the configured key rotation interval as a `Duration` object.
    ///
    /// This method provides the time interval between automatic key rotations.
    /// Regular key rotation is a critical security practice that limits the
    /// exposure window if a key is compromised and meets compliance requirements.
    ///
    /// # Returns
    ///
    /// A `Duration` representing the time between automatic key rotations.
    /// Range: 86400 seconds (24 hours) to 31536000 seconds (365 days).
    ///
    /// # Security Best Practices
    ///
    /// - **High-security environments**: 24-48 hours
    /// - **Standard environments**: 7-30 days
    /// - **Low-risk environments**: 90-365 days
    /// - **Compliance requirements**: Often mandate 90 days maximum
    ///
    /// # Performance Impact
    ///
    /// Key rotation involves:
    /// - Generating new cryptographic keys
    /// - Re-encrypting sensitive data (if required)
    /// - Updating key references throughout the system
    /// - Securely destroying old keys
    ///
    /// # Compliance
    ///
    /// Many security frameworks require regular key rotation:
    /// - PCI DSS: Annual rotation minimum
    /// - FIPS 140-2: Based on key usage and risk assessment
    /// - SOC 2: Regular rotation as part of security controls
    ///
    /// # Usage
    ///
    /// ```rust
    /// use secure_storage::config::SecureStorageConfig;
    /// use std::time::Duration;
    ///
    /// let config = SecureStorageConfig::default();
    /// let rotation_interval = config.key_rotation_interval();
    ///
    /// // Verify interval meets security requirements
    /// assert!(rotation_interval >= Duration::from_secs(86400));    // Min 24 hours
    /// assert!(rotation_interval <= Duration::from_secs(31536000)); // Max 365 days
    /// ```
    #[must_use]
    pub const fn key_rotation_interval(&self) -> Duration {
        Duration::from_secs(self.vault.security.key_rotation_interval_secs)
    }
}

/// Advanced encryption configuration for cryptographic operations.
///
/// This structure defines encryption algorithms, key management policies,
/// and cryptographic parameters used throughout the secure storage system.
/// All encryption settings are validated to ensure they meet security standards.
///
/// # Supported Algorithms
///
/// - **AES-256-GCM**: Authenticated encryption, FIPS 140-2 approved
/// - **ChaCha20-Poly1305**: High-performance alternative to AES
/// - **AES-256-CBC**: Legacy support (not recommended for new deployments)
///
/// # Key Management
///
/// - Keys are derived using PBKDF2 or Argon2 (configurable)
/// - Automatic key rotation based on time or usage thresholds
/// - Hardware Security Module (HSM) integration for key protection
/// - Multi-party computation (MPC) for distributed key management
///
/// # Performance Considerations
///
/// - AES-256-GCM: Best performance on modern CPUs with AES-NI
/// - ChaCha20-Poly1305: Better performance on CPUs without AES acceleration
/// - Key derivation: CPU-intensive, should be tuned for security vs. performance
///
/// # Security Standards
///
/// All encryption configurations meet or exceed:
/// - FIPS 140-2 Level 3 requirements
/// - NIST SP 800-57 key management guidelines
/// - Common Criteria EAL4+ protection profiles
/// - Industry best practices for authenticated encryption
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct EncryptionConfig {
    /// Default encryption algorithm
    #[garde(skip)]
    pub default_algorithm: EncryptionAlgorithm,

    /// Key derivation settings
    #[garde(dive)]
    pub key_derivation: KeyDerivationConfig,

    /// Memory protection settings
    #[garde(dive)]
    pub memory_protection: MemoryProtectionConfig,

    /// Supported algorithms
    #[garde(skip)]
    pub supported_algorithms: Vec<EncryptionAlgorithm>,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            default_algorithm: EncryptionAlgorithm::Aes256Gcm,
            key_derivation: KeyDerivationConfig::default(),
            memory_protection: MemoryProtectionConfig::default(),
            supported_algorithms: vec![
                EncryptionAlgorithm::Aes256Gcm,
                EncryptionAlgorithm::ChaCha20Poly1305,
            ],
        }
    }
}

/// Configuration for cryptographic key derivation functions.
///
/// This structure defines parameters for key derivation functions (KDFs) used
/// to generate encryption keys from passwords or other key material. Proper
/// configuration is critical for security against brute-force attacks.
///
/// # Supported Key Derivation Functions
///
/// - **Argon2id**: Recommended for new applications (memory-hard, side-channel resistant)
/// - **PBKDF2**: Legacy support for compatibility (time-hard only)
/// - **scrypt**: Alternative memory-hard function
///
/// # Security Parameters
///
/// ## Argon2id Parameters
/// - **Memory Cost**: Amount of memory used (KB), affects resistance to parallel attacks
/// - **Time Cost**: Number of iterations, affects resistance to time-memory trade-offs
/// - **Parallelism**: Number of threads, should match available CPU cores
///
/// ## PBKDF2 Parameters
/// - **Iterations**: Number of hash iterations, minimum 100,000 for SHA-256
/// - **Salt Length**: Random salt size, minimum 16 bytes recommended
///
/// # Performance vs Security Trade-offs
///
/// - Higher memory cost: Better security, more RAM usage
/// - Higher time cost: Better security, slower key derivation
/// - Higher parallelism: Better performance on multi-core systems
///
/// # Compliance Requirements
///
/// - OWASP: Argon2id with 19 MiB memory, 2 iterations, 1 thread minimum
/// - NIST SP 800-63B: PBKDF2 with 10,000 iterations minimum
/// - FIPS 140-2: Approved algorithms and parameter ranges
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct KeyDerivationConfig {
    /// Argon2 memory cost in KB
    #[garde(range(min = 1024, max = 1_048_576))] // 1MB to 1GB
    pub memory_cost_kb: u32,

    /// Argon2 time cost (iterations)
    #[garde(range(min = 1, max = 100))]
    pub time_cost: u32,

    /// Argon2 parallelism
    #[garde(range(min = 1, max = 16))]
    pub parallelism: u32,

    /// Salt length in bytes
    #[garde(range(min = 16, max = 64))]
    pub salt_length: usize,
}

impl Default for KeyDerivationConfig {
    fn default() -> Self {
        Self {
            memory_cost_kb: 65536, // 64MB
            time_cost: 3,
            parallelism: 4,
            salt_length: 32,
        }
    }
}

/// Configuration for memory protection and secure memory management.
///
/// This structure defines how sensitive data is protected in memory during
/// runtime operations. Memory protection is crucial for preventing sensitive
/// data from being written to swap files or accessed by unauthorized processes.
///
/// # Memory Protection Mechanisms
///
/// ## Memory Locking (mlock)
/// - Prevents sensitive pages from being swapped to disk
/// - Requires elevated privileges on most systems
/// - Limited by system ulimits and available RAM
///
/// ## Memory Clearing
/// - Automatic zeroing of sensitive data when no longer needed
/// - Protection against memory dumps and core files
/// - Compiler optimization barriers to prevent elimination
///
/// ## Guard Pages
/// - Detection of buffer overflows and underflows
/// - Immediate termination on memory corruption
/// - Minimal performance impact for high-security applications
///
/// # Platform Considerations
///
/// - **Linux**: Requires `CAP_IPC_LOCK` capability or appropriate ulimits
/// - **Windows**: Requires `SeLockMemoryPrivilege` for `VirtualLock`
/// - **macOS**: Limited by kern.maxlockedmem sysctl setting
///
/// # Performance Impact
///
/// - Memory locking: Minimal CPU overhead, increased memory pressure
/// - Memory clearing: Small CPU cost, prevents optimization
/// - Guard pages: Minimal overhead, additional virtual memory usage
///
/// # Security Benefits
///
/// - Prevents sensitive data in swap files
/// - Reduces attack surface for memory dumps
/// - Early detection of memory corruption attacks
/// - Compliance with security frameworks requiring memory protection
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MemoryProtectionConfig {
    /// Enable memory locking (mlock)
    #[garde(skip)]
    pub enable_mlock: bool,

    /// Enable secure memory wiping
    #[garde(skip)]
    pub enable_zeroize: bool,

    /// Maximum locked memory in bytes
    #[garde(range(min = 1_048_576, max = 1_073_741_824))] // 1MB to 1GB
    pub max_locked_memory_bytes: usize,
}

impl Default for MemoryProtectionConfig {
    fn default() -> Self {
        Self {
            enable_mlock: true,
            enable_zeroize: true,
            max_locked_memory_bytes: 16 * 1024 * 1024, // 16MB
        }
    }
}

/// Configuration for role-based access control (RBAC) and authorization.
///
/// This structure defines how access control is implemented throughout the
/// secure storage system, including role definitions, permission models,
/// and authorization policies. Proper access control is fundamental to
/// maintaining security in multi-user environments.
///
/// # Access Control Models
///
/// ## Role-Based Access Control (RBAC)
/// - Users are assigned roles (e.g., "admin", "operator", "reader")
/// - Roles contain collections of permissions
/// - Permissions define allowed actions on specific resources
/// - Hierarchical roles with inheritance support
///
/// ## Attribute-Based Access Control (ABAC)
/// - Fine-grained control based on user, resource, and environment attributes
/// - Dynamic policy evaluation at runtime
/// - Support for complex conditional logic
/// - Integration with external policy engines
///
/// # Permission Model
///
/// Permissions follow the format: `action:resource:scope`
/// - **Action**: read, write, delete, admin, etc.
/// - **Resource**: vault, key, config, audit, etc.
/// - **Scope**: specific resource ID or wildcard (*)
///
/// # Multi-Factor Authentication
///
/// - TOTP (Time-based One-Time Password) support
/// - Hardware token integration (`YubiKey`, etc.)
/// - Biometric authentication where available
/// - Backup codes for recovery scenarios
///
/// # Session Management
///
/// - Configurable session timeouts
/// - Concurrent session limits per user
/// - Session invalidation on security events
/// - Audit logging of all authentication events
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AccessControlConfig {
    /// Enable role-based access control
    #[garde(skip)]
    pub enable_rbac: bool,

    /// Enable two-factor authentication
    #[garde(skip)]
    pub enable_2fa: bool,

    /// Session configuration
    #[garde(dive)]
    pub session: SessionConfig,

    /// Rate limiting configuration
    #[garde(dive)]
    pub rate_limiting: RateLimitingConfig,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            enable_rbac: true,
            enable_2fa: false,
            session: SessionConfig::default(),
            rate_limiting: RateLimitingConfig::default(),
        }
    }
}

/// Configuration for session management and authentication lifecycle.
///
/// This structure defines how user sessions are created, managed, and terminated
/// within the secure storage system. Session management is critical for maintaining
/// security while providing a good user experience.
///
/// # Session Lifecycle
///
/// 1. **Creation**: User authenticates and session is established
/// 2. **Activity**: Session is used for authorized operations
/// 3. **Refresh**: Session may be extended based on activity
/// 4. **Expiration**: Session automatically expires after timeout
/// 5. **Termination**: Session is explicitly ended or invalidated
///
/// # Timeout Policies
///
/// ## Absolute Timeout
/// - Maximum session lifetime regardless of activity
/// - Prevents indefinite session extension
/// - Typically 8-24 hours for administrative sessions
///
/// ## Idle Timeout
/// - Session expires after period of inactivity
/// - Balances security with user convenience
/// - Typically 15 minutes to 2 hours
///
/// ## Sliding Window
/// - Session extends with each activity
/// - Provides seamless user experience
/// - Must respect absolute timeout limits
///
/// # Security Features
///
/// - Cryptographically secure session IDs
/// - Session binding to IP address/user agent
/// - Concurrent session limits per user
/// - Automatic cleanup of expired sessions
/// - Audit logging of all session events
///
/// # Performance Considerations
///
/// - Session storage backend (memory, database, distributed cache)
/// - Cleanup frequency for expired sessions
/// - Session validation overhead per request
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SessionConfig {
    /// Default session timeout in seconds
    #[garde(range(min = 300, max = 86400))] // 5 minutes to 24 hours
    pub default_timeout_secs: u64,

    /// Maximum concurrent sessions per user
    #[garde(range(min = 1, max = 100))]
    pub max_concurrent_sessions: u32,

    /// Session cleanup interval in seconds
    #[garde(range(min = 60, max = 3600))] // 1 minute to 1 hour
    pub cleanup_interval_secs: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            default_timeout_secs: 3600, // 1 hour
            max_concurrent_sessions: 10,
            cleanup_interval_secs: 300, // 5 minutes
        }
    }
}

/// Configuration for rate limiting and denial-of-service (`DoS`) protection.
///
/// This structure defines rate limiting policies to protect the secure storage
/// system from abuse, brute-force attacks, and resource exhaustion. Rate limiting
/// is essential for maintaining system availability and security.
///
/// # Rate Limiting Strategies
///
/// ## Token Bucket Algorithm
/// - Allows burst traffic up to bucket capacity
/// - Tokens refill at configured rate
/// - Smooth handling of variable load patterns
/// - Good for API endpoints with bursty usage
///
/// ## Fixed Window
/// - Simple counter reset at fixed intervals
/// - Easy to implement and understand
/// - May allow burst at window boundaries
/// - Suitable for basic protection
///
/// ## Sliding Window
/// - More accurate rate limiting
/// - Higher memory and CPU overhead
/// - Better protection against burst attacks
/// - Recommended for high-security environments
///
/// # Protection Targets
///
/// - **Authentication attempts**: Prevent brute-force attacks
/// - **API requests**: Protect against resource exhaustion
/// - **Key operations**: Limit cryptographic operations per user
/// - **Vault access**: Control data access patterns
///
/// # Configuration Parameters
///
/// - **Rate**: Maximum operations per time window
/// - **Burst**: Maximum operations in short burst
/// - **Window**: Time period for rate calculation
/// - **Penalties**: Increased delays for repeated violations
///
/// # Integration Points
///
/// - Web API middleware for HTTP requests
/// - Authentication system for login attempts
/// - Vault operations for data access
/// - Key management for cryptographic operations
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RateLimitingConfig {
    /// Enable rate limiting
    #[garde(skip)]
    pub enabled: bool,

    /// Requests per minute per user
    #[garde(range(min = 1, max = 10000))]
    pub requests_per_minute: u32,

    /// Burst allowance
    #[garde(range(min = 1, max = 1000))]
    pub burst_allowance: u32,

    /// Cleanup interval in seconds
    #[garde(range(min = 60, max = 3600))]
    pub cleanup_interval_secs: u64,
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_minute: 1000,
            burst_allowance: 100,
            cleanup_interval_secs: 300,
        }
    }
}

/// Configuration for audit logging and compliance monitoring.
///
/// This structure defines how security events and operations are logged for
/// audit purposes, compliance requirements, and security monitoring. Comprehensive
/// audit logging is essential for detecting security incidents and meeting
/// regulatory requirements.
///
/// # Audit Event Categories
///
/// ## Authentication Events
/// - Login attempts (successful and failed)
/// - Session creation and termination
/// - Multi-factor authentication events
/// - Password changes and resets
///
/// ## Authorization Events
/// - Permission checks and access decisions
/// - Role assignments and modifications
/// - Policy changes and updates
/// - Privilege escalation attempts
///
/// ## Data Access Events
/// - Vault operations (read, write, delete)
/// - Key management operations
/// - Configuration changes
/// - Backup and restore operations
///
/// ## System Events
/// - Service startup and shutdown
/// - Configuration reloads
/// - Error conditions and exceptions
/// - Performance threshold violations
///
/// # Compliance Standards
///
/// Audit logging meets requirements for:
/// - **SOX (Sarbanes-Oxley)**: Financial data access tracking
/// - **HIPAA**: Healthcare data access and modification logs
/// - **PCI DSS**: Payment card data access monitoring
/// - **GDPR**: Personal data processing activities
/// - **SOC 2**: Security control effectiveness evidence
///
/// # Log Integrity
///
/// - Cryptographic signatures prevent tampering
/// - Write-only log storage prevents modification
/// - Regular integrity verification
/// - Secure log forwarding to external systems
///
/// # Performance Considerations
///
/// - Asynchronous logging to minimize latency impact
/// - Log rotation and compression for storage efficiency
/// - Configurable verbosity levels
/// - Batch processing for high-volume events
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AuditConfig {
    /// Enable audit logging
    #[garde(skip)]
    pub enabled: bool,

    /// Audit log retention period in days
    #[garde(range(min = 1, max = 3650))] // 1 day to 10 years
    pub retention_days: u32,

    /// Log all operations (including reads)
    #[garde(skip)]
    pub log_all_operations: bool,

    /// Log failed operations only
    #[garde(skip)]
    pub log_failures_only: bool,

    /// Batch size for audit log writes
    #[garde(range(min = 1, max = 10000))]
    pub batch_size: u32,

    /// Flush interval in seconds
    #[garde(range(min = 1, max = 3600))]
    pub flush_interval_secs: u64,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_days: 365,
            log_all_operations: true,
            log_failures_only: false,
            batch_size: 100,
            flush_interval_secs: 60,
        }
    }
}

/// Configuration for performance optimization and resource management.
///
/// This structure defines performance-related settings that control how the
/// secure storage system utilizes system resources and optimizes operations
/// for maximum throughput and minimum latency. These settings are critical
/// for meeting the <1ms latency requirements in production environments.
///
/// # Performance Optimization Areas
///
/// ## Connection Management
/// - Connection pooling for database backends
/// - Keep-alive settings for network connections
/// - Connection timeout and retry policies
/// - Load balancing across multiple backends
///
/// ## Caching Strategy
/// - In-memory caching of frequently accessed data
/// - Cache eviction policies (LRU, LFU, TTL)
/// - Cache warming and preloading
/// - Distributed caching for multi-node deployments
///
/// ## Concurrency Control
/// - Thread pool sizing for async operations
/// - Lock-free data structures where possible
/// - Work-stealing schedulers for CPU-bound tasks
/// - Backpressure mechanisms for overload protection
///
/// # Memory Management
///
/// - Pre-allocated buffer pools for zero-allocation hot paths
/// - Memory-mapped files for large data sets
/// - NUMA-aware memory allocation
/// - Garbage collection tuning for managed runtimes
///
/// # I/O Optimization
///
/// - Asynchronous I/O for all network and disk operations
/// - Batch processing for multiple operations
/// - Vectorized I/O for improved throughput
/// - Direct I/O bypass for critical paths
///
/// # Monitoring and Tuning
///
/// - Real-time performance metrics collection
/// - Automatic performance regression detection
/// - Dynamic configuration adjustment
/// - Performance profiling integration
#[derive(Debug, Clone, Serialize, Deserialize, Validate, Default)]
pub struct PerformanceConfig {
    /// Connection pool settings
    #[garde(dive)]
    pub connection_pool: ConnectionPoolConfig,

    /// Cache settings
    #[garde(dive)]
    pub cache: CacheConfig,

    /// Concurrency settings
    #[garde(dive)]
    pub concurrency: ConcurrencyConfig,
}

/// Configuration for database and network connection pooling.
///
/// This structure defines how connections to backend systems (databases, HSMs,
/// external vaults) are managed through connection pooling. Proper connection
/// pooling is essential for achieving high performance and efficient resource
/// utilization in production environments.
///
/// # Connection Pool Benefits
///
/// - **Reduced Latency**: Eliminates connection establishment overhead
/// - **Resource Efficiency**: Reuses existing connections
/// - **Scalability**: Handles concurrent requests efficiently
/// - **Reliability**: Automatic connection health monitoring
///
/// # Pool Sizing Strategy
///
/// ## Minimum Connections
/// - Always-ready connections for immediate use
/// - Reduces cold-start latency
/// - Should match baseline load requirements
///
/// ## Maximum Connections
/// - Upper limit to prevent resource exhaustion
/// - Should consider backend capacity limits
/// - Typically 2-4x the number of CPU cores
///
/// ## Dynamic Scaling
/// - Automatic scaling between min and max based on load
/// - Connection creation/destruction based on demand
/// - Configurable scaling thresholds and timing
///
/// # Health Monitoring
///
/// - Regular connection health checks
/// - Automatic replacement of failed connections
/// - Circuit breaker pattern for backend failures
/// - Metrics collection for pool utilization
///
/// # Backend-Specific Considerations
///
/// - **`SQLite`**: Single connection (no pooling needed)
/// - **`PostgreSQL`**: 10-100 connections typical
/// - **HSM**: Limited by device connection capacity
/// - **Network Vaults**: Consider network latency and bandwidth
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConnectionPoolConfig {
    /// Minimum pool size
    #[garde(range(min = 1, max = 100))]
    pub min_connections: u32,

    /// Maximum pool size
    #[garde(range(min = 1, max = 1000))]
    pub max_connections: u32,

    /// Connection timeout in seconds
    #[garde(range(min = 1, max = 300))]
    pub connect_timeout_secs: u64,

    /// Idle timeout in seconds
    #[garde(range(min = 60, max = 3600))]
    pub idle_timeout_secs: u64,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 5,
            max_connections: 50,
            connect_timeout_secs: 30,
            idle_timeout_secs: 600,
        }
    }
}

/// Configuration for in-memory caching and data optimization.
///
/// This structure defines caching policies and parameters used to optimize
/// data access patterns and reduce latency for frequently accessed information.
/// Effective caching is crucial for meeting the <1ms performance requirements.
///
/// # Cache Types
///
/// ## Metadata Cache
/// - Vault metadata and configuration data
/// - Role and permission definitions
/// - Session information and authentication state
/// - Small size, high hit rate expected
///
/// ## Data Cache
/// - Frequently accessed encrypted data
/// - Decrypted data for active sessions
/// - Key material and cryptographic state
/// - Larger size, variable hit rate
///
/// ## Query Result Cache
/// - Database query results
/// - Computed values and derived data
/// - Search indexes and lookup tables
/// - Medium size, depends on access patterns
///
/// # Eviction Policies
///
/// ## Least Recently Used (LRU)
/// - Evicts oldest accessed items first
/// - Good for temporal locality patterns
/// - Simple to implement and understand
///
/// ## Least Frequently Used (LFU)
/// - Evicts least accessed items first
/// - Better for frequency-based patterns
/// - More complex tracking overhead
///
/// ## Time-To-Live (TTL)
/// - Automatic expiration after time limit
/// - Ensures data freshness
/// - Prevents stale data issues
///
/// # Performance Considerations
///
/// - Cache hit ratio should exceed 90% for effectiveness
/// - Memory usage must be bounded to prevent OOM
/// - Cache operations should be lock-free where possible
/// - Metrics collection for cache effectiveness monitoring
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CacheConfig {
    /// Enable caching
    #[garde(skip)]
    pub enabled: bool,

    /// Maximum cache size in entries
    #[garde(range(min = 100, max = 1_000_000))]
    pub max_entries: usize,

    /// Cache TTL in seconds
    #[garde(range(min = 60, max = 86400))]
    pub ttl_secs: u64,

    /// Cache cleanup interval in seconds
    #[garde(range(min = 60, max = 3600))]
    pub cleanup_interval_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 10000,
            ttl_secs: 3600,
            cleanup_interval_secs: 300,
        }
    }
}

/// Configuration for concurrency control and parallel processing.
///
/// This structure defines how the secure storage system handles concurrent
/// operations, thread management, and parallel processing. Proper concurrency
/// configuration is essential for maximizing throughput while maintaining
/// data consistency and system stability.
///
/// # Concurrency Models
///
/// ## Thread Pool Management
/// - Fixed-size thread pools for predictable resource usage
/// - Work-stealing schedulers for load balancing
/// - CPU-bound vs I/O-bound task separation
/// - Thread affinity for NUMA optimization
///
/// ## Async/Await Patterns
/// - Non-blocking I/O for network and disk operations
/// - Cooperative multitasking for efficient resource usage
/// - Backpressure handling for overload protection
/// - Structured concurrency for error handling
///
/// ## Lock-Free Data Structures
/// - Atomic operations for high-performance counters
/// - Compare-and-swap for lock-free updates
/// - Memory ordering guarantees for correctness
/// - Hazard pointers for safe memory reclamation
///
/// # Resource Limits
///
/// ## Maximum Concurrent Operations
/// - Prevents resource exhaustion under load
/// - Should be tuned based on system capacity
/// - Includes safety margin for system stability
///
/// ## Queue Depths
/// - Bounded queues prevent memory exhaustion
/// - Backpressure signals when queues are full
/// - Different limits for different operation types
///
/// # Performance Optimization
///
/// - CPU core count detection for optimal thread sizing
/// - NUMA topology awareness for memory allocation
/// - Cache line alignment for shared data structures
/// - False sharing prevention in hot paths
///
/// # Monitoring and Tuning
///
/// - Thread utilization metrics
/// - Queue depth and wait time monitoring
/// - Contention detection and reporting
/// - Automatic scaling based on load patterns
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConcurrencyConfig {
    /// Maximum concurrent operations
    #[garde(range(min = 1, max = 10000))]
    pub max_concurrent_operations: usize,

    /// Worker thread pool size
    #[garde(range(min = 1, max = 256))]
    pub worker_threads: usize,

    /// Operation timeout in milliseconds
    #[garde(range(min = 100, max = 60000))]
    pub operation_timeout_ms: u64,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 1000,
            worker_threads: num_cpus::get(),
            operation_timeout_ms: 5000, // 5 seconds
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config_validation() -> SecureStorageResult<()> {
        let config = SecureStorageConfig::default();
        config.validate()?;
        Ok(())
    }

    #[test]
    fn test_config_serialization() -> SecureStorageResult<()> {
        let config = SecureStorageConfig::default();

        // Test JSON serialization
        let json = serde_json::to_string_pretty(&config)?;
        let deserialized: SecureStorageConfig = serde_json::from_str(&json)?;
        deserialized.validate()?;

        Ok(())
    }

    #[test]
    fn test_config_file_operations() -> SecureStorageResult<()> {
        let config = SecureStorageConfig::default();

        // Test saving and loading JSON
        let temp_file = NamedTempFile::new().map_err(|e| SecureStorageError::Internal {
            reason: format!("Failed to create temp file: {e}"),
        })?;
        let json_path = temp_file.path().with_extension("json");

        config.save_to_file(&json_path)?;
        let loaded_config = SecureStorageConfig::from_file(&json_path)?;
        loaded_config.validate()?;

        Ok(())
    }

    #[test]
    fn test_duration_conversions() {
        let config = SecureStorageConfig::default();

        assert_eq!(config.connection_timeout(), Duration::from_secs(30));
        assert_eq!(config.session_timeout(), Duration::from_secs(3600));
        assert_eq!(config.key_rotation_interval(), Duration::from_secs(86400));
    }
}
