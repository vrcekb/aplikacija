//! Error types for secure storage operations

use thiserror::Error;

/// Result type for secure storage operations
pub type SecureStorageResult<T> = Result<T, SecureStorageError>;

/// Main error type for secure storage operations
#[derive(Error, Debug, Clone)]
/// TODO: Add documentation
pub enum SecureStorageError {
    /// Encryption/decryption errors
    #[error("Encryption error: {reason}")]
    Encryption {
        /// Error description
        reason: String,
    },

    /// Decryption failed
    #[error("Decryption failed: {reason}")]
    Decryption {
        /// Error description
        reason: String,
    },

    /// Key management errors
    #[error("Key error: {operation} failed - {reason}")]
    Key {
        /// Operation that failed
        operation: String,
        /// Error description
        reason: String,
    },

    /// Vault operation errors
    #[error("Vault error: {operation} failed - {reason}")]
    Vault {
        /// Operation that failed
        operation: String,
        /// Error description
        reason: String,
    },

    /// Access control violations
    #[error("Access denied: {reason}")]
    AccessDenied {
        /// Denial reason
        reason: String,
    },

    /// Authentication failures
    #[error("Authentication failed: {reason}")]
    Authentication {
        /// Authentication failure reason
        reason: String,
    },

    /// Authorization failures
    #[error("Authorization failed: {reason}")]
    Authorization {
        /// Authorization failure reason
        reason: String,
    },

    /// Configuration errors
    #[error("Configuration error: {field} - {reason}")]
    Configuration {
        /// Configuration field that failed
        field: String,
        /// Error description
        reason: String,
    },

    /// Database errors
    #[error("Database error: {operation} failed - {reason}")]
    Database {
        /// Database operation that failed
        operation: String,
        /// Error description
        reason: String,
    },

    /// Network errors for remote vaults
    #[error("Network error: {reason}")]
    Network {
        /// Network error description
        reason: String,
    },

    /// Timeout errors
    #[error("Operation timed out: {operation} after {duration_ms}ms")]
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Timeout duration in milliseconds
        duration_ms: u64,
    },

    /// Invalid input data
    #[error("Invalid input: {field} - {reason}")]
    InvalidInput {
        /// Input field that is invalid
        field: String,
        /// Validation error description
        reason: String,
    },

    /// Resource not found
    #[error("Not found: {resource} '{identifier}'")]
    NotFound {
        /// Resource type
        resource: String,
        /// Resource identifier
        identifier: String,
    },

    /// Resource already exists
    #[error("Already exists: {resource} '{identifier}'")]
    AlreadyExists {
        /// Resource type
        resource: String,
        /// Resource identifier
        identifier: String,
    },

    /// Rate limiting
    #[error("Rate limit exceeded: {limit} requests per {window}")]
    RateLimit {
        /// Rate limit threshold
        limit: u32,
        /// Time window description
        window: String,
    },

    /// HSM errors
    #[error("HSM error: {operation} failed - {reason}")]
    Hsm {
        /// HSM operation that failed
        operation: String,
        /// Error description
        reason: String,
    },

    /// Audit log errors
    #[error("Audit error: {reason}")]
    Audit {
        /// Audit error description
        reason: String,
    },

    /// Memory protection errors
    #[error("Memory protection error: {operation} failed - {reason}")]
    MemoryProtection {
        /// Memory protection operation that failed
        operation: String,
        /// Memory protection error description
        reason: String,
    },

    /// Serialization errors
    #[error("Serialization error: {reason}")]
    Serialization {
        /// Serialization error description
        reason: String,
    },

    /// Internal system errors
    #[error("Internal error: {reason}")]
    Internal {
        /// Internal error description
        reason: String,
    },

    /// Insufficient resources (memory, storage, etc.)
    #[error("Insufficient resources: {resource} - {reason}")]
    InsufficientResources {
        /// Resource type that is insufficient
        resource: String,
        /// Resource shortage description
        reason: String,
    },
}

/// Critical errors that require immediate attention
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
/// TODO: Add documentation
pub enum CriticalSecurityError {
    /// Key compromise detected
    #[error("Key compromise detected: code {code:04}")]
    KeyCompromise {
        /// Security incident code
        code: u16,
    },

    /// Unauthorized access attempt
    #[error("Unauthorized access: code {code:04}")]
    UnauthorizedAccess {
        /// Security incident code
        code: u16,
    },

    /// Data integrity violation
    #[error("Data integrity violation: code {code:04}")]
    IntegrityViolation {
        /// Security incident code
        code: u16,
    },

    /// System compromise
    #[error("System compromise: code {code:04}")]
    SystemCompromise {
        /// Security incident code
        code: u16,
    },
}

/// Encryption-specific errors
#[derive(Error, Debug, Clone)]
/// TODO: Add documentation
pub enum EncryptionError {
    /// Key generation failed
    #[error("Key generation failed: {reason}")]
    KeyGeneration {
        /// Key generation failure reason
        reason: String,
    },

    /// Invalid key format
    #[error("Invalid key format: {reason}")]
    InvalidKey {
        /// Key validation error description
        reason: String,
    },

    /// Encryption algorithm not supported
    #[error("Unsupported algorithm: {algorithm}")]
    UnsupportedAlgorithm {
        /// Algorithm name that is not supported
        algorithm: String,
    },

    /// Cipher operation failed
    #[error("Cipher operation failed: {reason}")]
    CipherOperation {
        /// Cipher operation failure reason
        reason: String,
    },

    /// Invalid nonce/IV
    #[error("Invalid nonce: {reason}")]
    InvalidNonce {
        /// Nonce validation error description
        reason: String,
    },

    /// Authentication tag verification failed
    #[error("Authentication failed: tag verification failed")]
    AuthenticationFailed,
}

/// Vault-specific errors
#[derive(Error, Debug, Clone)]
/// TODO: Add documentation
pub enum VaultError {
    /// Connection failed
    #[error("Connection failed: {reason}")]
    Connection {
        /// Connection failure reason
        reason: String,
    },

    /// Authentication with vault failed
    #[error("Vault authentication failed: {reason}")]
    Authentication {
        /// Authentication failure reason
        reason: String,
    },

    /// Secret not found
    #[error("Secret not found: {path}")]
    SecretNotFound {
        /// Secret path that was not found
        path: String,
    },

    /// Permission denied
    #[error("Permission denied: {operation} on {path}")]
    PermissionDenied {
        /// Operation that was denied
        operation: String,
        /// Path where permission was denied
        path: String,
    },

    /// Vault is sealed
    #[error("Vault is sealed")]
    VaultSealed,

    /// Invalid vault response
    #[error("Invalid response: {reason}")]
    InvalidResponse {
        /// Response validation error description
        reason: String,
    },
}

/// Access control errors
#[derive(Error, Debug, Clone)]
/// TODO: Add documentation
pub enum AccessControlError {
    /// Invalid session
    #[error("Invalid session: {reason}")]
    InvalidSession {
        /// Session validation error description
        reason: String,
    },

    /// Session expired
    #[error("Session expired: {session_id}")]
    SessionExpired {
        /// Expired session identifier
        session_id: String,
    },

    /// Insufficient permissions
    #[error("Insufficient permissions: required {required}, have {current}")]
    InsufficientPermissions {
        /// Required permission level
        required: String,
        /// Current permission level
        current: String,
    },

    /// Role not found
    #[error("Role not found: {role_id}")]
    RoleNotFound {
        /// Role identifier that was not found
        role_id: String,
    },

    /// Policy violation
    #[error("Policy violation: {policy} - {reason}")]
    PolicyViolation {
        /// Policy that was violated
        policy: String,
        /// Violation description
        reason: String,
    },
}

// Implement From conversions for common error types
impl From<EncryptionError> for SecureStorageError {
    fn from(err: EncryptionError) -> Self {
        match err {
            EncryptionError::KeyGeneration { reason } => Self::Key {
                operation: "generation".to_string(),
                reason,
            },
            EncryptionError::InvalidKey { reason } => Self::Key {
                operation: "validation".to_string(),
                reason,
            },
            EncryptionError::UnsupportedAlgorithm { algorithm } => Self::Configuration {
                field: "encryption_algorithm".to_string(),
                reason: format!("Unsupported algorithm: {algorithm}"),
            },
            EncryptionError::CipherOperation { reason }
            | EncryptionError::InvalidNonce { reason } => Self::Encryption { reason },
            EncryptionError::AuthenticationFailed => Self::Decryption {
                reason: "Authentication tag verification failed".to_string(),
            },
        }
    }
}

impl From<VaultError> for SecureStorageError {
    fn from(err: VaultError) -> Self {
        match err {
            VaultError::Connection { reason } => Self::Network { reason },
            VaultError::Authentication { reason } => Self::Authentication { reason },
            VaultError::SecretNotFound { path } => Self::NotFound {
                resource: "secret".to_string(),
                identifier: path,
            },
            VaultError::PermissionDenied { operation, path } => Self::AccessDenied {
                reason: format!("Permission denied: {operation} on {path}"),
            },
            VaultError::VaultSealed => Self::Vault {
                operation: "access".to_string(),
                reason: "Vault is sealed".to_string(),
            },
            VaultError::InvalidResponse { reason } => Self::Vault {
                operation: "response_parsing".to_string(),
                reason,
            },
        }
    }
}

impl From<AccessControlError> for SecureStorageError {
    fn from(err: AccessControlError) -> Self {
        match err {
            AccessControlError::InvalidSession { reason } => Self::Authentication { reason },
            AccessControlError::SessionExpired { session_id } => Self::Authentication {
                reason: format!("Session expired: {session_id}"),
            },
            AccessControlError::InsufficientPermissions { required, current } => {
                Self::Authorization {
                    reason: format!(
                        "Insufficient permissions: required {required}, have {current}"
                    ),
                }
            }
            AccessControlError::RoleNotFound { role_id } => Self::Configuration {
                field: "role".to_string(),
                reason: format!("Role not found: {role_id}"),
            },
            AccessControlError::PolicyViolation { policy, reason } => Self::Authorization {
                reason: format!("Policy violation: {policy} - {reason}"),
            },
        }
    }
}

// Temporarily disabled due to dependency conflict
// impl From<sqlx::Error> for SecureStorageError {
//     fn from(err: sqlx::Error) -> Self {
//         Self::Database {
//             operation: "sql_operation".to_string(),
//             reason: err.to_string(),
//         }
//     }
// }

impl From<rusqlite::Error> for SecureStorageError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Database {
            operation: "sqlite_operation".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for SecureStorageError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            reason: err.to_string(),
        }
    }
}

#[cfg(feature = "vault")]
impl From<vaultrs::error::ClientError> for SecureStorageError {
    fn from(err: vaultrs::error::ClientError) -> Self {
        Self::Vault {
            operation: "vault_client".to_string(),
            reason: err.to_string(),
        }
    }
}

#[cfg(feature = "hsm")]
impl From<cryptoki::error::Error> for SecureStorageError {
    fn from(err: cryptoki::error::Error) -> Self {
        Self::Hsm {
            operation: "pkcs11".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<crate::fips::FipsError> for SecureStorageError {
    fn from(err: crate::fips::FipsError) -> Self {
        Self::Internal {
            reason: err.to_string(),
        }
    }
}

/// Helper trait for creating specific error types
pub trait SecureStorageErrorExt {
    /// Create an encryption error
    fn encryption_error(reason: impl Into<String>) -> SecureStorageError;

    /// Create a decryption error
    fn decryption_error(reason: impl Into<String>) -> SecureStorageError;

    /// Create a key error
    fn key_error(operation: impl Into<String>, reason: impl Into<String>) -> SecureStorageError;

    /// Create a vault error
    fn vault_error(operation: impl Into<String>, reason: impl Into<String>) -> SecureStorageError;

    /// Create an access denied error
    fn access_denied(reason: impl Into<String>) -> SecureStorageError;
}

impl SecureStorageErrorExt for SecureStorageError {
    fn encryption_error(reason: impl Into<String>) -> SecureStorageError {
        Self::Encryption {
            reason: reason.into(),
        }
    }

    fn decryption_error(reason: impl Into<String>) -> SecureStorageError {
        Self::Decryption {
            reason: reason.into(),
        }
    }

    fn key_error(operation: impl Into<String>, reason: impl Into<String>) -> SecureStorageError {
        Self::Key {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    fn vault_error(operation: impl Into<String>, reason: impl Into<String>) -> SecureStorageError {
        Self::Vault {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    fn access_denied(reason: impl Into<String>) -> SecureStorageError {
        Self::AccessDenied {
            reason: reason.into(),
        }
    }
}
