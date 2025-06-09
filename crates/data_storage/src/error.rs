//! `TallyIO` Data Storage Error System
//!
//! Production-ready error handling for data storage operations.
//! Follows NAVODILA.md standards with zero-panic guarantees.

use std::time::Duration;
use thiserror::Error;

/// Data storage result type for all operations
pub type DataStorageResult<T> = Result<T, DataStorageError>;

/// Critical errors that require immediate attention
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CriticalError {
    /// Storage corruption detected
    #[error("Storage corruption detected: code {0}")]
    StorageCorruption(u16),

    /// Memory allocation failure
    #[error("Memory allocation failed: code {0}")]
    OutOfMemory(u16),

    /// Connection pool exhausted
    #[error("Connection pool exhausted: code {0}")]
    ConnectionPoolExhausted(u16),

    /// Hot storage failure
    #[error("Hot storage failure: code {0}")]
    HotStorageFailure(u16),
}

/// Main error type for data storage operations
#[derive(Error, Debug)]
pub enum DataStorageError {
    /// Critical system error (requires immediate action)
    #[error("Critical error: {0:?}")]
    Critical(#[from] CriticalError),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration {
        /// Error message
        message: String,
    },

    /// Database operation errors
    #[error("Database operation failed: {operation} - {reason}")]
    Database {
        /// Operation that failed
        operation: String,
        /// Reason for failure
        reason: String,
    },

    /// Cache operation errors
    #[error("Cache operation failed: {operation} - {reason}")]
    Cache {
        /// Operation that failed
        operation: String,
        /// Reason for failure
        reason: String,
    },

    /// Pipeline operation errors
    #[error("Pipeline operation failed: {stage} - {reason}")]
    Pipeline {
        /// Pipeline stage that failed
        stage: String,
        /// Reason for failure
        reason: String,
    },

    /// Stream processing errors
    #[error("Stream processing failed: {stream} - {reason}")]
    Stream {
        /// Stream that failed
        stream: String,
        /// Reason for failure
        reason: String,
    },

    /// Indexer errors
    #[error("Indexer operation failed: {indexer} - {reason}")]
    Indexer {
        /// Indexer that failed
        indexer: String,
        /// Reason for failure
        reason: String,
    },

    /// Validation errors
    #[error("Validation failed for field '{field}': {reason}")]
    Validation {
        /// Field that failed validation
        field: String,
        /// Reason for validation failure
        reason: String,
    },

    /// Timeout errors
    #[error("Operation timed out after {duration:?}: {operation}")]
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Duration before timeout
        duration: Duration,
    },

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// `PostgreSQL` errors
    #[cfg(feature = "warm-storage")]
    #[error("PostgreSQL error: {0}")]
    Postgres(#[from] tokio_postgres::Error),

    /// Redis errors
    #[cfg(feature = "cache")]
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    /// Generic internal error (use sparingly)
    #[error("Internal error: {message}")]
    Internal {
        /// Error message
        message: String,
    },
}

impl DataStorageError {
    /// Create configuration error
    pub fn configuration(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create database error
    pub fn database(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Database {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create cache error
    pub fn cache(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Cache {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create pipeline error
    pub fn pipeline(stage: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Pipeline {
            stage: stage.into(),
            reason: reason.into(),
        }
    }

    /// Create stream error
    pub fn stream(stream: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Stream {
            stream: stream.into(),
            reason: reason.into(),
        }
    }

    /// Create indexer error
    pub fn indexer(indexer: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Indexer {
            indexer: indexer.into(),
            reason: reason.into(),
        }
    }

    /// Create validation error
    pub fn validation(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Create timeout error
    pub fn timeout(operation: impl Into<String>, duration: Duration) -> Self {
        Self::Timeout {
            operation: operation.into(),
            duration,
        }
    }

    /// Create internal error (use sparingly)
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Create connection error (alias for database error)
    pub fn connection(message: impl Into<String>) -> Self {
        Self::database("connection", message)
    }

    /// Create serialization error
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::internal(format!("Serialization error: {}", message.into()))
    }

    /// Create migration error
    pub fn migration(message: impl Into<String>) -> Self {
        Self::database("migration", message)
    }

    /// Check if error is critical
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        matches!(self, Self::Critical(_))
    }

    /// Check if error is retryable
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Critical(_) => false,
            Self::Configuration { .. } => false,
            Self::Validation { .. } => false,
            Self::Database { .. } => true,
            Self::Cache { .. } => true,
            Self::Pipeline { .. } => true,
            Self::Stream { .. } => true,
            Self::Indexer { .. } => true,
            Self::Timeout { .. } => true,
            Self::Serialization(_) => false,
            Self::Io(_) => true,
            #[cfg(feature = "warm-storage")]
            Self::Postgres(_) => true,
            #[cfg(feature = "cache")]
            Self::Redis(_) => true,
            Self::Internal { .. } => false,
        }
    }
}

/// Storage-specific error codes for monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Hot storage errors (1000-1099)
    HotStorageRead = 1001,
    HotStorageWrite = 1002,
    HotStorageCorruption = 1003,

    /// Warm storage errors (1100-1199)
    WarmStorageConnection = 1101,
    WarmStorageQuery = 1102,
    WarmStorageTransaction = 1103,

    /// Cache errors (1200-1299)
    CacheConnection = 1201,
    CacheRead = 1202,
    CacheWrite = 1203,

    /// Pipeline errors (1300-1399)
    PipelineIngestion = 1301,
    PipelineTransformation = 1302,
    PipelineValidation = 1303,

    /// Stream errors (1400-1499)
    StreamBuffer = 1401,
    StreamProcessor = 1402,
    StreamAggregator = 1403,

    /// Indexer errors (1500-1599)
    IndexerBlock = 1501,
    IndexerEvent = 1502,
    IndexerTransaction = 1503,
}

impl ErrorCode {
    /// Get error code as u16
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }

    /// Get error category
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match *self as u16 {
            1000..=1099 => "hot_storage",
            1100..=1199 => "warm_storage",
            1200..=1299 => "cache",
            1300..=1399 => "pipeline",
            1400..=1499 => "stream",
            1500..=1599 => "indexer",
            _ => "unknown",
        }
    }
}
