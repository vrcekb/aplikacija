//! `TallyIO` Core Error System
//!
//! Production-ready error handling with specific error types for each domain.
//! Follows NAVODILA.md standards with zero-panic guarantees.

use std::time::Duration;
use thiserror::Error;

/// Core result type for all operations
pub type CoreResult<T> = Result<T, CoreError>;

/// Critical errors that require immediate attention (Copy for performance)
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CriticalError {
    /// Memory allocation failed
    #[error("Out of memory: code {0}")]
    OutOfMemory(u16),
    /// CPU affinity setting failed
    #[error("CPU affinity failed: code {0}")]
    CpuAffinityFailed(u16),
    /// Critical resource unavailable
    #[error("Resource unavailable: code {0}")]
    ResourceUnavailable(u16),
    /// System limit exceeded
    #[error("System limit exceeded: code {0}")]
    SystemLimitExceeded(u16),
    /// Invalid size parameter
    #[error("Invalid size: code {0}")]
    InvalidSize(u16),
    /// System error
    #[error("System error: code {0}")]
    SystemError(u16),
}

/// Main error type for core operations
#[derive(Error, Debug)]
pub enum CoreError {
    /// Critical system error (requires immediate action)
    #[error("Critical error: {0:?}")]
    Critical(#[from] CriticalError),

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration {
        /// Error message
        message: String,
    },

    /// Validation errors
    #[error("Validation failed for field '{field}': {reason}")]
    Validation {
        /// Field that failed validation
        field: String,
        /// Reason for validation failure
        reason: String,
    },

    /// Engine execution errors
    #[error("Engine execution failed: {operation} - {reason}")]
    Engine {
        /// Operation that failed
        operation: String,
        /// Reason for failure
        reason: String,
    },

    /// State management errors
    #[error("State error: {state_type} - {message}")]
    State {
        /// Type of state that failed
        state_type: String,
        /// Error message
        message: String,
    },

    /// Mempool monitoring errors
    #[error("Mempool error: {source} - {details}")]
    Mempool {
        /// Source error
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
        /// Additional details
        details: String,
    },

    /// Performance optimization errors
    #[error("Optimization failed: {optimization_type} - {reason}")]
    Optimization {
        /// Type of optimization that failed
        optimization_type: String,
        /// Reason for failure
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

    /// Resource errors
    #[error("Resource error: {resource} - {message}")]
    Resource {
        /// Resource that failed
        resource: String,
        /// Error message
        message: String,
    },

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Generic internal error (use sparingly)
    #[error("Internal error: {message}")]
    Internal {
        /// Error message
        message: String,
    },

    /// System error
    #[error("System error: {message}")]
    SystemError {
        /// Error message
        message: String,
    },

    /// Invalid size error
    #[error("Invalid size: {message}")]
    InvalidSize {
        /// Error message
        message: String,
    },

    /// Out of memory error
    #[error("Out of memory: {message}")]
    OutOfMemory {
        /// Error message
        message: String,
    },

    /// Invalid configuration error
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration {
        /// Error message
        message: String,
    },
}

/// Engine-specific errors
#[derive(Error, Debug)]
pub enum EngineError {
    /// Executor not initialized
    #[error("Executor not initialized")]
    ExecutorNotInitialized,

    /// Scheduler queue full
    #[error("Scheduler queue full: {capacity} items")]
    SchedulerQueueFull {
        /// Queue capacity
        capacity: usize,
    },

    /// Worker thread failed
    #[error("Worker thread failed: {worker_id} - {reason}")]
    WorkerFailed {
        /// Worker ID
        worker_id: u32,
        /// Failure reason
        reason: String,
    },

    /// Strategy execution failed
    #[error("Strategy execution failed: {strategy} - {error}")]
    StrategyFailed {
        /// Strategy name
        strategy: String,
        /// Error details
        error: String,
    },

    /// Circuit breaker open
    #[error("Circuit breaker open for: {component}")]
    CircuitBreakerOpen {
        /// Component name
        component: String,
    },
}

/// State management errors
#[derive(Error, Debug)]
pub enum StateError {
    /// State not initialized
    #[error("State not initialized: {state_type}")]
    NotInitialized {
        /// State type
        state_type: String,
    },

    /// State corruption detected
    #[error("State corruption detected: {details}")]
    Corruption {
        /// Corruption details
        details: String,
    },

    /// Synchronization failed
    #[error("Synchronization failed: {reason}")]
    SyncFailed {
        /// Failure reason
        reason: String,
    },

    /// Invalid state transition
    #[error("State transition invalid: {from} -> {to}")]
    InvalidTransition {
        /// Source state
        from: String,
        /// Target state
        to: String,
    },

    /// Lock acquisition timeout
    #[error("Lock acquisition timeout: {resource}")]
    LockTimeout {
        /// Resource name
        resource: String,
    },
}

/// Mempool monitoring errors
#[derive(Error, Debug)]
pub enum MempoolError {
    /// Connection failed
    #[error("Connection failed to: {endpoint}")]
    ConnectionFailed {
        /// Endpoint URL
        endpoint: String,
    },

    /// Transaction parsing failed
    #[error("Transaction parsing failed: {tx_hash}")]
    TransactionParsingFailed {
        /// Transaction hash
        tx_hash: String,
    },

    /// Filter error
    #[error("Filter error: {filter_type} - {message}")]
    FilterError {
        /// Filter type
        filter_type: String,
        /// Error message
        message: String,
    },

    /// Analysis timeout
    #[error("Analysis timeout for transaction: {tx_hash}")]
    AnalysisTimeout {
        /// Transaction hash
        tx_hash: String,
    },

    /// Watcher not running
    #[error("Watcher not running")]
    WatcherNotRunning,
}

/// Optimization errors
#[derive(Error, Debug)]
pub enum OptimizationError {
    /// CPU affinity failed
    #[error("CPU affinity failed: core {core_id}")]
    CpuAffinityFailed {
        /// Core ID
        core_id: u32,
    },

    /// Memory pool exhausted
    #[error("Memory pool exhausted: {pool_type}")]
    MemoryPoolExhausted {
        /// Pool type
        pool_type: String,
    },

    /// SIMD not supported
    #[error("SIMD not supported: {instruction_set}")]
    SimdNotSupported {
        /// Instruction set
        instruction_set: String,
    },

    /// Cache optimization failed
    #[error("Cache optimization failed: {cache_level}")]
    CacheOptimizationFailed {
        /// Cache level
        cache_level: String,
    },
}

// Convenience constructors for common errors
impl CoreError {
    /// Create configuration error
    pub fn config(message: impl Into<String>) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create validation error
    pub fn validation(field: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Validation {
            field: field.into(),
            reason: reason.into(),
        }
    }

    /// Create engine error
    pub fn engine(operation: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Engine {
            operation: operation.into(),
            reason: reason.into(),
        }
    }

    /// Create state error
    pub fn state(state_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::State {
            state_type: state_type.into(),
            message: message.into(),
        }
    }

    /// Create mempool error
    pub fn mempool(
        source: impl std::error::Error + Send + Sync + 'static,
        details: impl Into<String>,
    ) -> Self {
        Self::Mempool {
            source: Box::new(source),
            details: details.into(),
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

    /// Create system error
    pub fn system_error(message: impl Into<String>) -> Self {
        Self::SystemError {
            message: message.into(),
        }
    }

    /// Create invalid size error
    pub fn invalid_size(message: impl Into<String>) -> Self {
        Self::InvalidSize {
            message: message.into(),
        }
    }

    /// Create out of memory error
    pub fn out_of_memory(message: impl Into<String>) -> Self {
        Self::OutOfMemory {
            message: message.into(),
        }
    }

    /// Create invalid configuration error
    pub fn invalid_configuration(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }
}

// Convert from domain-specific errors
impl From<EngineError> for CoreError {
    fn from(err: EngineError) -> Self {
        Self::engine("engine_operation", err.to_string())
    }
}

impl From<StateError> for CoreError {
    fn from(err: StateError) -> Self {
        Self::state("state_operation", err.to_string())
    }
}

impl From<MempoolError> for CoreError {
    fn from(err: MempoolError) -> Self {
        Self::mempool(err, "mempool_operation")
    }
}

impl From<OptimizationError> for CoreError {
    fn from(err: OptimizationError) -> Self {
        Self::Optimization {
            optimization_type: "optimization_operation".to_string(),
            reason: err.to_string(),
        }
    }
}

// Convert from module-specific errors
impl From<crate::engine::EngineError> for CoreError {
    fn from(err: crate::engine::EngineError) -> Self {
        Self::engine("engine_module", err.to_string())
    }
}

impl From<crate::state::StateError> for CoreError {
    fn from(err: crate::state::StateError) -> Self {
        Self::state("state_module", err.to_string())
    }
}

impl From<crate::mempool::MempoolError> for CoreError {
    fn from(err: crate::mempool::MempoolError) -> Self {
        Self::mempool(err, "mempool_module")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = CoreError::config("Invalid port");
        assert!(matches!(err, CoreError::Configuration { .. }));

        let err = CoreError::validation("email", "Invalid format");
        assert!(matches!(err, CoreError::Validation { .. }));
    }

    #[test]
    fn test_error_conversion() {
        let engine_err = EngineError::ExecutorNotInitialized;
        let core_err: CoreError = engine_err.into();
        assert!(matches!(core_err, CoreError::Engine { .. }));
    }

    #[test]
    fn test_critical_error_copy() {
        let err = CriticalError::OutOfMemory(1001);
        let err_copy = err; // Should compile (Copy trait)
        assert_eq!(err, err_copy);
    }
}
