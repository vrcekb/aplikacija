//! `TallyIO` Network Error System
//!
//! Production-ready error handling for network operations with specific error types.
//! Follows NAVODILA.md standards with zero-panic guarantees.

use std::time::Duration;
use thiserror::Error;

/// Network result type for all operations
pub type NetworkResult<T> = Result<T, NetworkError>;

/// Critical network errors that require immediate attention (Copy for performance)
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum CriticalNetworkError {
    /// Connection pool exhausted
    #[error("Connection pool exhausted: code {0}")]
    ConnectionPoolExhausted(u16),
    /// DNS resolution failed
    #[error("DNS resolution failed: code {0}")]
    DnsResolutionFailed(u16),
    /// TLS handshake failed
    #[error("TLS handshake failed: code {0}")]
    TlsHandshakeFailed(u16),
    /// Network interface unavailable
    #[error("Network interface unavailable: code {0}")]
    NetworkInterfaceUnavailable(u16),
}

/// Main error type for network operations
#[derive(Error, Debug)]
pub enum NetworkError {
    /// Critical network error (requires immediate action)
    #[error("Critical network error: {0:?}")]
    Critical(#[from] CriticalNetworkError),

    /// HTTP client errors
    #[error("HTTP error: {status_code} - {message}")]
    Http {
        /// HTTP status code
        status_code: u16,
        /// Error message
        message: String,
        /// Request URL
        url: Option<String>,
    },

    /// WebSocket errors
    #[error("WebSocket error: {operation} - {reason}")]
    WebSocket {
        /// Operation that failed
        operation: String,
        /// Reason for failure
        reason: String,
        /// Connection ID
        connection_id: Option<String>,
    },

    /// Connection errors
    #[error("Connection failed to {endpoint}: {reason}")]
    Connection {
        /// Endpoint URL
        endpoint: String,
        /// Failure reason
        reason: String,
        /// Retry attempt number
        retry_attempt: Option<u32>,
    },

    /// Timeout errors
    #[error("Operation timed out after {duration:?}: {operation}")]
    Timeout {
        /// Operation that timed out
        operation: String,
        /// Duration before timeout
        duration: Duration,
        /// Endpoint involved
        endpoint: Option<String>,
    },

    /// Load balancer errors
    #[error("Load balancer error: {strategy} - {message}")]
    LoadBalancer {
        /// Load balancing strategy
        strategy: String,
        /// Error message
        message: String,
        /// Available endpoints count
        available_endpoints: u32,
    },

    /// Circuit breaker errors
    #[error("Circuit breaker {state} for {component}")]
    CircuitBreaker {
        /// Circuit breaker state
        state: String,
        /// Component name
        component: String,
        /// Failure count
        failure_count: u32,
    },

    /// Retry policy errors
    #[error("Retry policy exhausted: {attempts} attempts for {operation}")]
    RetryExhausted {
        /// Number of attempts made
        attempts: u32,
        /// Operation that failed
        operation: String,
        /// Last error encountered
        last_error: String,
    },

    /// Configuration errors
    #[error("Configuration error: {field} - {message}")]
    Configuration {
        /// Configuration field
        field: String,
        /// Error message
        message: String,
    },

    /// Validation errors
    #[error("Validation failed for {field}: {reason}")]
    Validation {
        /// Field that failed validation
        field: String,
        /// Reason for validation failure
        reason: String,
    },

    /// TLS/SSL errors
    #[error("TLS error: {operation} - {details}")]
    Tls {
        /// TLS operation
        operation: String,
        /// Error details
        details: String,
        /// Certificate info
        certificate_info: Option<String>,
    },

    /// Serialization/Deserialization errors
    #[error("Serialization error: {format} - {message}")]
    Serialization {
        /// Data format
        format: String,
        /// Error message
        message: String,
    },

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// URL parsing errors
    #[error("URL parsing error: {0}")]
    UrlParse(#[from] url::ParseError),

    /// Generic internal error (use sparingly)
    #[error("Internal network error: {message}")]
    Internal {
        /// Error message
        message: String,
    },
}

/// HTTP-specific errors
#[derive(Error, Debug)]
pub enum HttpError {
    /// Request building failed
    #[error("Request building failed: {reason}")]
    RequestBuildFailed {
        /// Failure reason
        reason: String,
    },

    /// Response parsing failed
    #[error("Response parsing failed: {content_type} - {reason}")]
    ResponseParsingFailed {
        /// Content type
        content_type: String,
        /// Failure reason
        reason: String,
    },

    /// Invalid HTTP method
    #[error("Invalid HTTP method: {method}")]
    InvalidMethod {
        /// HTTP method
        method: String,
    },

    /// Invalid headers
    #[error("Invalid headers: {header_name} - {reason}")]
    InvalidHeaders {
        /// Header name
        header_name: String,
        /// Failure reason
        reason: String,
    },

    /// Compression error
    #[error("Compression error: {algorithm} - {message}")]
    Compression {
        /// Compression algorithm
        algorithm: String,
        /// Error message
        message: String,
    },
}

/// WebSocket-specific errors
#[derive(Error, Debug)]
pub enum WebSocketError {
    /// Connection establishment failed
    #[error("WebSocket connection failed: {url} - {reason}")]
    ConnectionFailed {
        /// WebSocket URL
        url: String,
        /// Failure reason
        reason: String,
    },

    /// Message sending failed
    #[error("Message send failed: {message_type} - {reason}")]
    MessageSendFailed {
        /// Message type
        message_type: String,
        /// Failure reason
        reason: String,
    },

    /// Protocol error
    #[error("WebSocket protocol error: {code} - {description}")]
    ProtocolError {
        /// Error code
        code: u16,
        /// Error description
        description: String,
    },

    /// Handshake failed
    #[error("WebSocket handshake failed: {reason}")]
    HandshakeFailed {
        /// Failure reason
        reason: String,
    },

    /// Connection closed unexpectedly
    #[error("Connection closed: {code} - {reason}")]
    ConnectionClosed {
        /// Close code
        code: u16,
        /// Close reason
        reason: String,
    },
}

/// Load balancer errors
#[derive(Error, Debug)]
pub enum LoadBalancerError {
    /// No healthy endpoints available
    #[error("No healthy endpoints available")]
    NoHealthyEndpoints,

    /// Health check failed
    #[error("Health check failed for {endpoint}: {reason}")]
    HealthCheckFailed {
        /// Endpoint URL
        endpoint: String,
        /// Failure reason
        reason: String,
    },

    /// Invalid strategy
    #[error("Invalid load balancing strategy: {strategy}")]
    InvalidStrategy {
        /// Strategy name
        strategy: String,
    },

    /// Endpoint registration failed
    #[error("Endpoint registration failed: {endpoint} - {reason}")]
    EndpointRegistrationFailed {
        /// Endpoint URL
        endpoint: String,
        /// Failure reason
        reason: String,
    },
}

// Convenience constructors for common errors
impl NetworkError {
    /// Create HTTP error
    pub fn http(status_code: u16, message: impl Into<String>, url: Option<String>) -> Self {
        Self::Http {
            status_code,
            message: message.into(),
            url,
        }
    }

    /// Create WebSocket error
    pub fn websocket(
        operation: impl Into<String>,
        reason: impl Into<String>,
        connection_id: Option<String>,
    ) -> Self {
        Self::WebSocket {
            operation: operation.into(),
            reason: reason.into(),
            connection_id,
        }
    }

    /// Create connection error
    pub fn connection(
        endpoint: impl Into<String>,
        reason: impl Into<String>,
        retry_attempt: Option<u32>,
    ) -> Self {
        Self::Connection {
            endpoint: endpoint.into(),
            reason: reason.into(),
            retry_attempt,
        }
    }

    /// Create timeout error
    pub fn timeout(
        operation: impl Into<String>,
        duration: Duration,
        endpoint: Option<String>,
    ) -> Self {
        Self::Timeout {
            operation: operation.into(),
            duration,
            endpoint,
        }
    }

    /// Create configuration error
    pub fn config(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Configuration {
            field: field.into(),
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

    /// Create internal error (use sparingly)
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

// Convert from domain-specific errors
impl From<HttpError> for NetworkError {
    fn from(err: HttpError) -> Self {
        Self::http(500, err.to_string(), None)
    }
}

impl From<WebSocketError> for NetworkError {
    fn from(err: WebSocketError) -> Self {
        Self::websocket("websocket_operation", err.to_string(), None)
    }
}

impl From<LoadBalancerError> for NetworkError {
    fn from(err: LoadBalancerError) -> Self {
        Self::LoadBalancer {
            strategy: "unknown".to_string(),
            message: err.to_string(),
            available_endpoints: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = NetworkError::http(404, "Not found", Some("https://example.com".to_string()));
        assert!(matches!(err, NetworkError::Http { .. }));

        let err = NetworkError::websocket("connect", "Connection refused", None);
        assert!(matches!(err, NetworkError::WebSocket { .. }));
    }

    #[test]
    fn test_error_conversion() {
        let http_err = HttpError::RequestBuildFailed {
            reason: "Invalid URL".to_string(),
        };
        let network_err: NetworkError = http_err.into();
        assert!(matches!(network_err, NetworkError::Http { .. }));
    }

    #[test]
    fn test_critical_error_copy() {
        let err = CriticalNetworkError::ConnectionPoolExhausted(1001);
        let err_copy = err; // Should compile (Copy trait)
        assert_eq!(err, err_copy);
    }
}
