//! `TallyIO` Network Configuration System
//!
//! Production-ready configuration with validation and environment support.
//! Follows NAVODILA.md standards with type-safe validation.

use crate::error::{NetworkError, NetworkResult};
use crate::types::Endpoint;
use garde::Validate;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Network configuration for `TallyIO`
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct NetworkConfig {
    /// HTTP client configuration
    #[garde(dive)]
    pub http: HttpConfig,

    /// WebSocket configuration
    #[garde(dive)]
    pub websocket: WebSocketConfig,

    /// Load balancer configuration
    #[garde(dive)]
    pub load_balancer: LoadBalancerConfig,

    /// P2P network configuration (optional)
    #[garde(dive)]
    pub p2p: Option<P2PConfig>,

    /// Security configuration
    #[garde(dive)]
    pub security: SecurityConfig,

    /// Metrics configuration
    #[garde(dive)]
    pub metrics: MetricsConfig,
}

/// HTTP client configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HttpConfig {
    /// Maximum number of connections per host
    #[garde(range(min = 1, max = 1000))]
    pub max_connections_per_host: u32,

    /// Connection timeout (seconds)
    #[garde(range(min = 1, max = 300))]
    pub connection_timeout_s: u64,

    /// Request timeout (seconds)
    #[garde(range(min = 1, max = 300))]
    pub request_timeout_s: u64,

    /// Keep-alive timeout (seconds)
    #[garde(range(min = 1, max = 3600))]
    pub keep_alive_timeout_s: u64,

    /// Enable HTTP/2
    #[garde(skip)]
    pub enable_http2: bool,

    /// Enable HTTP/3 (QUIC)
    #[garde(skip)]
    pub enable_http3: bool,

    /// Enable compression
    #[garde(skip)]
    pub enable_compression: bool,

    /// User agent string
    #[garde(length(min = 1, max = 200))]
    pub user_agent: String,

    /// Retry configuration
    #[garde(dive)]
    pub retry: RetryConfig,

    /// Circuit breaker configuration
    #[garde(dive)]
    pub circuit_breaker: CircuitBreakerConfig,
}

/// WebSocket configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct WebSocketConfig {
    /// Connection timeout (seconds)
    #[garde(range(min = 1, max = 300))]
    pub connection_timeout_s: u64,

    /// Ping interval (seconds)
    #[garde(range(min = 1, max = 300))]
    pub ping_interval_s: u64,

    /// Pong timeout (seconds)
    #[garde(range(min = 1, max = 60))]
    pub pong_timeout_s: u64,

    /// Maximum message size (bytes)
    #[garde(range(min = 1024, max = 16_777_216))] // 1KB to 16MB
    pub max_message_size: u32,

    /// Message buffer size
    #[garde(range(min = 10, max = 10000))]
    pub message_buffer_size: u32,

    /// Enable automatic reconnection
    #[garde(skip)]
    pub enable_auto_reconnect: bool,

    /// Reconnection configuration
    #[garde(dive)]
    pub reconnect: ReconnectConfig,

    /// Enable message compression
    #[garde(skip)]
    pub enable_compression: bool,
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LoadBalancerConfig {
    /// Load balancing strategy
    #[garde(skip)]
    pub strategy: LoadBalancingStrategy,

    /// Health check configuration
    #[garde(dive)]
    pub health_check: HealthCheckConfig,

    /// Endpoints
    #[garde(length(min = 1))]
    pub endpoints: Vec<Endpoint>,

    /// Enable sticky sessions
    #[garde(skip)]
    pub enable_sticky_sessions: bool,

    /// Session timeout (seconds)
    #[garde(range(min = 60, max = 86400))] // 1 minute to 1 day
    pub session_timeout_s: u64,
}

/// P2P network configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct P2PConfig {
    /// Listen address
    #[garde(length(min = 1))]
    pub listen_addr: String,

    /// Maximum number of peers
    #[garde(range(min = 1, max = 1000))]
    pub max_peers: u32,

    /// Peer discovery configuration
    #[garde(dive)]
    pub discovery: PeerDiscoveryConfig,

    /// Enable DHT (Distributed Hash Table)
    #[garde(skip)]
    pub enable_dht: bool,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SecurityConfig {
    /// Enable TLS verification
    #[garde(skip)]
    pub enable_tls_verification: bool,

    /// TLS minimum version
    #[garde(skip)]
    pub tls_min_version: TlsVersion,

    /// Enable certificate pinning
    #[garde(skip)]
    pub enable_cert_pinning: bool,

    /// Pinned certificates (PEM format)
    #[garde(skip)]
    pub pinned_certificates: Vec<String>,

    /// Enable request signing
    #[garde(skip)]
    pub enable_request_signing: bool,

    /// Rate limiting configuration
    #[garde(dive)]
    pub rate_limiting: RateLimitingConfig,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MetricsConfig {
    /// Enable metrics collection
    #[garde(skip)]
    pub enabled: bool,

    /// Metrics collection interval (milliseconds)
    #[garde(range(min = 100, max = 60000))]
    pub collection_interval_ms: u64,

    /// Enable detailed connection metrics
    #[garde(skip)]
    pub enable_connection_metrics: bool,

    /// Enable latency histograms
    #[garde(skip)]
    pub enable_latency_histograms: bool,
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    #[garde(range(min = 0, max = 10))]
    pub max_attempts: u32,

    /// Initial retry delay (milliseconds)
    #[garde(range(min = 10, max = 10000))]
    pub initial_delay_ms: u64,

    /// Maximum retry delay (milliseconds)
    #[garde(range(min = 100, max = 60000))]
    pub max_delay_ms: u64,

    /// Backoff multiplier
    #[garde(range(min = 1.0_f64, max = 10.0_f64))]
    pub backoff_multiplier: f64,

    /// Enable jitter
    #[garde(skip)]
    pub enable_jitter: bool,

    /// Retry on specific HTTP status codes
    #[garde(skip)]
    pub retry_on_status_codes: Vec<u16>,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CircuitBreakerConfig {
    /// Failure threshold
    #[garde(range(min = 1, max = 100))]
    pub failure_threshold: u32,

    /// Success threshold for recovery
    #[garde(range(min = 1, max = 100))]
    pub success_threshold: u32,

    /// Timeout duration (seconds)
    #[garde(range(min = 1, max = 300))]
    pub timeout_s: u64,

    /// Half-open max calls
    #[garde(range(min = 1, max = 100))]
    pub half_open_max_calls: u32,
}

/// Reconnection configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ReconnectConfig {
    /// Maximum reconnection attempts
    #[garde(range(min = 1, max = 100))]
    pub max_attempts: u32,

    /// Initial reconnection delay (milliseconds)
    #[garde(range(min = 100, max = 10000))]
    pub initial_delay_ms: u64,

    /// Maximum reconnection delay (milliseconds)
    #[garde(range(min = 1000, max = 300_000))]
    pub max_delay_ms: u64,

    /// Backoff multiplier
    #[garde(range(min = 1.0_f64, max = 5.0_f64))]
    pub backoff_multiplier: f64,
}

/// Health check configuration
#[derive(Debug, Clone, Hash, Serialize, Deserialize, Validate)]
pub struct HealthCheckConfig {
    /// Health check interval (seconds)
    #[garde(range(min = 1, max = 300))]
    pub interval_s: u64,

    /// Health check timeout (seconds)
    #[garde(range(min = 1, max = 60))]
    pub timeout_s: u64,

    /// Failure threshold
    #[garde(range(min = 1, max = 10))]
    pub failure_threshold: u32,

    /// Success threshold
    #[garde(range(min = 1, max = 10))]
    pub success_threshold: u32,

    /// Health check path
    #[garde(length(min = 1, max = 200))]
    pub path: String,
}

/// Peer discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PeerDiscoveryConfig {
    /// Bootstrap nodes
    #[garde(skip)]
    pub bootstrap_nodes: Vec<String>,

    /// Discovery interval (seconds)
    #[garde(range(min = 10, max = 3600))]
    pub discovery_interval_s: u64,

    /// Maximum discovered peers
    #[garde(range(min = 10, max = 1000))]
    pub max_discovered_peers: u32,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct RateLimitingConfig {
    /// Enable rate limiting
    #[garde(skip)]
    pub enabled: bool,

    /// Requests per second limit
    #[garde(range(min = 1, max = 10000))]
    pub requests_per_second: u32,

    /// Burst size
    #[garde(range(min = 1, max = 1000))]
    pub burst_size: u32,

    /// Rate limiting window (seconds)
    #[garde(range(min = 1, max = 3600))]
    pub window_s: u64,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin strategy
    RoundRobin,
    /// Weighted round-robin strategy
    WeightedRoundRobin,
    /// Least connections strategy
    LeastConnections,
    /// Random strategy
    Random,
    /// Consistent hashing strategy
    ConsistentHash,
}

impl Default for LoadBalancingStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// TLS version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlsVersion {
    /// TLS 1.2
    Tls12,
    /// TLS 1.3
    Tls13,
}

impl Default for TlsVersion {
    fn default() -> Self {
        Self::Tls13
    }
}



impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 100,
            connection_timeout_s: 30,
            request_timeout_s: 60,
            keep_alive_timeout_s: 90,
            enable_http2: true,
            enable_http3: false,
            enable_compression: true,
            user_agent: "TallyIO/1.0".to_string(),
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            connection_timeout_s: 30,
            ping_interval_s: 30,
            pong_timeout_s: 10,
            max_message_size: 1_048_576, // 1MB
            message_buffer_size: 1000,
            enable_auto_reconnect: true,
            reconnect: ReconnectConfig::default(),
            enable_compression: true,
        }
    }
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::default(),
            health_check: HealthCheckConfig::default(),
            endpoints: Vec::new(),
            enable_sticky_sessions: false,
            session_timeout_s: 3600,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_tls_verification: true,
            tls_min_version: TlsVersion::default(),
            enable_cert_pinning: false,
            pinned_certificates: Vec::new(),
            enable_request_signing: false,
            rate_limiting: RateLimitingConfig::default(),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_ms: 1000,
            enable_connection_metrics: true,
            enable_latency_histograms: true,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0_f64,
            enable_jitter: true,
            retry_on_status_codes: vec![500, 502, 503, 504],
        }
    }
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_s: 60,
            half_open_max_calls: 10,
        }
    }
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            max_attempts: 10,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0_f64,
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval_s: 5,
            timeout_s: 2,
            failure_threshold: 3,
            success_threshold: 2,
            path: "/health".to_string(),
        }
    }
}

impl Default for RateLimitingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: 100,
            burst_size: 10,
            window_s: 60,
        }
    }
}

impl NetworkConfig {
    /// Validate configuration for production environment
    ///
    /// # Errors
    /// Returns error if configuration is invalid for production use
    pub fn validate_production(&self) -> NetworkResult<()> {
        self.validate(&()).map_err(|e| {
            NetworkError::validation("network_config", format!("Validation failed: {e}"))
        })?;

        // Additional production-specific validations
        if !self.security.enable_tls_verification {
            return Err(NetworkError::config(
                "security.enable_tls_verification",
                "TLS verification must be enabled in production",
            ));
        }

        if self.security.tls_min_version != TlsVersion::Tls13 {
            return Err(NetworkError::config(
                "security.tls_min_version",
                "TLS 1.3 is required in production",
            ));
        }

        if self.load_balancer.endpoints.is_empty() {
            return Err(NetworkError::config(
                "load_balancer.endpoints",
                "At least one endpoint must be configured",
            ));
        }

        Ok(())
    }

    /// Get timeout durations
    #[must_use]
    pub const fn timeouts(&self) -> NetworkTimeouts {
        NetworkTimeouts {
            connection: Duration::from_secs(self.http.connection_timeout_s),
            request: Duration::from_secs(self.http.request_timeout_s),
            keep_alive: Duration::from_secs(self.http.keep_alive_timeout_s),
            websocket_ping: Duration::from_secs(self.websocket.ping_interval_s),
            websocket_pong: Duration::from_secs(self.websocket.pong_timeout_s),
        }
    }
}

/// Network timeout configuration
#[derive(Debug, Clone)]
pub struct NetworkTimeouts {
    /// Connection timeout
    pub connection: Duration,
    /// Request timeout
    pub request: Duration,
    /// Keep-alive timeout
    pub keep_alive: Duration,
    /// WebSocket ping interval
    pub websocket_ping: Duration,
    /// WebSocket pong timeout
    pub websocket_pong: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = NetworkConfig::default();
        assert!(config.http.enable_http2);
        assert!(!config.http.enable_http3);
        assert!(config.security.enable_tls_verification);
        assert_eq!(config.security.tls_min_version, TlsVersion::Tls13);
    }

    #[test]
    fn test_config_validation() {
        let mut config = NetworkConfig::default();
        config.load_balancer.endpoints.push(crate::types::Endpoint::new("https://api.example.com"));
        
        // Should pass validation
        assert!(config.validate_production().is_ok());

        // Should fail with TLS disabled
        config.security.enable_tls_verification = false;
        assert!(config.validate_production().is_err());
    }

    #[test]
    fn test_timeouts() {
        let config = NetworkConfig::default();
        let timeouts = config.timeouts();
        
        assert_eq!(timeouts.connection, Duration::from_secs(30));
        assert_eq!(timeouts.request, Duration::from_secs(60));
        assert_eq!(timeouts.websocket_ping, Duration::from_secs(30));
    }
}
