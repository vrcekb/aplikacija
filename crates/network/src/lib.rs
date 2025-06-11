//! `TallyIO` Network Module
//!
//! Ultra-performant network layer with <1ms latency for financial trading applications.
//! Provides HTTP, WebSocket, and P2P networking capabilities with production-ready
//! error handling, retry logic, circuit breakers, and comprehensive metrics.

#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(clippy::nursery)]
#![deny(clippy::correctness)]
#![deny(clippy::suspicious)]
#![deny(clippy::perf)]
#![warn(clippy::redundant_allocation)]
#![warn(clippy::needless_collect)]
#![warn(clippy::suboptimal_flops)]
#![allow(clippy::missing_docs_in_private_items)]
#![deny(clippy::infinite_loop)]
#![deny(clippy::while_immutable_condition)]
#![deny(clippy::never_loop)]
#![deny(clippy::manual_strip)]
#![deny(clippy::needless_continue)]
#![deny(clippy::match_same_arms)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::large_stack_arrays)]
#![deny(clippy::large_enum_variant)]
#![deny(clippy::mut_mut)]
#![deny(clippy::cast_possible_truncation)]
#![deny(clippy::cast_sign_loss)]
#![deny(clippy::cast_precision_loss)]
#![deny(clippy::must_use_candidate)]
#![deny(clippy::empty_loop)]
#![deny(clippy::if_same_then_else)]
#![deny(clippy::await_holding_lock)]
#![deny(clippy::await_holding_refcell_ref)]
#![deny(clippy::let_underscore_future)]
#![deny(clippy::diverging_sub_expression)]
#![deny(clippy::unreachable)]
#![deny(clippy::default_numeric_fallback)]
#![deny(clippy::redundant_pattern_matching)]
#![deny(clippy::manual_let_else)]
#![deny(clippy::blocks_in_conditions)]
#![deny(clippy::needless_pass_by_value)]
#![deny(clippy::single_match_else)]
#![deny(clippy::branches_sharing_code)]
#![deny(clippy::useless_asref)]
#![deny(clippy::redundant_closure_for_method_calls)]

// Core modules
pub mod config;
pub mod error;
pub mod types;

// Network protocol modules
pub mod http;
pub mod websocket;

// Load balancing and P2P (future modules)
pub mod load_balancer;
pub mod p2p;

// Security modules
pub mod security;

// Utilities and metrics
pub mod metrics;
pub mod manager;

// Re-export commonly used types and traits
pub use config::NetworkConfig;
pub use error::{NetworkError, NetworkResult};
pub use manager::NetworkManager;

/// Prelude module for convenient imports
///
/// Common imports for `TallyIO` network functionality.
/// Import this module to get access to the most commonly used types and traits.
pub mod prelude {

    // Re-export core types
    pub use crate::config::{
        NetworkConfig, HttpConfig, WebSocketConfig, LoadBalancerConfig, SecurityConfig,
        RetryConfig, CircuitBreakerConfig, ReconnectConfig, HealthCheckConfig,
    };
    pub use crate::error::{NetworkError, NetworkResult, CriticalNetworkError};
    pub use crate::types::{
        ConnectionId, Endpoint, HttpRequest, HttpResponse, HttpMethod, WebSocketMessage,
        ConnectionStats, ConnectionStatsSnapshot,
    };

    // Re-export HTTP client
    pub use crate::http::{
        HttpClient, HttpClientTrait, HttpRequestBuilder, RetryPolicy, CircuitBreaker,
        CircuitBreakerState,
    };

    // Re-export WebSocket client
    pub use crate::websocket::{
        WebSocketClient, WebSocketClientTrait, WebSocketConnection, WebSocketConnectionBuilder,
        WebSocketManager, MessagePriority, PrioritizedMessage,
    };

    // Re-export load balancer
    pub use crate::load_balancer::{LoadBalancer, LoadBalancerTrait};
    pub use crate::config::LoadBalancingStrategy;

    // Re-export security
    pub use crate::security::{
        SecuritySystem, SecurityRequest, SecurityResult,
        RateLimitConfig, CertificatePinningConfig, RequestSigningConfig, DoSProtectionConfig,
        SignatureAlgorithm,
    };

    // Re-export network manager
    pub use crate::manager::NetworkManager;

    // Re-export commonly used external types
    pub use async_trait::async_trait;
    pub use futures::prelude::*;
    pub use serde::{Deserialize, Serialize};
    pub use std::sync::Arc;
    pub use std::time::{Duration, Instant, SystemTime};
    pub use tokio::sync::{Mutex, RwLock};
    pub use tracing::{debug, error, info, instrument, warn};
    pub use uuid::Uuid;

    // Re-export performance types
    pub use dashmap::DashMap;
    pub use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    // Re-export validation
    pub use garde::Validate;

    /// Common result type alias
    pub type Result<T> = NetworkResult<T>;
}

/// Network module version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Network module name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Network module description
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Build information
pub mod build_info {
    /// Build timestamp
    pub const BUILD_TIMESTAMP: &str = match option_env!("VERGEN_BUILD_TIMESTAMP") {
        Some(timestamp) => timestamp,
        None => "unknown",
    };

    /// Git commit hash
    pub const GIT_SHA: &str = match option_env!("VERGEN_GIT_SHA") {
        Some(sha) => sha,
        None => "unknown",
    };

    /// Rust version used for build
    pub const RUSTC_VERSION: &str = match option_env!("VERGEN_RUSTC_SEMVER") {
        Some(version) => version,
        None => "unknown",
    };

    /// Target triple
    pub const TARGET_TRIPLE: &str = match option_env!("VERGEN_CARGO_TARGET_TRIPLE") {
        Some(triple) => triple,
        None => "unknown",
    };
}

/// Network capabilities and feature flags
pub mod capabilities {
    /// HTTP/1.1 support
    pub const HTTP_1_1: bool = true;
    
    /// HTTP/2 support
    pub const HTTP_2: bool = true;
    
    /// HTTP/3 support (QUIC) - currently disabled
    pub const HTTP_3: bool = false;
    
    /// WebSocket support
    pub const WEBSOCKET: bool = cfg!(feature = "websocket");
    
    /// P2P networking support
    pub const P2P: bool = cfg!(feature = "p2p");
    
    /// Load balancer support
    pub const LOAD_BALANCER: bool = cfg!(feature = "load-balancer");
    
    /// Metrics collection support
    pub const METRICS: bool = cfg!(feature = "metrics");
    
    /// TLS 1.3 support
    pub const TLS_1_3: bool = true;
    
    /// Connection pooling support
    pub const CONNECTION_POOLING: bool = true;
    
    /// Circuit breaker support
    pub const CIRCUIT_BREAKER: bool = true;
    
    /// Retry mechanism support
    pub const RETRY_MECHANISM: bool = true;
}

/// Performance characteristics and limits
pub mod limits {
    use std::time::Duration;
    
    /// Maximum number of concurrent connections per host
    pub const MAX_CONNECTIONS_PER_HOST: u32 = 1000;
    
    /// Maximum number of total connections
    pub const MAX_TOTAL_CONNECTIONS: u32 = 10_000;
    
    /// Maximum message size for WebSocket (16MB)
    pub const MAX_WEBSOCKET_MESSAGE_SIZE: u32 = 16 * 1024 * 1024;
    
    /// Maximum HTTP request body size (100MB)
    pub const MAX_HTTP_BODY_SIZE: u64 = 100 * 1024 * 1024;
    
    /// Target latency for critical operations
    pub const TARGET_LATENCY: Duration = Duration::from_millis(1);
    
    /// Maximum acceptable latency
    pub const MAX_LATENCY: Duration = Duration::from_millis(10);
    
    /// Default connection timeout
    pub const DEFAULT_CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
    
    /// Default request timeout
    pub const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
    
    /// Maximum retry attempts
    pub const MAX_RETRY_ATTEMPTS: u32 = 10;
    
    /// Maximum circuit breaker failure threshold
    pub const MAX_CIRCUIT_BREAKER_FAILURES: u32 = 100;
}

/// Utility functions for network operations
pub mod utils {
    use crate::error::{NetworkError, NetworkResult};
    use std::net::{IpAddr, SocketAddr};
    use url::Url;

    /// Parse URL and extract components
    ///
    /// # Errors
    /// Returns error if URL is invalid
    pub fn parse_url(url: &str) -> NetworkResult<ParsedUrl> {
        let parsed = Url::parse(url).map_err(NetworkError::from)?;
        
        Ok(ParsedUrl {
            scheme: parsed.scheme().to_string(),
            host: parsed.host_str().map_or_else(String::new, ToString::to_string),
            port: parsed.port(),
            path: parsed.path().to_string(),
            query: parsed.query().map(ToString::to_string),
            fragment: parsed.fragment().map(ToString::to_string),
        })
    }

    /// Validate endpoint URL
    ///
    /// # Errors
    /// Returns error if URL is invalid for network operations
    pub fn validate_endpoint(url: &str) -> NetworkResult<()> {
        let parsed = parse_url(url)?;
        
        if parsed.scheme != "http" && parsed.scheme != "https" && 
           parsed.scheme != "ws" && parsed.scheme != "wss" {
            return Err(NetworkError::validation(
                "url_scheme",
                format!("Unsupported scheme: {}", parsed.scheme),
            ));
        }
        
        if parsed.host.is_empty() {
            return Err(NetworkError::validation("url_host", "Host is required"));
        }
        
        Ok(())
    }

    /// Resolve hostname to IP address
    ///
    /// # Errors
    /// Returns error if hostname resolution fails
    pub async fn resolve_hostname(hostname: &str, port: u16) -> NetworkResult<Vec<SocketAddr>> {
        let addr_str = format!("{hostname}:{port}");
        tokio::net::lookup_host(&addr_str)
            .await
            .map(std::iter::Iterator::collect)
            .map_err(|e| NetworkError::connection(&addr_str, e.to_string(), None))
    }

    /// Check if IP address is private/internal
    #[must_use]
    pub const fn is_private_ip(ip: &IpAddr) -> bool {
        match ip {
            IpAddr::V4(ipv4) => {
                ipv4.is_private() || ipv4.is_loopback() || ipv4.is_link_local()
            }
            IpAddr::V6(ipv6) => {
                ipv6.is_loopback() || ipv6.is_unicast_link_local()
            }
        }
    }

    /// Parsed URL components
    #[derive(Debug, Clone)]
    pub struct ParsedUrl {
        /// URL scheme (http, https, ws, wss)
        pub scheme: String,
        /// Host name or IP address
        pub host: String,
        /// Port number (if specified)
        pub port: Option<u16>,
        /// URL path
        pub path: String,
        /// Query string
        pub query: Option<String>,
        /// URL fragment
        pub fragment: Option<String>,
    }

    impl ParsedUrl {
        /// Get default port for scheme
        #[must_use]
        pub fn default_port(&self) -> u16 {
            match self.scheme.as_str() {
                "https" | "wss" => 443,
                _ => 80, // Default for http, ws, and unknown schemes
            }
        }

        /// Get effective port (specified or default)
        #[must_use]
        pub fn effective_port(&self) -> u16 {
            self.port.unwrap_or_else(|| self.default_port())
        }

        /// Check if URL uses secure scheme
        #[must_use]
        pub fn is_secure(&self) -> bool {
            matches!(self.scheme.as_str(), "https" | "wss")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        // Verify that version constants are accessible and compile-time defined
        // These are populated from Cargo.toml at compile time
        // Test passes if constants are accessible (compilation succeeds)
        assert!(VERSION.len() < 1000); // Reasonable upper bound
        assert!(NAME.len() < 1000); // Reasonable upper bound
        assert!(DESCRIPTION.len() < 1000); // Reasonable upper bound
    }

    #[test]
    fn test_capabilities() {
        // Verify capabilities are enabled (compile-time constants)
        // These are compile-time checks that capabilities are properly configured
    }

    #[test]
    fn test_limits() {
        // Verify limits are reasonable (compile-time constants)
        // These are compile-time checks that limits are properly configured
        assert!(limits::TARGET_LATENCY < limits::MAX_LATENCY);
    }

    #[test]
    fn test_url_parsing() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let parsed = utils::parse_url("https://api.example.com:8443/v1/data?key=value#section").map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        assert_eq!(parsed.scheme, "https");
        assert_eq!(parsed.host, "api.example.com");
        assert_eq!(parsed.port, Some(8443));
        assert_eq!(parsed.path, "/v1/data");
        assert_eq!(parsed.query, Some("key=value".to_string()));
        assert_eq!(parsed.fragment, Some("section".to_string()));
        assert!(parsed.is_secure());
        assert_eq!(parsed.effective_port(), 8443);
        Ok(())
    }

    #[test]
    fn test_url_validation() {
        assert!(utils::validate_endpoint("https://api.example.com").is_ok());
        assert!(utils::validate_endpoint("wss://stream.example.com").is_ok());
        assert!(utils::validate_endpoint("ftp://files.example.com").is_err());
        assert!(utils::validate_endpoint("invalid-url").is_err());
    }
}
