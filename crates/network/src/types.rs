//! `TallyIO` Network Types
//!
//! Core types and data structures for network operations.
//! Optimized for performance and memory efficiency.

use crate::error::NetworkResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use uuid::Uuid;

/// Unique identifier for network connections
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConnectionId(pub Uuid);

impl ConnectionId {
    /// Generate new connection ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ConnectionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Network endpoint information
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Endpoint {
    /// Endpoint URL
    pub url: String,
    /// Optional socket address for direct connections
    pub socket_addr: Option<SocketAddr>,
    /// Endpoint priority (higher = more preferred)
    pub priority: u32,
    /// Endpoint weight for load balancing
    pub weight: u32,
    /// Health check configuration
    pub health_check: Option<HealthCheckConfig>,
}

impl Endpoint {
    /// Create new endpoint
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            socket_addr: None,
            priority: 100,
            weight: 100,
            health_check: None,
        }
    }

    /// Set priority
    #[must_use]
    pub const fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set weight
    #[must_use]
    pub const fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }

    /// Set health check configuration
    #[must_use]
    pub fn with_health_check(mut self, health_check: HealthCheckConfig) -> Self {
        self.health_check = Some(health_check);
        self
    }
}

/// Health check configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
    /// Health check path (for HTTP endpoints)
    pub path: Option<String>,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(2),
            failure_threshold: 3,
            success_threshold: 2,
            path: Some("/health".to_string()),
        }
    }
}

/// Endpoint health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointHealth {
    /// Endpoint is healthy
    Healthy,
    /// Endpoint is unhealthy
    Unhealthy,
    /// Endpoint health is unknown
    Unknown,
}

impl Default for EndpointHealth {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    /// Connection ID
    pub connection_id: ConnectionId,
    /// Connection establishment time
    pub connected_at: SystemTime,
    /// Last activity timestamp
    pub last_activity: Instant,
    /// Total bytes sent
    pub bytes_sent: Arc<AtomicU64>,
    /// Total bytes received
    pub bytes_received: Arc<AtomicU64>,
    /// Total messages sent
    pub messages_sent: Arc<AtomicU64>,
    /// Total messages received
    pub messages_received: Arc<AtomicU64>,
    /// Connection errors count
    pub error_count: Arc<AtomicU64>,
    /// Average latency (microseconds)
    pub avg_latency_us: Arc<AtomicU64>,
}

impl ConnectionStats {
    /// Create new connection statistics
    #[must_use]
    pub fn new(connection_id: ConnectionId) -> Self {
        Self {
            connection_id,
            connected_at: SystemTime::now(),
            last_activity: Instant::now(),
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            error_count: Arc::new(AtomicU64::new(0)),
            avg_latency_us: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record bytes sent
    pub fn record_bytes_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes received
    pub fn record_bytes_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record message sent
    pub fn record_message_sent(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Record message received
    pub fn record_message_received(&self) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Record error
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update latency (exponential moving average)
    pub fn update_latency(&self, latency_us: u64) {
        let current = self.avg_latency_us.load(Ordering::Relaxed);
        // Simple exponential moving average with alpha = 0.1
        let new_avg = if current == 0 {
            latency_us
        } else {
            (current * 9 + latency_us) / 10
        };
        self.avg_latency_us.store(new_avg, Ordering::Relaxed);
    }

    /// Get current statistics snapshot
    #[must_use]
    pub fn snapshot(&self) -> ConnectionStatsSnapshot {
        ConnectionStatsSnapshot {
            connection_id: self.connection_id,
            connected_at: self.connected_at,
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            avg_latency_us: self.avg_latency_us.load(Ordering::Relaxed),
        }
    }
}

/// Immutable snapshot of connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatsSnapshot {
    /// Connection ID
    pub connection_id: ConnectionId,
    /// Connection establishment time
    pub connected_at: SystemTime,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Connection errors count
    pub error_count: u64,
    /// Average latency (microseconds)
    pub avg_latency_us: u64,
}

/// HTTP request method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HttpMethod {
    /// GET method
    Get,
    /// POST method
    Post,
    /// PUT method
    Put,
    /// DELETE method
    Delete,
    /// PATCH method
    Patch,
    /// HEAD method
    Head,
    /// OPTIONS method
    Options,
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Post => write!(f, "POST"),
            Self::Put => write!(f, "PUT"),
            Self::Delete => write!(f, "DELETE"),
            Self::Patch => write!(f, "PATCH"),
            Self::Head => write!(f, "HEAD"),
            Self::Options => write!(f, "OPTIONS"),
        }
    }
}

/// HTTP headers type
pub type HttpHeaders = HashMap<String, String>;

/// HTTP request builder
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method
    pub method: HttpMethod,
    /// Request URL
    pub url: String,
    /// Request headers
    pub headers: HttpHeaders,
    /// Request body
    pub body: Option<Vec<u8>>,
    /// Request timeout
    pub timeout: Option<Duration>,
}

impl HttpRequest {
    /// Create new HTTP request
    pub fn new(method: HttpMethod, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: HashMap::new(),
            body: None,
            timeout: None,
        }
    }

    /// Add header
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Set body
    #[must_use]
    pub fn body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
    }

    /// Set JSON body
    ///
    /// # Errors
    ///
    /// Returns `NetworkError` if JSON serialization fails
    pub fn json_body<T: Serialize>(mut self, data: &T) -> NetworkResult<Self> {
        let json_bytes = serde_json::to_vec(data).map_err(crate::error::NetworkError::from)?;
        self.headers.insert("Content-Type".to_string(), "application/json".to_string());
        self.body = Some(json_bytes);
        Ok(self)
    }

    /// Set timeout
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// HTTP response
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// Response status code
    pub status_code: u16,
    /// Response headers
    pub headers: HttpHeaders,
    /// Response body
    pub body: Vec<u8>,
    /// Response latency
    pub latency: Duration,
}

impl HttpResponse {
    /// Parse JSON response body
    ///
    /// # Errors
    ///
    /// Returns `NetworkError` if JSON deserialization fails
    pub fn json<T: for<'de> Deserialize<'de>>(&self) -> NetworkResult<T> {
        serde_json::from_slice(&self.body).map_err(crate::error::NetworkError::from)
    }

    /// Get response body as string
    ///
    /// # Errors
    ///
    /// Returns `NetworkError` if UTF-8 conversion fails
    pub fn text(&self) -> NetworkResult<String> {
        String::from_utf8(self.body.clone()).map_err(|e| {
            crate::error::NetworkError::internal(format!("Invalid UTF-8 in response: {e}"))
        })
    }

    /// Check if response is successful (2xx status code)
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.status_code >= 200 && self.status_code < 300
    }

    /// Check if response is client error (4xx status code)
    #[must_use]
    pub const fn is_client_error(&self) -> bool {
        self.status_code >= 400 && self.status_code < 500
    }

    /// Check if response is server error (5xx status code)
    #[must_use]
    pub const fn is_server_error(&self) -> bool {
        self.status_code >= 500 && self.status_code < 600
    }
}

/// WebSocket message types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebSocketMessage {
    /// Text message
    Text(String),
    /// Binary message
    Binary(Vec<u8>),
    /// Ping frame
    Ping(Vec<u8>),
    /// Pong frame
    Pong(Vec<u8>),
    /// Close frame
    Close(Option<WebSocketCloseFrame>),
}

/// WebSocket close frame
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebSocketCloseFrame {
    /// Close code
    pub code: u16,
    /// Close reason
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_id() {
        let id1 = ConnectionId::new();
        let id2 = ConnectionId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_endpoint_builder() {
        let endpoint = Endpoint::new("https://api.example.com")
            .with_priority(200)
            .with_weight(150)
            .with_health_check(HealthCheckConfig::default());

        assert_eq!(endpoint.url, "https://api.example.com");
        assert_eq!(endpoint.priority, 200);
        assert_eq!(endpoint.weight, 150);
        assert!(endpoint.health_check.is_some());
    }

    #[test]
    fn test_connection_stats() {
        let id = ConnectionId::new();
        let stats = ConnectionStats::new(id);

        stats.record_bytes_sent(100);
        stats.record_bytes_received(200);
        stats.record_message_sent();
        stats.record_message_received();
        stats.update_latency(1500);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.connection_id, id);
        assert_eq!(snapshot.bytes_sent, 100);
        assert_eq!(snapshot.bytes_received, 200);
        assert_eq!(snapshot.messages_sent, 1);
        assert_eq!(snapshot.messages_received, 1);
        assert_eq!(snapshot.avg_latency_us, 1500);
    }

    #[test]
    fn test_http_request_builder() -> NetworkResult<()> {
        let request = HttpRequest::new(HttpMethod::Post, "https://api.example.com/data")
            .header("Authorization", "Bearer token")
            .json_body(&serde_json::json!({"key": "value"}))?
            .timeout(Duration::from_secs(30));

        assert_eq!(request.method, HttpMethod::Post);
        assert_eq!(request.url, "https://api.example.com/data");
        assert!(request.headers.contains_key("Authorization"));
        assert!(request.headers.contains_key("Content-Type"));
        assert!(request.body.is_some());
        assert_eq!(request.timeout, Some(Duration::from_secs(30)));

        Ok(())
    }

    #[test]
    fn test_http_response() -> Result<(), Box<dyn std::error::Error>> {
        let response = HttpResponse {
            status_code: 200,
            headers: HashMap::new(),
            body: b"Hello, World!".to_vec(),
            latency: Duration::from_millis(100),
        };

        assert!(response.is_success());
        assert!(!response.is_client_error());
        assert!(!response.is_server_error());

        let text = response.text().map_err(|e| format!("Failed to get response text: {e}"))?;
        assert_eq!(text, "Hello, World!");
        Ok(())
    }
}
