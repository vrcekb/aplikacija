//! WebSocket Module
//!
//! High-performance WebSocket client with automatic reconnection, message buffering, and heartbeat.
//! Optimized for ultra-low latency financial data streaming.

pub mod client;
pub mod connection;
pub mod handler;
pub mod manager;

pub use client::WebSocketClient;
pub use connection::{WebSocketConnection, ConnectionState};
pub use handler::{MessageHandler, WebSocketHandlers};
pub use manager::WebSocketManager;

use crate::error::NetworkResult;
use crate::types::{ConnectionId, WebSocketMessage};
use async_trait::async_trait;
use std::time::Duration;

/// WebSocket client trait for testability and flexibility
#[async_trait]
pub trait WebSocketClientTrait: Send + Sync {
    /// Connect to WebSocket endpoint
    ///
    /// # Errors
    /// Returns error if connection fails
    async fn connect(&self, url: &str) -> NetworkResult<ConnectionId>;

    /// Send message to specific connection
    ///
    /// # Errors
    /// Returns error if send fails or connection is not found
    async fn send(&self, connection_id: ConnectionId, message: WebSocketMessage) -> NetworkResult<()>;

    /// Close specific connection
    ///
    /// # Errors
    /// Returns error if close fails or connection is not found
    async fn close(&self, connection_id: ConnectionId) -> NetworkResult<()>;

    /// Get connection state
    fn connection_state(&self, connection_id: ConnectionId) -> Option<ConnectionState>;

    /// Get client statistics
    fn stats(&self) -> WebSocketClientStats;

    /// Check if client is healthy
    fn is_healthy(&self) -> bool;
}

/// WebSocket client statistics
#[derive(Debug, Clone)]
pub struct WebSocketClientStats {
    /// Total active connections
    pub active_connections: u32,
    /// Total messages sent
    pub total_messages_sent: u64,
    /// Total messages received
    pub total_messages_received: u64,
    /// Total connection attempts
    pub total_connection_attempts: u64,
    /// Total successful connections
    pub successful_connections: u64,
    /// Total failed connections
    pub failed_connections: u64,
    /// Total reconnection attempts
    pub reconnection_attempts: u64,
    /// Average message latency (microseconds)
    pub avg_message_latency_us: u64,
    /// Connection uptime percentage
    pub connection_uptime_percentage: f64,
}

impl Default for WebSocketClientStats {
    fn default() -> Self {
        Self {
            active_connections: 0,
            total_messages_sent: 0,
            total_messages_received: 0,
            total_connection_attempts: 0,
            successful_connections: 0,
            failed_connections: 0,
            reconnection_attempts: 0,
            avg_message_latency_us: 0,
            connection_uptime_percentage: 0.0_f64,
        }
    }
}

/// WebSocket connection builder with fluent API
pub struct WebSocketConnectionBuilder {
    url: String,
    handlers: Option<WebSocketHandlers>,
    auto_reconnect: bool,
    ping_interval: Option<Duration>,
    pong_timeout: Option<Duration>,
    max_message_size: Option<u32>,
    compression_enabled: bool,
    headers: std::collections::HashMap<String, String>,
}

impl WebSocketConnectionBuilder {
    /// Create new connection builder
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            handlers: None,
            auto_reconnect: true,
            ping_interval: None,
            pong_timeout: None,
            max_message_size: None,
            compression_enabled: false,
            headers: std::collections::HashMap::new(),
        }
    }

    /// Set message handlers
    #[must_use]
    pub fn handlers(mut self, handlers: WebSocketHandlers) -> Self {
        self.handlers = Some(handlers);
        self
    }

    /// Enable/disable automatic reconnection
    #[must_use]
    pub fn auto_reconnect(mut self, enabled: bool) -> Self {
        self.auto_reconnect = enabled;
        self
    }

    /// Set ping interval
    #[must_use]
    pub fn ping_interval(mut self, interval: Duration) -> Self {
        self.ping_interval = Some(interval);
        self
    }

    /// Set pong timeout
    #[must_use]
    pub fn pong_timeout(mut self, timeout: Duration) -> Self {
        self.pong_timeout = Some(timeout);
        self
    }

    /// Set maximum message size
    #[must_use]
    pub fn max_message_size(mut self, size: u32) -> Self {
        self.max_message_size = Some(size);
        self
    }

    /// Enable compression
    #[must_use]
    pub fn compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    /// Add header
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    /// Connect using the specified client
    pub async fn connect(self, client: &dyn WebSocketClientTrait) -> NetworkResult<ConnectionId> {
        // For now, use basic connection - in full implementation, this would pass all options
        client.connect(&self.url).await
    }
}

/// Convenience functions for common WebSocket operations
impl WebSocketConnectionBuilder {
    /// Create connection for real-time data streaming
    pub fn streaming(url: impl Into<String>) -> Self {
        Self::new(url)
            .auto_reconnect(true)
            .ping_interval(Duration::from_secs(30))
            .pong_timeout(Duration::from_secs(10))
            .compression(true)
    }

    /// Create connection for trading operations (low latency)
    pub fn trading(url: impl Into<String>) -> Self {
        Self::new(url)
            .auto_reconnect(true)
            .ping_interval(Duration::from_secs(15))
            .pong_timeout(Duration::from_secs(5))
            .compression(false) // Disable compression for lower latency
    }

    /// Create connection for notifications (reliable delivery)
    pub fn notifications(url: impl Into<String>) -> Self {
        Self::new(url)
            .auto_reconnect(true)
            .ping_interval(Duration::from_secs(60))
            .pong_timeout(Duration::from_secs(30))
            .compression(true)
    }
}

/// WebSocket message priority for queue management
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Low priority - can be dropped if queue is full
    Low = 0,
    /// Normal priority - standard messages
    Normal = 1,
    /// High priority - important messages
    High = 2,
    /// Critical priority - must be delivered
    Critical = 3,
}

impl Default for MessagePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// WebSocket message with metadata
#[derive(Debug, Clone)]
pub struct PrioritizedMessage {
    /// The actual message
    pub message: WebSocketMessage,
    /// Message priority
    pub priority: MessagePriority,
    /// Message timestamp
    pub timestamp: std::time::Instant,
    /// Retry count
    pub retry_count: u32,
}

impl PrioritizedMessage {
    /// Create new prioritized message
    pub fn new(message: WebSocketMessage, priority: MessagePriority) -> Self {
        Self {
            message,
            priority,
            timestamp: std::time::Instant::now(),
            retry_count: 0,
        }
    }

    /// Create high priority message
    pub fn high_priority(message: WebSocketMessage) -> Self {
        Self::new(message, MessagePriority::High)
    }

    /// Create critical priority message
    pub fn critical(message: WebSocketMessage) -> Self {
        Self::new(message, MessagePriority::Critical)
    }

    /// Get message age
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Check if message has expired
    pub fn is_expired(&self, max_age: Duration) -> bool {
        self.age() > max_age
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_builder() {
        let builder = WebSocketConnectionBuilder::new("wss://api.example.com/ws")
            .auto_reconnect(true)
            .ping_interval(Duration::from_secs(30))
            .compression(true)
            .header("Authorization", "Bearer token");

        assert_eq!(builder.url, "wss://api.example.com/ws");
        assert!(builder.auto_reconnect);
        assert_eq!(builder.ping_interval, Some(Duration::from_secs(30)));
        assert!(builder.compression_enabled);
        assert!(builder.headers.contains_key("Authorization"));
    }

    #[test]
    fn test_specialized_builders() {
        let streaming = WebSocketConnectionBuilder::streaming("wss://stream.example.com");
        assert!(streaming.auto_reconnect);
        assert!(streaming.compression_enabled);

        let trading = WebSocketConnectionBuilder::trading("wss://trade.example.com");
        assert!(trading.auto_reconnect);
        assert!(!trading.compression_enabled); // Disabled for low latency

        let notifications = WebSocketConnectionBuilder::notifications("wss://notify.example.com");
        assert!(notifications.auto_reconnect);
        assert!(notifications.compression_enabled);
    }

    #[test]
    fn test_message_priority() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    #[test]
    fn test_prioritized_message() {
        let msg = WebSocketMessage::Text("test".to_string());
        let prioritized = PrioritizedMessage::new(msg.clone(), MessagePriority::High);

        assert_eq!(prioritized.priority, MessagePriority::High);
        assert_eq!(prioritized.retry_count, 0);
        assert!(prioritized.age() < Duration::from_millis(10));

        let critical = PrioritizedMessage::critical(msg);
        assert_eq!(critical.priority, MessagePriority::Critical);
    }

    #[test]
    fn test_stats_default() {
        let stats = WebSocketClientStats::default();
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.total_messages_sent, 0);
        assert_eq!(stats.connection_uptime_percentage, 0.0_f64);
    }
}
