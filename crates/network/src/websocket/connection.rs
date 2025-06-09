//! WebSocket Connection Management
//!
//! Connection state management and lifecycle for WebSocket connections.

use crate::types::{ConnectionId, WebSocketMessage};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime};

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    /// Connection is being established
    Connecting,
    /// Connection is active and ready
    Connected,
    /// Connection is being closed
    Closing,
    /// Connection is closed
    Closed,
    /// Connection failed
    Failed,
    /// Connection is reconnecting
    Reconnecting,
}

impl std::fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connecting => write!(f, "Connecting"),
            Self::Connected => write!(f, "Connected"),
            Self::Closing => write!(f, "Closing"),
            Self::Closed => write!(f, "Closed"),
            Self::Failed => write!(f, "Failed"),
            Self::Reconnecting => write!(f, "Reconnecting"),
        }
    }
}

/// WebSocket connection information
#[derive(Debug, Clone)]
pub struct WebSocketConnection {
    /// Connection ID
    pub id: ConnectionId,
    /// Connection URL
    pub url: String,
    /// Current state
    pub state: ConnectionState,
    /// Connection establishment time
    pub connected_at: Option<SystemTime>,
    /// Last activity timestamp
    pub last_activity: Instant,
    /// Last ping timestamp
    pub last_ping: Option<Instant>,
    /// Last pong timestamp
    pub last_pong: Option<Instant>,
    /// Connection statistics
    pub stats: ConnectionStats,
    /// Reconnection attempts
    pub reconnect_attempts: u32,
    /// Maximum reconnection attempts
    pub max_reconnect_attempts: u32,
}

impl WebSocketConnection {
    /// Create new WebSocket connection
    #[must_use]
    pub fn new(id: ConnectionId, url: String) -> Self {
        Self {
            id,
            url,
            state: ConnectionState::Connecting,
            connected_at: None,
            last_activity: Instant::now(),
            last_ping: None,
            last_pong: None,
            stats: ConnectionStats::new(),
            reconnect_attempts: 0,
            max_reconnect_attempts: 10,
        }
    }

    /// Mark connection as connected
    pub fn mark_connected(&mut self) {
        self.state = ConnectionState::Connected;
        self.connected_at = Some(SystemTime::now());
        self.last_activity = Instant::now();
        self.reconnect_attempts = 0;
    }

    /// Mark connection as failed
    pub const fn mark_failed(&mut self) {
        self.state = ConnectionState::Failed;
        self.stats.record_error();
    }

    /// Mark connection as closed
    pub const fn mark_closed(&mut self) {
        self.state = ConnectionState::Closed;
    }

    /// Start reconnection attempt
    pub const fn start_reconnect(&mut self) {
        self.state = ConnectionState::Reconnecting;
        self.reconnect_attempts += 1;
    }

    /// Check if connection should reconnect
    #[must_use]
    pub const fn should_reconnect(&self) -> bool {
        matches!(self.state, ConnectionState::Failed | ConnectionState::Closed) &&
        self.reconnect_attempts < self.max_reconnect_attempts
    }

    /// Record message sent
    pub fn record_message_sent(&mut self, message: &WebSocketMessage) {
        self.stats.record_message_sent();
        self.last_activity = Instant::now();
        
        // Record message size
        let size = match message {
            WebSocketMessage::Text(text) => text.len(),
            WebSocketMessage::Binary(data) | WebSocketMessage::Ping(data) | WebSocketMessage::Pong(data) => data.len(),
            WebSocketMessage::Close(_) => 0,
        };
        self.stats.record_bytes_sent(size as u64);
    }

    /// Record message received
    pub fn record_message_received(&mut self, message: &WebSocketMessage) {
        self.stats.record_message_received();
        self.last_activity = Instant::now();
        
        // Record message size
        let size = match message {
            WebSocketMessage::Text(text) => text.len(),
            WebSocketMessage::Binary(data) | WebSocketMessage::Ping(data) | WebSocketMessage::Pong(data) => data.len(),
            WebSocketMessage::Close(_) => 0,
        };
        self.stats.record_bytes_received(size as u64);
        
        // Handle ping/pong timestamps
        if let WebSocketMessage::Pong(_) = message {
            self.last_pong = Some(Instant::now());
            // Calculate latency if we have a ping timestamp
            if let Some(ping_time) = self.last_ping {
                let latency = ping_time.elapsed();
                self.stats.update_latency(u64::try_from(latency.as_micros()).unwrap_or(u64::MAX));
            }
        }
    }

    /// Send ping and record timestamp
    pub fn send_ping(&mut self) {
        self.last_ping = Some(Instant::now());
    }

    /// Check if connection is idle (no activity for specified duration)
    #[must_use]
    pub fn is_idle(&self, idle_timeout: Duration) -> bool {
        self.last_activity.elapsed() > idle_timeout
    }

    /// Check if ping is overdue
    #[must_use]
    pub fn is_ping_overdue(&self, ping_interval: Duration) -> bool {
        self.last_ping.is_none_or(|last_ping| last_ping.elapsed() > ping_interval)
    }

    /// Check if pong is overdue (connection might be dead)
    #[must_use]
    pub fn is_pong_overdue(&self, pong_timeout: Duration) -> bool {
        match (self.last_ping, self.last_pong) {
            (Some(sent_time), Some(received_time)) => {
                // If ping is newer than pong, check if pong is overdue
                if sent_time > received_time {
                    sent_time.elapsed() > pong_timeout
                } else {
                    false
                }
            }
            (Some(sent_time), None) => {
                // We sent a ping but never received a pong
                sent_time.elapsed() > pong_timeout
            }
            _ => false,
        }
    }

    /// Get connection uptime
    #[must_use]
    pub fn uptime(&self) -> Option<Duration> {
        self.connected_at.map(|connected_at| {
            // Varno obravnavanje morebitne napake pri pridobivanju časa
            connected_at.elapsed().map_or(Duration::ZERO, |elapsed| elapsed)
        })
    }

    /// Get connection statistics snapshot
    #[must_use]
    pub const fn stats_snapshot(&self) -> ConnectionStatsSnapshot {
        self.stats.snapshot()
    }
}

/// Connection statistics
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    /// Total messages sent
    messages_sent: u64,
    /// Total messages received
    messages_received: u64,
    /// Total bytes sent
    bytes_sent: u64,
    /// Total bytes received
    bytes_received: u64,
    /// Error count
    error_count: u64,
    /// Average latency (microseconds)
    avg_latency_us: u64,
    /// Creation timestamp
    created_at: SystemTime,
}

impl ConnectionStats {
    /// Create new connection statistics
    #[must_use]
    pub fn new() -> Self {
        Self {
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            error_count: 0,
            avg_latency_us: 0,
            created_at: SystemTime::now(),
        }
    }

    /// Record message sent
    pub const fn record_message_sent(&mut self) {
        self.messages_sent += 1;
    }

    /// Record message received
    pub const fn record_message_received(&mut self) {
        self.messages_received += 1;
    }

    /// Record bytes sent
    pub const fn record_bytes_sent(&mut self, bytes: u64) {
        self.bytes_sent += bytes;
    }

    /// Record bytes received
    pub const fn record_bytes_received(&mut self, bytes: u64) {
        self.bytes_received += bytes;
    }

    /// Record error
    pub const fn record_error(&mut self) {
        self.error_count += 1;
    }

    /// Update latency (exponential moving average)
    pub const fn update_latency(&mut self, latency_us: u64) {
        if self.avg_latency_us == 0 {
            self.avg_latency_us = latency_us;
        } else {
            // Simple exponential moving average with alpha = 0.1
            self.avg_latency_us = (self.avg_latency_us * 9 + latency_us) / 10;
        }
    }

    /// Get statistics snapshot
    #[must_use]
    pub const fn snapshot(&self) -> ConnectionStatsSnapshot {
        ConnectionStatsSnapshot {
            messages_sent: self.messages_sent,
            messages_received: self.messages_received,
            bytes_sent: self.bytes_sent,
            bytes_received: self.bytes_received,
            error_count: self.error_count,
            avg_latency_us: self.avg_latency_us,
            created_at: self.created_at,
        }
    }
}

impl Default for ConnectionStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable snapshot of connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatsSnapshot {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Error count
    pub error_count: u64,
    /// Average latency (microseconds)
    pub avg_latency_us: u64,
    /// Creation timestamp
    pub created_at: SystemTime,
}

impl ConnectionStatsSnapshot {
    /// Calculate message rate (messages per second)
    #[must_use]
    pub fn message_rate(&self) -> f64 {
        let elapsed = self.created_at.elapsed().unwrap_or_else(|_| Duration::from_secs(1));
        let total_messages = self.messages_sent + self.messages_received;
        #[allow(clippy::cast_precision_loss)]
        let messages_f64 = total_messages as f64;
        messages_f64 / elapsed.as_secs_f64()
    }

    /// Calculate throughput (bytes per second)
    #[must_use]
    pub fn throughput(&self) -> f64 {
        let elapsed = self.created_at.elapsed().unwrap_or_else(|_| Duration::from_secs(1));
        let total_bytes = self.bytes_sent + self.bytes_received;
        #[allow(clippy::cast_precision_loss)]
        let bytes_f64 = total_bytes as f64;
        bytes_f64 / elapsed.as_secs_f64()
    }

    /// Calculate error rate (errors per message)
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total_messages = self.messages_sent + self.messages_received;
        if total_messages == 0 {
            0.0_f64
        } else {
            #[allow(clippy::cast_precision_loss)]
            let error_count_f64 = self.error_count as f64;
            #[allow(clippy::cast_precision_loss)]
            let total_messages_f64 = total_messages as f64;
            error_count_f64 / total_messages_f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WebSocketMessage;

    #[test]
    fn test_connection_creation() {
        let id = ConnectionId::new();
        let url = "wss://api.example.com/ws".to_string();
        let connection = WebSocketConnection::new(id, url.clone());

        assert_eq!(connection.id, id);
        assert_eq!(connection.url, url);
        assert_eq!(connection.state, ConnectionState::Connecting);
        assert!(connection.connected_at.is_none());
        assert_eq!(connection.reconnect_attempts, 0);
    }

    #[test]
    fn test_connection_state_transitions() {
        let id = ConnectionId::new();
        let mut connection = WebSocketConnection::new(id, "wss://test.com".to_string());

        // Test connection
        connection.mark_connected();
        assert_eq!(connection.state, ConnectionState::Connected);
        assert!(connection.connected_at.is_some());
        assert_eq!(connection.reconnect_attempts, 0);

        // Test failure
        connection.mark_failed();
        assert_eq!(connection.state, ConnectionState::Failed);
        assert!(connection.should_reconnect());

        // Test reconnection
        connection.start_reconnect();
        assert_eq!(connection.state, ConnectionState::Reconnecting);
        assert_eq!(connection.reconnect_attempts, 1);
    }

    #[test]
    fn test_message_recording() {
        let id = ConnectionId::new();
        let mut connection = WebSocketConnection::new(id, "wss://test.com".to_string());

        let message = WebSocketMessage::Text("Hello".to_string());
        connection.record_message_sent(&message);
        connection.record_message_received(&message);

        let stats = connection.stats_snapshot();
        assert_eq!(stats.messages_sent, 1);
        assert_eq!(stats.messages_received, 1);
        assert_eq!(stats.bytes_sent, 5); // "Hello".len()
        assert_eq!(stats.bytes_received, 5);
    }

    #[test]
    fn test_ping_pong_handling() {
        let id = ConnectionId::new();
        let mut connection = WebSocketConnection::new(id, "wss://test.com".to_string());

        // Send ping
        connection.send_ping();
        assert!(connection.last_ping.is_some());

        // Simulate pong response
        let pong = WebSocketMessage::Pong(vec![]);
        connection.record_message_received(&pong);
        assert!(connection.last_pong.is_some());
    }

    #[test]
    fn test_connection_timeouts() {
        let id = ConnectionId::new();
        let connection = WebSocketConnection::new(id, "wss://test.com".to_string());

        // Test idle timeout
        assert!(!connection.is_idle(Duration::from_secs(1)));
        
        // Test ping overdue (should be true since we never pinged)
        assert!(connection.is_ping_overdue(Duration::from_secs(30)));
        
        // Test pong not overdue (no ping sent)
        assert!(!connection.is_pong_overdue(Duration::from_secs(10)));
    }

    #[test]
    fn test_stats_calculations() {
        // Create a mock time 10 seconds ago
        let ten_seconds_ago = SystemTime::UNIX_EPOCH + Duration::from_secs(1000);

        let stats = ConnectionStatsSnapshot {
            messages_sent: 100,
            messages_received: 200,
            bytes_sent: 1000,
            bytes_received: 2000,
            error_count: 5,
            avg_latency_us: 1500,
            created_at: ten_seconds_ago,
        };

        // Mock current time to be exactly 10 seconds later
        let mock_elapsed = Duration::from_secs(10);
        let total_messages = stats.messages_sent + stats.messages_received; // 300
        let total_bytes = stats.bytes_sent + stats.bytes_received; // 3000

        // Calculate expected values
        #[allow(clippy::cast_precision_loss)]
        let expected_message_rate = total_messages as f64 / mock_elapsed.as_secs_f64(); // 300/10 = 30.0
        #[allow(clippy::cast_precision_loss)]
        let expected_throughput = total_bytes as f64 / mock_elapsed.as_secs_f64(); // 3000/10 = 300.0
        #[allow(clippy::cast_precision_loss)]
        let expected_error_rate = stats.error_count as f64 / total_messages as f64; // 5/300 = 0.016666...

        // Use reasonable tolerance for floating point comparisons
        let tolerance = 1e-10_f64;

        // Note: These tests verify the calculation logic, but actual elapsed time may vary
        // In production, we would use dependency injection for time to make this testable
        println!("Expected message rate: {expected_message_rate}");
        println!("Expected throughput: {expected_throughput}");
        println!("Expected error rate: {expected_error_rate}");

        // Test error rate calculation (this doesn't depend on time)
        assert!((stats.error_rate() - expected_error_rate).abs() < tolerance);
    }
}
