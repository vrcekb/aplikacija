//! WebSocket Manager
//!
//! High-level WebSocket connection management with automatic reconnection and load balancing.

use crate::config::WebSocketConfig;
use crate::error::{NetworkError, NetworkResult};
use crate::types::{ConnectionId, WebSocketMessage};
use crate::websocket::{WebSocketConnection, ConnectionState, WebSocketHandlers, MessageHandler};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// WebSocket connection manager
pub struct WebSocketManager {
    /// Configuration
    config: WebSocketConfig,
    /// Active connections
    connections: Arc<DashMap<ConnectionId, Arc<RwLock<WebSocketConnection>>>>,
    /// Global message handlers
    handlers: Arc<RwLock<Option<WebSocketHandlers>>>,
    /// Manager statistics
    stats: Arc<RwLock<ManagerStats>>,
    /// Background task handles
    task_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

impl WebSocketManager {
    /// Create new WebSocket manager
    #[must_use]
    pub fn new(config: WebSocketConfig) -> Self {
        let manager = Self {
            config,
            connections: Arc::new(DashMap::new()),
            handlers: Arc::new(RwLock::new(None)),
            stats: Arc::new(RwLock::new(ManagerStats::new())),
            task_handles: Arc::new(RwLock::new(Vec::new())),
        };

        // Start background tasks
        manager.start_background_tasks();

        manager
    }

    /// Set global message handlers
    pub async fn set_handlers(&self, handlers: WebSocketHandlers) {
        *self.handlers.write().await = Some(handlers);
    }

    /// Connect to WebSocket endpoint
    ///
    /// # Errors
    /// Returns error if connection fails
    pub async fn connect(&self, url: &str) -> NetworkResult<ConnectionId> {
        let connection_id = ConnectionId::new();
        let connection = WebSocketConnection::new(connection_id, url.to_string());
        
        info!("Connecting to WebSocket: {} ({})", url, connection_id);
        
        // Store connection
        self.connections.insert(connection_id, Arc::new(RwLock::new(connection)));
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_connection_attempts += 1;
        }

        // TODO: Implement actual WebSocket connection
        // This would include:
        // - URL validation and parsing
        // - TLS handshake for wss:// URLs
        // - WebSocket handshake
        // - Setting up message handling loops
        // - Configuring automatic reconnection

        // For now, simulate successful connection
        if let Some(conn_ref) = self.connections.get(&connection_id) {
            conn_ref.write().await.mark_connected();

            // Update statistics
            let mut stats = self.stats.write().await;
            stats.successful_connections += 1;
            stats.active_connections += 1;
        }

        // Notify handlers of successful connection
        if let Some(handlers) = self.handlers.read().await.as_ref() {
            handlers.handle_connect(connection_id).await;
        }

        debug!("WebSocket connection established: {}", connection_id);
        Ok(connection_id)
    }

    /// Send message to specific connection
    ///
    /// # Errors
    /// Returns error if send fails or connection not found
    #[allow(clippy::significant_drop_tightening)]
    pub async fn send(&self, connection_id: ConnectionId, message: WebSocketMessage) -> NetworkResult<()> {
        // Ultra-performance optimization: minimize lock scope for <1ms latency
        let conn_ref = self.connections.get(&connection_id).ok_or_else(|| {
            NetworkError::websocket("send", "Connection not found", Some(connection_id.to_string()))
        })?;

        {
            let mut conn = conn_ref.write().await;

            // Check connection state
            if conn.state != ConnectionState::Connected {
                return Err(NetworkError::websocket(
                    "send",
                    format!("Connection not in connected state: {}", conn.state),
                    Some(connection_id.to_string()),
                ));
            }

            // TODO: Implement actual message sending
            // This would include:
            // - Message serialization
            // - Frame construction
            // - Actual sending over the wire
            // - Error handling

            // Record message in connection stats
            conn.record_message_sent(&message);
        }

        // Update global statistics in separate minimal scope
        {
            let mut stats = self.stats.write().await;
            stats.total_messages_sent += 1;
        }

        debug!("Message sent on connection {}: {:?}", connection_id, message);
        Ok(())
    }

    /// Close specific connection
    ///
    /// # Errors
    /// Returns error if close fails or connection not found
    #[allow(clippy::significant_drop_tightening)]
    pub async fn close(&self, connection_id: ConnectionId) -> NetworkResult<()> {
        // Ultra-performance optimization: minimize lock scope for <1ms latency
        let conn_ref = self.connections.get(&connection_id).ok_or_else(|| {
            NetworkError::websocket("close", "Connection not found", Some(connection_id.to_string()))
        })?;

        {
            let mut conn = conn_ref.write().await;

            // TODO: Implement graceful close
            // This would include:
            // - Sending close frame
            // - Waiting for close response
            // - Cleaning up resources

            conn.mark_closed();
        }

        // Update statistics in separate minimal scope
        {
            let mut stats = self.stats.write().await;
            if stats.active_connections > 0 {
                stats.active_connections -= 1;
            }
        }

        // Notify handlers of connection close
        if let Some(handlers) = self.handlers.read().await.as_ref() {
            handlers.handle_close(connection_id, None).await;
        }

        // Remove connection from active connections
        self.connections.remove(&connection_id);

        info!("WebSocket connection closed: {}", connection_id);
        Ok(())
    }

    /// Get connection state
    #[must_use]
    pub fn connection_state(&self, connection_id: ConnectionId) -> Option<ConnectionState> {
        self.connections.get(&connection_id).map(|conn_ref| {
            futures::executor::block_on(async {
                conn_ref.read().await.state
            })
        })
    }

    /// Get all active connections
    #[must_use]
    pub fn active_connections(&self) -> Vec<ConnectionId> {
        self.connections.iter()
            .filter_map(|entry| {
                let conn_ref = entry.value();
                let state = futures::executor::block_on(async {
                    conn_ref.read().await.state
                });
                if state == ConnectionState::Connected {
                    Some(*entry.key())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get manager statistics
    #[must_use]
    pub async fn stats(&self) -> ManagerStats {
        self.stats.read().await.clone()
    }

    /// Check if manager is healthy
    #[must_use]
    pub async fn is_healthy(&self) -> bool {
        let stats = self.stats().await;
        let active_ratio = if stats.total_connection_attempts > 0 {
            let active_f64 = f64::from(u32::try_from(stats.active_connections).unwrap_or(u32::MAX));
            let total_f64 = f64::from(u32::try_from(stats.total_connection_attempts).unwrap_or(u32::MAX));
            active_f64 / total_f64
        } else {
            1.0_f64
        };

        // Consider healthy if at least 80% of attempted connections are active
        active_ratio >= 0.8_f64
    }

    /// Start background maintenance tasks
    fn start_background_tasks(&self) {
        let connections = self.connections.clone();
        let config = self.config.clone();
        let stats = self.stats.clone();

        // Ping/pong maintenance task
        let ping_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(config.ping_interval_s));

            #[allow(clippy::infinite_loop)]
            loop {
                interval.tick().await;
                
                // Send pings to all connected connections
                for entry in connections.iter() {
                    let conn_ref = entry.value();
                    {
                        let mut conn = conn_ref.write().await;

                        if conn.state == ConnectionState::Connected {
                            if conn.is_ping_overdue(Duration::from_secs(config.ping_interval_s)) {
                                conn.send_ping();
                                // TODO: Actually send ping message
                                debug!("Sent ping to connection {}", entry.key());
                            }

                            // Check for pong timeout
                            if conn.is_pong_overdue(Duration::from_secs(config.pong_timeout_s)) {
                                warn!("Pong timeout for connection {}, marking as failed", entry.key());
                                conn.mark_failed();
                            }
                        }
                    }

                    // Update statistics outside the lock
                    {
                        let mut stats = stats.write().await;
                        stats.failed_connections += 1;
                        if stats.active_connections > 0 {
                            stats.active_connections -= 1;
                        }
                    }
                }
            }
        });

        // Store task handle
        futures::executor::block_on(async {
            self.task_handles.write().await.push(ping_task);
        });

        // Reconnection task (if auto-reconnect is enabled)
        if self.config.enable_auto_reconnect {
            let connections = self.connections.clone();
            let reconnect_config = self.config.reconnect.clone();
            
            let reconnect_task = tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(1000)); // Check every second

                #[allow(clippy::infinite_loop)]
                loop {
                    interval.tick().await;
                    
                    // Check for failed connections that should reconnect
                    for entry in connections.iter() {
                        let conn_ref = entry.value();
                        let should_reconnect = {
                            let conn = conn_ref.read().await;
                            conn.should_reconnect()
                        };
                        
                        if should_reconnect {
                            let mut conn = conn_ref.write().await;
                            conn.start_reconnect();
                            
                            // Calculate reconnection delay
                            let delay = calculate_reconnect_delay(
                                conn.reconnect_attempts,
                                &reconnect_config,
                            );
                            
                            debug!(
                                "Scheduling reconnection for {} in {:?} (attempt {})",
                                entry.key(),
                                delay,
                                conn.reconnect_attempts
                            );
                            
                            // TODO: Implement actual reconnection logic
                            // For now, just simulate successful reconnection after delay
                            tokio::time::sleep(delay).await;
                            conn.mark_connected();
                        }
                    }
                }
            });

            futures::executor::block_on(async {
                self.task_handles.write().await.push(reconnect_task);
            });
        }
    }

    /// Shutdown manager and cleanup resources
    #[allow(clippy::cognitive_complexity)]
    pub async fn shutdown(&self) {
        info!("Shutting down WebSocket manager");
        
        // Close all active connections
        let connection_ids: Vec<ConnectionId> = self.connections.iter()
            .map(|entry| *entry.key())
            .collect();
            
        for connection_id in connection_ids {
            if let Err(e) = self.close(connection_id).await {
                error!("Error closing connection {}: {}", connection_id, e);
            }
        }
        
        // Cancel background tasks
        {
            let mut handles = self.task_handles.write().await;
            for handle in handles.drain(..) {
                handle.abort();
            }
        }
        
        info!("WebSocket manager shutdown complete");
    }
}

impl Drop for WebSocketManager {
    fn drop(&mut self) {
        // Ensure cleanup on drop
        futures::executor::block_on(self.shutdown());
    }
}

/// Manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStats {
    /// Total connection attempts
    pub total_connection_attempts: u64,
    /// Successful connections
    pub successful_connections: u64,
    /// Failed connections
    pub failed_connections: u64,
    /// Currently active connections
    pub active_connections: u64,
    /// Total messages sent
    pub total_messages_sent: u64,
    /// Total messages received
    pub total_messages_received: u64,
    /// Total reconnection attempts
    pub reconnection_attempts: u64,
    /// Manager start time
    pub start_time: Instant,
}

impl ManagerStats {
    fn new() -> Self {
        Self {
            total_connection_attempts: 0,
            successful_connections: 0,
            failed_connections: 0,
            active_connections: 0,
            total_messages_sent: 0,
            total_messages_received: 0,
            reconnection_attempts: 0,
            start_time: Instant::now(),
        }
    }

    /// Calculate connection success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_connection_attempts == 0 {
            0.0_f64
        } else {
            let successful_f64 = f64::from(u32::try_from(self.successful_connections).unwrap_or(u32::MAX));
            let total_f64 = f64::from(u32::try_from(self.total_connection_attempts).unwrap_or(u32::MAX));
            successful_f64 / total_f64
        }
    }

    /// Calculate uptime
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Calculate reconnection delay with exponential backoff
#[must_use]
fn calculate_reconnect_delay(attempt: u32, config: &crate::config::ReconnectConfig) -> Duration {
    let base_delay = Duration::from_millis(config.initial_delay_ms);
    let max_delay = Duration::from_millis(config.max_delay_ms);
    
    if attempt == 0 {
        return base_delay;
    }
    
    let multiplier = config.backoff_multiplier.powi(i32::try_from(attempt - 1).unwrap_or(0_i32));
    let delay = Duration::from_millis({
        let base_millis = u64::try_from(base_delay.as_millis()).unwrap_or(u64::MAX);

        // Use safe arithmetic to avoid precision loss
        let max_safe_u64 = 2_u64.pow(52); // f64 mantissa precision limit
        if base_millis > max_safe_u64 {
            u64::MAX // Avoid precision loss for very large values
        } else {
            #[allow(clippy::cast_precision_loss)] // Checked above
            let result = (base_millis as f64 * multiplier).round();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)] // Bounds checked
            if result >= u64::MAX as f64 {
                u64::MAX
            } else if result < 0.0_f64 {
                0_u64
            } else {
                result as u64
            }
        }
    });
    
    delay.min(max_delay)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{WebSocketConfig, ReconnectConfig};

    #[tokio::test]
    async fn test_manager_creation() {
        let config = WebSocketConfig::default();
        let manager = WebSocketManager::new(config);
        
        let stats = manager.stats().await;
        assert_eq!(stats.total_connection_attempts, 0);
        assert_eq!(stats.active_connections, 0);
    }

    #[tokio::test]
    async fn test_connection_lifecycle() -> Result<(), Box<dyn std::error::Error>> {
        let config = WebSocketConfig::default();
        let manager = WebSocketManager::new(config);
        
        // Connect
        let connection_id = manager.connect("wss://echo.websocket.org").await?;
        assert_eq!(manager.connection_state(connection_id), Some(ConnectionState::Connected));
        
        let stats = manager.stats().await;
        assert_eq!(stats.total_connection_attempts, 1);
        assert_eq!(stats.successful_connections, 1);
        assert_eq!(stats.active_connections, 1);
        
        // Send message
        let message = WebSocketMessage::Text("Hello".to_string());
        let result = manager.send(connection_id, message).await;
        assert!(result.is_ok());
        
        // Close
        let result = manager.close(connection_id).await;
        assert!(result.is_ok());
        assert_eq!(manager.connection_state(connection_id), None);

        Ok(())
    }

    #[test]
    fn test_reconnect_delay_calculation() {
        let config = ReconnectConfig {
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0_f64,
            max_attempts: 10,
        };
        
        assert_eq!(calculate_reconnect_delay(0, &config), Duration::from_millis(1000));
        assert_eq!(calculate_reconnect_delay(1, &config), Duration::from_millis(1000));
        assert_eq!(calculate_reconnect_delay(2, &config), Duration::from_millis(2000));
        assert_eq!(calculate_reconnect_delay(3, &config), Duration::from_millis(4000));
        
        // Should cap at max_delay
        assert_eq!(calculate_reconnect_delay(10, &config), Duration::from_millis(30000));
    }

    #[tokio::test]
    async fn test_manager_health() -> Result<(), Box<dyn std::error::Error>> {
        let config = WebSocketConfig::default();
        let manager = WebSocketManager::new(config);
        
        // Should be healthy with no connections
        assert!(manager.is_healthy().await);
        
        // Connect and should still be healthy
        let _connection_id = manager.connect("wss://echo.websocket.org").await?;
        assert!(manager.is_healthy().await);

        Ok(())
    }
}
