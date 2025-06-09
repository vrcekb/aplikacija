//! Network Manager
//!
//! Central orchestrator for all network operations in TallyIO.
//! Provides unified interface for HTTP, WebSocket, and P2P networking.

use crate::config::NetworkConfig;
use crate::error::{NetworkError, NetworkResult};
use crate::http::{HttpClient, HttpClientTrait};
use crate::load_balancer::{LoadBalancer, LoadBalancerTrait};
use crate::metrics::NetworkMetrics;
use crate::p2p::{P2PNetwork, P2PNetworkTrait};
use crate::types::{ConnectionId, Endpoint, HttpRequest, HttpResponse, WebSocketMessage};
use crate::websocket::{WebSocketClient as WsClient, WebSocketClientTrait};
use std::sync::Arc;
use tallyio_core::prelude::*;
use tracing::{debug, error, info, instrument, warn};

/// Network manager - central orchestrator for all network operations
pub struct NetworkManager {
    /// Configuration
    config: NetworkConfig,
    /// HTTP client
    http_client: Arc<HttpClient>,
    /// WebSocket client
    websocket_client: Arc<WebSocketClientStub>,
    /// Load balancer
    load_balancer: Arc<LoadBalancer>,
    /// P2P network (optional)
    p2p_network: Option<Arc<P2PNetwork>>,
    /// Metrics collector
    metrics: Arc<NetworkMetrics>,
    /// Component status
    status: ComponentStatus,
}

impl NetworkManager {
    /// Create new network manager
    ///
    /// # Errors
    /// Returns error if configuration is invalid or components fail to initialize
    #[instrument(skip(config))]
    pub async fn new(config: NetworkConfig) -> NetworkResult<Self> {
        info!("Initializing TallyIO Network Manager");

        // Validate configuration for production
        config.validate_production().map_err(|e| {
            error!("Network configuration validation failed: {}", e);
            e
        })?;

        // Initialize HTTP client
        let http_client = Arc::new(HttpClient::new(config.http.clone()).map_err(|e| {
            error!("Failed to initialize HTTP client: {}", e);
            e
        })?);
        debug!("HTTP client initialized");

        // Initialize WebSocket client (stub for now)
        let websocket_client = Arc::new(WebSocketClientStub::new(config.websocket.clone()).map_err(|e| {
            error!("Failed to initialize WebSocket client: {}", e);
            e
        })?);
        debug!("WebSocket client initialized");

        // Initialize load balancer
        let load_balancer = Arc::new(LoadBalancer::new(
            config.load_balancer.strategy,
            config.load_balancer.endpoints.clone(),
        ));
        debug!("Load balancer initialized with {} endpoints", config.load_balancer.endpoints.len());

        // Initialize P2P network if configured
        let p2p_network = if let Some(p2p_config) = &config.p2p {
            let p2p_net = P2PNetwork::new(p2p_config.clone());
            Some(Arc::new(p2p_net))
        } else {
            None
        };

        if p2p_network.is_some() {
            debug!("P2P network initialized");
        }

        // Initialize metrics
        let metrics = Arc::new(NetworkMetrics::new());
        debug!("Network metrics initialized");

        info!("TallyIO Network Manager initialized successfully");

        Ok(Self {
            config,
            http_client,
            websocket_client,
            load_balancer,
            p2p_network,
            metrics,
            status: ComponentStatus::Stopped,
        })
    }

    /// Get HTTP client
    pub fn http_client(&self) -> &Arc<HttpClient> {
        &self.http_client
    }

    /// Get WebSocket client
    pub fn websocket_client(&self) -> &Arc<WebSocketClientStub> {
        &self.websocket_client
    }

    /// Get load balancer
    pub fn load_balancer(&self) -> &Arc<LoadBalancer> {
        &self.load_balancer
    }

    /// Get P2P network (if available)
    pub fn p2p_network(&self) -> Option<&Arc<P2PNetwork>> {
        self.p2p_network.as_ref()
    }

    /// Get metrics
    pub fn metrics(&self) -> &Arc<NetworkMetrics> {
        &self.metrics
    }

    /// Get configuration
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }

    /// Send HTTP request using load balancer
    ///
    /// # Errors
    /// Returns error if request fails or no healthy endpoints available
    #[instrument(skip(self, request))]
    pub async fn send_http_request(&self, mut request: HttpRequest) -> NetworkResult<HttpResponse> {
        // If URL is relative, use load balancer to select endpoint
        if !request.url.starts_with("http://") && !request.url.starts_with("https://") {
            let endpoint = self.load_balancer.select_endpoint().await?;
            let base_url = endpoint.url.trim_end_matches('/');
            let path = request.url.trim_start_matches('/');
            request.url = format!("{base_url}/{path}");
            debug!("Using load-balanced endpoint: {}", endpoint.url);
        }

        // Send request through HTTP client
        let response = self.http_client.send(request).await?;

        // Record metrics
        self.metrics.http().record_request(
            response.status_code,
            response.latency,
            0, // bytes_sent - would need to track from request
            response.body.len() as u64,
        );

        Ok(response)
    }

    /// Connect to WebSocket endpoint using load balancer
    ///
    /// # Errors
    /// Returns error if connection fails or no healthy endpoints available
    #[instrument(skip(self))]
    pub async fn connect_websocket(&self, path: &str) -> NetworkResult<ConnectionId> {
        // Select endpoint from load balancer
        let endpoint = self.load_balancer.select_endpoint().await?;
        
        // Convert HTTP(S) URL to WebSocket URL
        let ws_url = if endpoint.url.starts_with("https://") {
            endpoint.url.replace("https://", "wss://")
        } else if endpoint.url.starts_with("http://") {
            endpoint.url.replace("http://", "ws://")
        } else {
            endpoint.url.clone()
        };

        let full_url = format!("{}/{}", ws_url.trim_end_matches('/'), path.trim_start_matches('/'));
        debug!("Connecting to WebSocket: {}", full_url);

        // Connect through WebSocket client
        let connection_id = self.websocket_client.connect(&full_url).await?;

        // Record metrics
        self.metrics.websocket().record_connection_attempt();
        self.metrics.websocket().record_connection_success();

        Ok(connection_id)
    }

    /// Send WebSocket message
    ///
    /// # Errors
    /// Returns error if send fails or connection not found
    pub async fn send_websocket_message(
        &self,
        connection_id: ConnectionId,
        message: WebSocketMessage,
    ) -> NetworkResult<()> {
        let result = self.websocket_client.send(connection_id, message).await;

        // Record metrics
        if result.is_ok() {
            self.metrics.websocket().record_message_sent();
        } else {
            self.metrics.error().record_error("websocket_send_failed");
        }

        result
    }

    /// Close WebSocket connection
    ///
    /// # Errors
    /// Returns error if close fails or connection not found
    pub async fn close_websocket(&self, connection_id: ConnectionId) -> NetworkResult<()> {
        let result = self.websocket_client.close(connection_id).await;

        // Record metrics
        if result.is_ok() {
            self.metrics.websocket().record_connection_close();
        }

        result
    }

    /// Add endpoint to load balancer
    pub fn add_endpoint(&self, endpoint: Endpoint) {
        self.load_balancer.add_endpoint(endpoint);
    }

    /// Remove endpoint from load balancer
    pub fn remove_endpoint(&self, url: &str) {
        self.load_balancer.remove_endpoint(url);
    }

    /// Get comprehensive network statistics
    pub fn network_stats(&self) -> NetworkStats {
        NetworkStats {
            http: self.http_client.stats(),
            websocket: self.websocket_client.stats(),
            load_balancer: self.load_balancer.stats(),
            p2p: self.p2p_network.as_ref().map(|p2p| p2p.stats()),
            metrics: self.metrics.snapshot(),
        }
    }

    /// Check overall network health
    pub fn is_healthy(&self) -> bool {
        let http_healthy = self.http_client.is_healthy();
        let websocket_healthy = self.websocket_client.is_healthy();
        let load_balancer_healthy = self.load_balancer.is_healthy();

        http_healthy && websocket_healthy && load_balancer_healthy
    }

    /// Reset all network statistics
    pub fn reset_stats(&self) {
        self.http_client.reset_stats();
        self.metrics.reset();
        info!("Network statistics reset");
    }
}

#[async_trait]
impl Lifecycle for NetworkManager {
    async fn start(&mut self) -> CoreResult<()> {
        if self.status == ComponentStatus::Running {
            return Ok(());
        }

        info!("Starting TallyIO Network Manager");
        self.status = ComponentStatus::Starting;

        // Start P2P network if configured
        if let Some(p2p_network) = &self.p2p_network {
            if let Ok(mut p2p) = Arc::try_unwrap(p2p_network.clone()) {
                p2p.start().await.map_err(|e| {
                    error!("Failed to start P2P network: {}", e);
                    CoreError::engine("network_manager", e.to_string())
                })?;
            }
        }

        self.status = ComponentStatus::Running;
        info!("TallyIO Network Manager started successfully");
        Ok(())
    }

    async fn stop(&mut self) -> CoreResult<()> {
        if self.status == ComponentStatus::Stopped {
            return Ok(());
        }

        info!("Stopping TallyIO Network Manager");
        self.status = ComponentStatus::Stopping;

        // Stop P2P network if running
        if let Some(p2p_network) = &self.p2p_network {
            if let Ok(mut p2p) = Arc::try_unwrap(p2p_network.clone()) {
                p2p.stop().await.map_err(|e| {
                    error!("Failed to stop P2P network: {}", e);
                    CoreError::engine("network_manager", e.to_string())
                })?;
            }
        }

        self.status = ComponentStatus::Stopped;
        info!("TallyIO Network Manager stopped successfully");
        Ok(())
    }

    fn is_running(&self) -> bool {
        self.status == ComponentStatus::Running
    }

    fn status(&self) -> ComponentStatus {
        self.status
    }
}

impl HealthCheck for NetworkManager {
    fn health_check(&self) -> CoreResult<HealthStatus> {
        let is_healthy = self.is_healthy();
        let stats = self.network_stats();

        let health_level = if is_healthy {
            HealthLevel::Healthy
        } else {
            HealthLevel::Unhealthy
        };

        let details = if is_healthy {
            Some("All network components operational".to_string())
        } else {
            Some("One or more network components unhealthy".to_string())
        };

        let metrics = Some(HealthMetrics {
            cpu_usage: 0.0_f64, // Would be populated from system metrics
            memory_usage: 0,     // Would be populated from system metrics
            active_connections: stats.http.active_connections + stats.websocket.active_connections,
            request_rate: 0.0_f64, // Would be calculated from metrics
            error_rate: 0.0_f64,   // Would be calculated from metrics
            avg_response_time_us: stats.http.avg_response_time_us,
        });

        Ok(HealthStatus {
            component: "network_manager".to_string(),
            status: health_level,
            timestamp: std::time::SystemTime::now(),
            details,
            metrics,
        })
    }
}

/// Comprehensive network statistics
#[derive(Debug, Clone)]
pub struct NetworkStats {
    /// HTTP client statistics
    pub http: crate::http::HttpClientStats,
    /// WebSocket client statistics
    pub websocket: crate::websocket::WebSocketClientStats,
    /// Load balancer statistics
    pub load_balancer: crate::load_balancer::LoadBalancerStats,
    /// P2P network statistics (if available)
    pub p2p: Option<crate::p2p::P2PStats>,
    /// Detailed metrics
    pub metrics: crate::metrics::NetworkMetricsSnapshot,
}

// Stub implementations for WebSocketClient
pub struct WebSocketClientStub {
    _config: crate::config::WebSocketConfig,
}

impl WebSocketClientStub {
    pub fn new(config: crate::config::WebSocketConfig) -> NetworkResult<Self> {
        Ok(Self { _config: config })
    }

    pub fn stats(&self) -> crate::websocket::WebSocketClientStats {
        crate::websocket::WebSocketClientStats::default()
    }

    pub fn is_healthy(&self) -> bool {
        true
    }
}

#[async_trait]
impl WebSocketClientTrait for WebSocketClientStub {
    async fn connect(&self, _url: &str) -> NetworkResult<ConnectionId> {
        Ok(ConnectionId::new())
    }

    async fn send(&self, _connection_id: ConnectionId, _message: WebSocketMessage) -> NetworkResult<()> {
        Ok(())
    }

    async fn close(&self, _connection_id: ConnectionId) -> NetworkResult<()> {
        Ok(())
    }

    fn connection_state(&self, _connection_id: ConnectionId) -> Option<crate::websocket::ConnectionState> {
        None
    }

    fn stats(&self) -> crate::websocket::WebSocketClientStats {
        crate::websocket::WebSocketClientStats::default()
    }

    fn is_healthy(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{HttpConfig, LoadBalancerConfig, SecurityConfig, MetricsConfig};
    use crate::types::Endpoint;

    fn create_test_config() -> NetworkConfig {
        NetworkConfig {
            http: HttpConfig::default(),
            websocket: crate::config::WebSocketConfig::default(),
            load_balancer: LoadBalancerConfig {
                endpoints: vec![Endpoint::new("https://api.example.com")],
                ..Default::default()
            },
            p2p: None,
            security: SecurityConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_network_manager_creation() {
        let config = create_test_config();
        let manager = NetworkManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_network_manager_lifecycle() {
        let config = create_test_config();
        let mut manager = NetworkManager::new(config).await.unwrap();
        
        assert!(!manager.is_running());
        
        let result = manager.start().await;
        assert!(result.is_ok());
        assert!(manager.is_running());
        
        let result = manager.stop().await;
        assert!(result.is_ok());
        assert!(!manager.is_running());
    }

    #[test]
    fn test_network_stats() {
        let config = create_test_config();
        let manager = futures::executor::block_on(NetworkManager::new(config)).unwrap();
        
        let stats = manager.network_stats();
        assert_eq!(stats.http.total_requests, 0);
        assert_eq!(stats.websocket.active_connections, 0);
        assert_eq!(stats.load_balancer.total_endpoints, 1);
    }

    #[test]
    fn test_health_check() {
        let config = create_test_config();
        let manager = futures::executor::block_on(NetworkManager::new(config)).unwrap();
        
        let health = manager.health_check().unwrap();
        assert_eq!(health.component, "network_manager");
        assert!(matches!(health.status, HealthLevel::Healthy));
    }
}
