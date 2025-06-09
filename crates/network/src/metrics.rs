//! Network Metrics Module
//!
//! Comprehensive metrics collection for network operations.
//! Provides detailed insights into performance, reliability, and resource utilization.

use crate::error::NetworkResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Network metrics collector
pub struct NetworkMetrics {
    /// HTTP metrics
    http_metrics: Arc<HttpMetrics>,
    /// WebSocket metrics
    websocket_metrics: Arc<WebSocketMetrics>,
    /// Connection metrics
    connection_metrics: Arc<ConnectionMetrics>,
    /// Performance metrics
    performance_metrics: Arc<PerformanceMetrics>,
    /// Error metrics
    error_metrics: Arc<ErrorMetrics>,
    /// Start time for uptime calculation
    start_time: Instant,
}

impl NetworkMetrics {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            http_metrics: Arc::new(HttpMetrics::new()),
            websocket_metrics: Arc::new(WebSocketMetrics::new()),
            connection_metrics: Arc::new(ConnectionMetrics::new()),
            performance_metrics: Arc::new(PerformanceMetrics::new()),
            error_metrics: Arc::new(ErrorMetrics::new()),
            start_time: Instant::now(),
        }
    }

    /// Get HTTP metrics
    pub fn http(&self) -> &HttpMetrics {
        &self.http_metrics
    }

    /// Get WebSocket metrics
    pub fn websocket(&self) -> &WebSocketMetrics {
        &self.websocket_metrics
    }

    /// Get connection metrics
    pub fn connection(&self) -> &ConnectionMetrics {
        &self.connection_metrics
    }

    /// Get performance metrics
    pub fn performance(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Get error metrics
    pub fn error(&self) -> &ErrorMetrics {
        &self.error_metrics
    }

    /// Get comprehensive metrics snapshot
    pub fn snapshot(&self) -> NetworkMetricsSnapshot {
        NetworkMetricsSnapshot {
            timestamp: SystemTime::now(),
            uptime: self.start_time.elapsed(),
            http: self.http_metrics.snapshot(),
            websocket: self.websocket_metrics.snapshot(),
            connection: self.connection_metrics.snapshot(),
            performance: self.performance_metrics.snapshot(),
            error: self.error_metrics.snapshot(),
        }
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.http_metrics.reset();
        self.websocket_metrics.reset();
        self.connection_metrics.reset();
        self.performance_metrics.reset();
        self.error_metrics.reset();
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// HTTP-specific metrics
pub struct HttpMetrics {
    /// Total HTTP requests
    total_requests: AtomicU64,
    /// Successful responses (2xx)
    successful_responses: AtomicU64,
    /// Client errors (4xx)
    client_errors: AtomicU64,
    /// Server errors (5xx)
    server_errors: AtomicU64,
    /// Total bytes sent
    bytes_sent: AtomicU64,
    /// Total bytes received
    bytes_received: AtomicU64,
    /// Request latency histogram
    latency_histogram: parking_lot::Mutex<Vec<u64>>,
}

impl HttpMetrics {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_responses: AtomicU64::new(0),
            client_errors: AtomicU64::new(0),
            server_errors: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            latency_histogram: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Record HTTP request
    pub fn record_request(&self, status_code: u16, latency: Duration, bytes_sent: u64, bytes_received: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes_sent, Ordering::Relaxed);
        self.bytes_received.fetch_add(bytes_received, Ordering::Relaxed);

        // Categorize by status code
        match status_code {
            200..=299 => { self.successful_responses.fetch_add(1, Ordering::Relaxed); }
            400..=499 => { self.client_errors.fetch_add(1, Ordering::Relaxed); }
            500..=599 => { self.server_errors.fetch_add(1, Ordering::Relaxed); }
            _ => {} // Other status codes
        }

        // Record latency
        if let Ok(mut histogram) = self.latency_histogram.try_lock() {
            histogram.push(latency.as_micros() as u64);
            // Keep only last 1000 samples
            if histogram.len() > 1000 {
                histogram.remove(0);
            }
        }
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> HttpMetricsSnapshot {
        let latencies = self.latency_histogram.lock().clone();
        let (avg_latency, p95_latency, p99_latency) = calculate_latency_percentiles(&latencies);

        HttpMetricsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            successful_responses: self.successful_responses.load(Ordering::Relaxed),
            client_errors: self.client_errors.load(Ordering::Relaxed),
            server_errors: self.server_errors.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            avg_latency_us: avg_latency,
            p95_latency_us: p95_latency,
            p99_latency_us: p99_latency,
        }
    }

    fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_responses.store(0, Ordering::Relaxed);
        self.client_errors.store(0, Ordering::Relaxed);
        self.server_errors.store(0, Ordering::Relaxed);
        self.bytes_sent.store(0, Ordering::Relaxed);
        self.bytes_received.store(0, Ordering::Relaxed);
        self.latency_histogram.lock().clear();
    }
}

/// WebSocket-specific metrics
pub struct WebSocketMetrics {
    /// Active connections
    active_connections: AtomicU64,
    /// Total connection attempts
    connection_attempts: AtomicU64,
    /// Successful connections
    successful_connections: AtomicU64,
    /// Failed connections
    failed_connections: AtomicU64,
    /// Messages sent
    messages_sent: AtomicU64,
    /// Messages received
    messages_received: AtomicU64,
    /// Reconnection attempts
    reconnection_attempts: AtomicU64,
}

impl WebSocketMetrics {
    fn new() -> Self {
        Self {
            active_connections: AtomicU64::new(0),
            connection_attempts: AtomicU64::new(0),
            successful_connections: AtomicU64::new(0),
            failed_connections: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            reconnection_attempts: AtomicU64::new(0),
        }
    }

    /// Record connection attempt
    pub fn record_connection_attempt(&self) {
        self.connection_attempts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record successful connection
    pub fn record_connection_success(&self) {
        self.successful_connections.fetch_add(1, Ordering::Relaxed);
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record failed connection
    pub fn record_connection_failure(&self) {
        self.failed_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record connection close
    pub fn record_connection_close(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record message sent
    pub fn record_message_sent(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Record message received
    pub fn record_message_received(&self) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }

    /// Record reconnection attempt
    pub fn record_reconnection_attempt(&self) {
        self.reconnection_attempts.fetch_add(1, Ordering::Relaxed);
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> WebSocketMetricsSnapshot {
        WebSocketMetricsSnapshot {
            active_connections: self.active_connections.load(Ordering::Relaxed),
            connection_attempts: self.connection_attempts.load(Ordering::Relaxed),
            successful_connections: self.successful_connections.load(Ordering::Relaxed),
            failed_connections: self.failed_connections.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            reconnection_attempts: self.reconnection_attempts.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.active_connections.store(0, Ordering::Relaxed);
        self.connection_attempts.store(0, Ordering::Relaxed);
        self.successful_connections.store(0, Ordering::Relaxed);
        self.failed_connections.store(0, Ordering::Relaxed);
        self.messages_sent.store(0, Ordering::Relaxed);
        self.messages_received.store(0, Ordering::Relaxed);
        self.reconnection_attempts.store(0, Ordering::Relaxed);
    }
}

/// Connection pool metrics
pub struct ConnectionMetrics {
    /// Active connections
    active_connections: AtomicU64,
    /// Pool utilization percentage
    pool_utilization: parking_lot::Mutex<f64>,
    /// Connection timeouts
    connection_timeouts: AtomicU64,
    /// Connection errors
    connection_errors: AtomicU64,
}

impl ConnectionMetrics {
    fn new() -> Self {
        Self {
            active_connections: AtomicU64::new(0),
            pool_utilization: parking_lot::Mutex::new(0.0_f64),
            connection_timeouts: AtomicU64::new(0),
            connection_errors: AtomicU64::new(0),
        }
    }

    /// Update pool utilization
    pub fn update_pool_utilization(&self, utilization: f64) {
        *self.pool_utilization.lock() = utilization;
    }

    /// Record connection timeout
    pub fn record_timeout(&self) {
        self.connection_timeouts.fetch_add(1, Ordering::Relaxed);
    }

    /// Record connection error
    pub fn record_error(&self) {
        self.connection_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> ConnectionMetricsSnapshot {
        ConnectionMetricsSnapshot {
            active_connections: self.active_connections.load(Ordering::Relaxed),
            pool_utilization: *self.pool_utilization.lock(),
            connection_timeouts: self.connection_timeouts.load(Ordering::Relaxed),
            connection_errors: self.connection_errors.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.active_connections.store(0, Ordering::Relaxed);
        *self.pool_utilization.lock() = 0.0_f64;
        self.connection_timeouts.store(0, Ordering::Relaxed);
        self.connection_errors.store(0, Ordering::Relaxed);
    }
}

/// Performance metrics
pub struct PerformanceMetrics {
    /// CPU usage percentage
    cpu_usage: parking_lot::Mutex<f64>,
    /// Memory usage in bytes
    memory_usage: AtomicU64,
    /// Network throughput (bytes per second)
    network_throughput: parking_lot::Mutex<f64>,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            cpu_usage: parking_lot::Mutex::new(0.0_f64),
            memory_usage: AtomicU64::new(0),
            network_throughput: parking_lot::Mutex::new(0.0_f64),
        }
    }

    /// Update CPU usage
    pub fn update_cpu_usage(&self, usage: f64) {
        *self.cpu_usage.lock() = usage;
    }

    /// Update memory usage
    pub fn update_memory_usage(&self, usage: u64) {
        self.memory_usage.store(usage, Ordering::Relaxed);
    }

    /// Update network throughput
    pub fn update_network_throughput(&self, throughput: f64) {
        *self.network_throughput.lock() = throughput;
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> PerformanceMetricsSnapshot {
        PerformanceMetricsSnapshot {
            cpu_usage: *self.cpu_usage.lock(),
            memory_usage: self.memory_usage.load(Ordering::Relaxed),
            network_throughput: *self.network_throughput.lock(),
        }
    }

    fn reset(&self) {
        *self.cpu_usage.lock() = 0.0_f64;
        self.memory_usage.store(0, Ordering::Relaxed);
        *self.network_throughput.lock() = 0.0_f64;
    }
}

/// Error tracking metrics
pub struct ErrorMetrics {
    /// Error counts by type
    error_counts: parking_lot::Mutex<HashMap<String, u64>>,
    /// Total errors
    total_errors: AtomicU64,
}

impl ErrorMetrics {
    fn new() -> Self {
        Self {
            error_counts: parking_lot::Mutex::new(HashMap::new()),
            total_errors: AtomicU64::new(0),
        }
    }

    /// Record error
    pub fn record_error(&self, error_type: &str) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        let mut counts = self.error_counts.lock();
        *counts.entry(error_type.to_string()).or_insert(0) += 1;
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> ErrorMetricsSnapshot {
        ErrorMetricsSnapshot {
            total_errors: self.total_errors.load(Ordering::Relaxed),
            error_counts: self.error_counts.lock().clone(),
        }
    }

    fn reset(&self) {
        self.total_errors.store(0, Ordering::Relaxed);
        self.error_counts.lock().clear();
    }
}

/// Calculate latency percentiles
fn calculate_latency_percentiles(latencies: &[u64]) -> (u64, u64, u64) {
    if latencies.is_empty() {
        return (0, 0, 0);
    }

    let mut sorted = latencies.to_vec();
    sorted.sort_unstable();

    let avg = sorted.iter().sum::<u64>() / sorted.len() as u64;
    let p95_index = (sorted.len() as f64 * 0.95_f64) as usize;
    let p99_index = (sorted.len() as f64 * 0.99_f64) as usize;

    let p95 = sorted.get(p95_index.saturating_sub(1)).copied().unwrap_or(0);
    let p99 = sorted.get(p99_index.saturating_sub(1)).copied().unwrap_or(0);

    (avg, p95, p99)
}

// Snapshot types for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetricsSnapshot {
    pub timestamp: SystemTime,
    pub uptime: Duration,
    pub http: HttpMetricsSnapshot,
    pub websocket: WebSocketMetricsSnapshot,
    pub connection: ConnectionMetricsSnapshot,
    pub performance: PerformanceMetricsSnapshot,
    pub error: ErrorMetricsSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpMetricsSnapshot {
    pub total_requests: u64,
    pub successful_responses: u64,
    pub client_errors: u64,
    pub server_errors: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub avg_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMetricsSnapshot {
    pub active_connections: u64,
    pub connection_attempts: u64,
    pub successful_connections: u64,
    pub failed_connections: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub reconnection_attempts: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionMetricsSnapshot {
    pub active_connections: u64,
    pub pool_utilization: f64,
    pub connection_timeouts: u64,
    pub connection_errors: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsSnapshot {
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub network_throughput: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetricsSnapshot {
    pub total_errors: u64,
    pub error_counts: HashMap<String, u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = NetworkMetrics::new();
        let snapshot = metrics.snapshot();
        
        assert_eq!(snapshot.http.total_requests, 0);
        assert_eq!(snapshot.websocket.active_connections, 0);
        assert_eq!(snapshot.error.total_errors, 0);
    }

    #[test]
    fn test_http_metrics() {
        let metrics = HttpMetrics::new();
        
        // Record some requests
        metrics.record_request(200, Duration::from_millis(100), 1024, 2048);
        metrics.record_request(404, Duration::from_millis(50), 512, 1024);
        metrics.record_request(500, Duration::from_millis(200), 256, 512);
        
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_requests, 3);
        assert_eq!(snapshot.successful_responses, 1);
        assert_eq!(snapshot.client_errors, 1);
        assert_eq!(snapshot.server_errors, 1);
        assert_eq!(snapshot.bytes_sent, 1792);
        assert_eq!(snapshot.bytes_received, 3584);
    }

    #[test]
    fn test_websocket_metrics() {
        let metrics = WebSocketMetrics::new();
        
        metrics.record_connection_attempt();
        metrics.record_connection_success();
        metrics.record_message_sent();
        metrics.record_message_received();
        
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.connection_attempts, 1);
        assert_eq!(snapshot.successful_connections, 1);
        assert_eq!(snapshot.active_connections, 1);
        assert_eq!(snapshot.messages_sent, 1);
        assert_eq!(snapshot.messages_received, 1);
    }

    #[test]
    fn test_error_metrics() {
        let metrics = ErrorMetrics::new();
        
        metrics.record_error("connection_timeout");
        metrics.record_error("connection_timeout");
        metrics.record_error("parse_error");
        
        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_errors, 3);
        assert_eq!(snapshot.error_counts.get("connection_timeout"), Some(&2));
        assert_eq!(snapshot.error_counts.get("parse_error"), Some(&1));
    }

    #[test]
    fn test_latency_percentiles() {
        let latencies = vec![100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        let (avg, p95, p99) = calculate_latency_percentiles(&latencies);
        
        assert_eq!(avg, 550); // Average
        assert_eq!(p95, 900);  // 95th percentile
        assert_eq!(p99, 1000); // 99th percentile
    }
}
