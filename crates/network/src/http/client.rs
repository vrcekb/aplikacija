//! HTTP Client Implementation
//!
//! High-performance HTTP client with connection pooling, retry logic, and circuit breaker.
//! Optimized for ultra-low latency financial applications with <1ms critical paths.

use crate::config::HttpConfig;
use crate::error::{NetworkError, NetworkResult};
use crate::http::{CircuitBreaker, HttpClientTrait, RetryPolicy};
use crate::http::circuit_breaker::CircuitBreakerConfig;
use crate::types::{HttpRequest, HttpResponse, HttpMethod};
use async_trait::async_trait;
use dashmap::DashMap;
use reqwest::{Client, Method, RequestBuilder};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};

/// High-performance HTTP client implementation
pub struct HttpClient {
    /// Underlying reqwest client
    client: Client,
    /// Configuration
    config: HttpConfig,
    /// Circuit breaker for fault tolerance
    circuit_breaker: Arc<CircuitBreaker>,
    /// Connection statistics per host
    host_stats: Arc<DashMap<String, HostStats>>,
    /// Global statistics
    global_stats: Arc<GlobalStats>,
    /// Active requests counter
    active_requests: Arc<AtomicU64>,
}

/// Statistics per host
#[derive(Debug, Clone)]
struct HostStats {
    /// Total requests to this host
    total_requests: AtomicU64,
    /// Successful requests
    successful_requests: AtomicU64,
    /// Failed requests
    failed_requests: AtomicU64,
    /// Average response time (microseconds)
    avg_response_time_us: AtomicU64,
    /// Last request timestamp
    last_request_at: RwLock<Option<Instant>>,
}

impl Default for HostStats {
    fn default() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            avg_response_time_us: AtomicU64::new(0),
            last_request_at: RwLock::new(None),
        }
    }
}

/// Global HTTP client statistics
#[derive(Debug)]
struct GlobalStats {
    /// Total requests across all hosts
    total_requests: AtomicU64,
    /// Total successful responses
    successful_responses: AtomicU64,
    /// Total failed requests
    failed_requests: AtomicU64,
    /// Average response time (microseconds)
    avg_response_time_us: AtomicU64,
    /// Client start time
    start_time: Instant,
}

impl Default for GlobalStats {
    fn default() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            successful_responses: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            avg_response_time_us: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }
}

impl HttpClient {
    /// Create new HTTP client
    ///
    /// # Errors
    /// Returns error if client configuration is invalid
    pub fn new(config: HttpConfig) -> NetworkResult<Self> {
        // Build reqwest client with optimized settings
        let mut client_builder = Client::builder()
            .pool_max_idle_per_host(config.max_connections_per_host as usize)
            .pool_idle_timeout(Duration::from_secs(config.keep_alive_timeout_s))
            .connect_timeout(Duration::from_secs(config.connection_timeout_s))
            .timeout(Duration::from_secs(config.request_timeout_s))
            .user_agent(&config.user_agent)
            .tcp_keepalive(Duration::from_secs(60))
            .tcp_nodelay(true); // Critical for low latency

        // Enable HTTP/2 if configured
        if config.enable_http2 {
            client_builder = client_builder.http2_prior_knowledge();
        }

        // Enable compression if configured
        if config.enable_compression {
            client_builder = client_builder.gzip(true).brotli(true);
        }

        let client = client_builder.build().map_err(|e| {
            NetworkError::config("http_client", format!("Failed to build HTTP client: {e}"))
        })?;

        // Create circuit breaker
        let circuit_breaker_config = CircuitBreakerConfig {
            failure_threshold: config.circuit_breaker.failure_threshold,
            success_threshold: config.circuit_breaker.success_threshold,
            timeout: Duration::from_secs(config.circuit_breaker.timeout_s),
            half_open_max_requests: config.circuit_breaker.half_open_max_calls,
            ..Default::default()
        };
        let circuit_breaker = Arc::new(CircuitBreaker::new(circuit_breaker_config));

        Ok(Self {
            client,
            config,
            circuit_breaker,
            host_stats: Arc::new(DashMap::new()),
            global_stats: Arc::new(GlobalStats::default()),
            active_requests: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Extract host from URL for statistics tracking
    fn extract_host(&self, url: &str) -> String {
        url::Url::parse(url)
            .map(|u| u.host_str().unwrap_or("unknown").to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    }

    /// Convert our HTTP method to reqwest method
    fn convert_method(&self, method: HttpMethod) -> Method {
        match method {
            HttpMethod::Get => Method::GET,
            HttpMethod::Post => Method::POST,
            HttpMethod::Put => Method::PUT,
            HttpMethod::Delete => Method::DELETE,
            HttpMethod::Patch => Method::PATCH,
            HttpMethod::Head => Method::HEAD,
            HttpMethod::Options => Method::OPTIONS,
        }
    }

    /// Build reqwest request from our request type
    fn build_request(&self, request: &HttpRequest) -> NetworkResult<RequestBuilder> {
        let method = self.convert_method(request.method);
        let mut req_builder = self.client.request(method, &request.url);

        // Add headers
        for (name, value) in &request.headers {
            req_builder = req_builder.header(name, value);
        }

        // Add body if present
        if let Some(body) = &request.body {
            req_builder = req_builder.body(body.clone());
        }

        // Set timeout if specified
        if let Some(timeout) = request.timeout {
            req_builder = req_builder.timeout(timeout);
        }

        Ok(req_builder)
    }

    /// Record request statistics
    fn record_request_stats(&self, host: &str, latency: Duration, success: bool) {
        let latency_us = latency.as_micros() as u64;

        // Update global stats
        self.global_stats.total_requests.fetch_add(1, Ordering::Relaxed);
        if success {
            self.global_stats.successful_responses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.global_stats.failed_requests.fetch_add(1, Ordering::Relaxed);
        }

        // Update global average latency (exponential moving average)
        let current_avg = self.global_stats.avg_response_time_us.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0 {
            latency_us
        } else {
            (current_avg * 9 + latency_us) / 10 // Alpha = 0.1
        };
        self.global_stats.avg_response_time_us.store(new_avg, Ordering::Relaxed);

        // Update host-specific stats
        let host_stats = self.host_stats.entry(host.to_string()).or_default();
        host_stats.total_requests.fetch_add(1, Ordering::Relaxed);
        
        if success {
            host_stats.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            host_stats.failed_requests.fetch_add(1, Ordering::Relaxed);
        }

        // Update host average latency
        let current_host_avg = host_stats.avg_response_time_us.load(Ordering::Relaxed);
        let new_host_avg = if current_host_avg == 0 {
            latency_us
        } else {
            (current_host_avg * 9 + latency_us) / 10
        };
        host_stats.avg_response_time_us.store(new_host_avg, Ordering::Relaxed);

        // Update last request time
        if let Ok(mut last_request) = host_stats.last_request_at.try_write() {
            *last_request = Some(Instant::now());
        }
    }

    /// Execute HTTP request with circuit breaker protection
    #[instrument(skip(self, request), fields(method = %request.method, url = %request.url))]
    async fn execute_request(&self, request: HttpRequest) -> NetworkResult<HttpResponse> {
        let host = self.extract_host(&request.url);
        let start_time = Instant::now();

        // Increment active requests counter
        self.active_requests.fetch_add(1, Ordering::Relaxed);

        let result = self.circuit_breaker.execute(|| async {
            let req_builder = self.build_request(&request)?;
            
            debug!("Sending HTTP request: {} {}", request.method, request.url);
            
            let response = req_builder.send().await.map_err(|e| {
                error!("HTTP request failed: {}", e);
                NetworkError::connection(&request.url, e.to_string(), None)
            })?;

            let status_code = response.status().as_u16();
            let mut headers = std::collections::HashMap::new();
            
            // Extract headers
            for (name, value) in response.headers() {
                if let Ok(value_str) = value.to_str() {
                    headers.insert(name.to_string(), value_str.to_string());
                }
            }

            // Read response body
            let body = response.bytes().await.map_err(|e| {
                error!("Failed to read response body: {}", e);
                NetworkError::http(status_code, e.to_string(), Some(request.url.clone()))
            })?.to_vec();

            let latency = start_time.elapsed();

            debug!(
                "HTTP response received: {} {} - {} bytes in {:?}",
                status_code,
                request.url,
                body.len(),
                latency
            );

            Ok(HttpResponse {
                status_code,
                headers,
                body,
                latency,
            })
        }).await;

        // Decrement active requests counter
        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        // Record statistics
        let latency = start_time.elapsed();
        let success = result.is_ok();
        self.record_request_stats(&host, latency, success);

        result
    }
}

#[async_trait]
impl HttpClientTrait for HttpClient {
    /// Send HTTP request
    async fn send(&self, request: HttpRequest) -> NetworkResult<HttpResponse> {
        self.execute_request(request).await
    }

    /// Send HTTP request with custom retry policy
    async fn send_with_retry(
        &self,
        request: HttpRequest,
        retry_policy: RetryPolicy,
    ) -> NetworkResult<HttpResponse> {
        use crate::http::retry::RetryExecutor;

        let request_clone = request.clone();
        RetryExecutor::new(
            move || {
                let req = request_clone.clone();
                let client_ref = self;
                async move { client_ref.execute_request(req).await }
            },
            retry_policy,
        )
        .execute()
        .await
    }

    /// Get client statistics
    fn stats(&self) -> crate::http::HttpClientStats {
        let pool_utilization = {
            let active = self.active_requests.load(Ordering::Relaxed) as f64;
            let max_per_host = self.config.max_connections_per_host as f64;
            let host_count = self.host_stats.len() as f64;
            let total_capacity = max_per_host * host_count.max(1.0_f64);
            if total_capacity > 0.0_f64 {
                active / total_capacity
            } else {
                0.0_f64
            }
        };

        crate::http::HttpClientStats {
            total_requests: self.global_stats.total_requests.load(Ordering::Relaxed),
            successful_responses: self.global_stats.successful_responses.load(Ordering::Relaxed),
            failed_requests: self.global_stats.failed_requests.load(Ordering::Relaxed),
            avg_response_time_us: self.global_stats.avg_response_time_us.load(Ordering::Relaxed),
            active_connections: self.active_requests.load(Ordering::Relaxed) as u32,
            pool_utilization,
            circuit_breaker_state: futures::executor::block_on(self.circuit_breaker.state()),
        }
    }

    /// Check if client is healthy
    fn is_healthy(&self) -> bool {
        let stats = self.stats();
        let circuit_breaker_ok = stats.circuit_breaker_state != crate::http::CircuitBreakerState::Open;
        let pool_not_exhausted = stats.pool_utilization < 0.9_f64;

        circuit_breaker_ok && pool_not_exhausted
    }
}

impl HttpClient {
    /// Get detailed statistics for a specific host
    pub fn host_stats(&self, host: &str) -> Option<HostStatsSnapshot> {
        self.host_stats.get(host).map(|stats| {
            let last_request_at = futures::executor::block_on(async {
                *stats.last_request_at.read().await
            });

            HostStatsSnapshot {
                host: host.to_string(),
                total_requests: stats.total_requests.load(Ordering::Relaxed),
                successful_requests: stats.successful_requests.load(Ordering::Relaxed),
                failed_requests: stats.failed_requests.load(Ordering::Relaxed),
                avg_response_time_us: stats.avg_response_time_us.load(Ordering::Relaxed),
                last_request_at,
            }
        })
    }

    /// Get statistics for all hosts
    pub fn all_host_stats(&self) -> Vec<HostStatsSnapshot> {
        self.host_stats
            .iter()
            .map(|entry| {
                let host = entry.key().clone();
                let stats = entry.value();
                let last_request_at = futures::executor::block_on(async {
                    *stats.last_request_at.read().await
                });

                HostStatsSnapshot {
                    host,
                    total_requests: stats.total_requests.load(Ordering::Relaxed),
                    successful_requests: stats.successful_requests.load(Ordering::Relaxed),
                    failed_requests: stats.failed_requests.load(Ordering::Relaxed),
                    avg_response_time_us: stats.avg_response_time_us.load(Ordering::Relaxed),
                    last_request_at,
                }
            })
            .collect()
    }

    /// Reset all statistics
    pub fn reset_stats(&self) {
        // Reset global stats
        self.global_stats.total_requests.store(0, Ordering::Relaxed);
        self.global_stats.successful_responses.store(0, Ordering::Relaxed);
        self.global_stats.failed_requests.store(0, Ordering::Relaxed);
        self.global_stats.avg_response_time_us.store(0, Ordering::Relaxed);

        // Reset host stats
        self.host_stats.clear();

        // Reset circuit breaker
        futures::executor::block_on(self.circuit_breaker.reset());
    }

    /// Get circuit breaker statistics
    pub async fn circuit_breaker_stats(&self) -> crate::http::circuit_breaker::CircuitBreakerStats {
        self.circuit_breaker.stats().await
    }

    /// Manually open circuit breaker (for testing/maintenance)
    pub async fn open_circuit_breaker(&self) {
        // Force circuit breaker to open by simulating failures
        for _ in 0..self.config.circuit_breaker.failure_threshold {
            let _ = self.circuit_breaker.execute(|| async {
                Err::<(), _>(NetworkError::internal("Manual circuit breaker open"))
            }).await;
        }
    }

    /// Get configuration
    pub fn config(&self) -> &HttpConfig {
        &self.config
    }

    /// Update configuration (creates new client instance)
    ///
    /// # Errors
    /// Returns error if new configuration is invalid
    pub fn with_config(&self, config: HttpConfig) -> NetworkResult<Self> {
        Self::new(config)
    }
}

/// Snapshot of host statistics
#[derive(Debug, Clone)]
pub struct HostStatsSnapshot {
    /// Host name
    pub host: String,
    /// Total requests to this host
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time (microseconds)
    pub avg_response_time_us: u64,
    /// Last request timestamp
    pub last_request_at: Option<Instant>,
}

impl HostStatsSnapshot {
    /// Calculate success rate (0.0 - 1.0)
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0_f64
        } else {
            self.successful_requests as f64 / self.total_requests as f64
        }
    }

    /// Calculate failure rate (0.0 - 1.0)
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        1.0_f64 - self.success_rate()
    }

    /// Check if host is considered healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.success_rate() > 0.95_f64 && self.avg_response_time_us < 10_000 // < 10ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CircuitBreakerConfig, RetryConfig};

    fn create_test_config() -> HttpConfig {
        HttpConfig {
            max_connections_per_host: 10,
            connection_timeout_s: 5,
            request_timeout_s: 10,
            keep_alive_timeout_s: 30,
            enable_http2: true,
            enable_http3: false,
            enable_compression: true,
            user_agent: "TallyIO-Test/1.0".to_string(),
            retry: RetryConfig::default(),
            circuit_breaker: CircuitBreakerConfig::default(),
        }
    }

    #[test]
    fn test_http_client_creation() {
        let config = create_test_config();
        let client = HttpClient::new(config);
        assert!(client.is_ok());
    }

    #[test]
    fn test_method_conversion() {
        let config = create_test_config();
        let client = HttpClient::new(config).unwrap();

        assert_eq!(client.convert_method(HttpMethod::Get), Method::GET);
        assert_eq!(client.convert_method(HttpMethod::Post), Method::POST);
        assert_eq!(client.convert_method(HttpMethod::Put), Method::PUT);
        assert_eq!(client.convert_method(HttpMethod::Delete), Method::DELETE);
    }

    #[test]
    fn test_host_extraction() {
        let config = create_test_config();
        let client = HttpClient::new(config).unwrap();

        assert_eq!(client.extract_host("https://api.example.com/path"), "api.example.com");
        assert_eq!(client.extract_host("http://localhost:8080"), "localhost");
        assert_eq!(client.extract_host("invalid-url"), "unknown");
    }

    #[test]
    fn test_stats_initialization() {
        let config = create_test_config();
        let client = HttpClient::new(config).unwrap();
        let stats = client.stats();

        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_responses, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.active_connections, 0);
        assert_eq!(stats.pool_utilization, 0.0_f64);
    }

    #[test]
    fn test_host_stats_snapshot() {
        let snapshot = HostStatsSnapshot {
            host: "api.example.com".to_string(),
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            avg_response_time_us: 5000,
            last_request_at: Some(Instant::now()),
        };

        assert_eq!(snapshot.success_rate(), 0.95_f64);
        assert_eq!(snapshot.failure_rate(), 0.05_f64);
        assert!(snapshot.is_healthy());
    }
}
