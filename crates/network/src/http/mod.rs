//! HTTP Client Module
//!
//! High-performance HTTP client with connection pooling, retry logic, and circuit breaker.
//! Optimized for ultra-low latency financial applications.

pub mod client;
pub mod retry;
pub mod circuit_breaker;

pub use client::HttpClient;
pub use retry::{RetryPolicy, RetryState};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerState};

use crate::error::NetworkResult;
use crate::types::{HttpRequest, HttpResponse};
use async_trait::async_trait;

/// HTTP client trait for testability and flexibility
#[async_trait]
pub trait HttpClientTrait: Send + Sync {
    /// Send HTTP request
    ///
    /// # Errors
    /// Returns error if request fails or times out
    async fn send(&self, request: HttpRequest) -> NetworkResult<HttpResponse>;

    /// Send HTTP request with custom retry policy
    ///
    /// # Errors
    /// Returns error if all retry attempts fail
    async fn send_with_retry(
        &self,
        request: HttpRequest,
        retry_policy: RetryPolicy,
    ) -> NetworkResult<HttpResponse>;

    /// Get client statistics
    fn stats(&self) -> HttpClientStats;

    /// Check if client is healthy
    fn is_healthy(&self) -> bool;
}

/// HTTP client statistics
#[derive(Debug, Clone)]
pub struct HttpClientStats {
    /// Total requests sent
    pub total_requests: u64,
    /// Total successful responses
    pub successful_responses: u64,
    /// Total failed requests
    pub failed_requests: u64,
    /// Average response time (microseconds)
    pub avg_response_time_us: u64,
    /// Active connections count
    pub active_connections: u32,
    /// Connection pool utilization (0.0 - 1.0)
    pub pool_utilization: f64,
    /// Circuit breaker state
    pub circuit_breaker_state: CircuitBreakerState,
}

impl Default for HttpClientStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_responses: 0,
            failed_requests: 0,
            avg_response_time_us: 0,
            active_connections: 0,
            pool_utilization: 0.0_f64,
            circuit_breaker_state: CircuitBreakerState::Closed,
        }
    }
}

/// HTTP request builder with fluent API
pub struct HttpRequestBuilder {
    request: HttpRequest,
    retry_policy: Option<RetryPolicy>,
    circuit_breaker_enabled: bool,
}

impl HttpRequestBuilder {
    /// Create new request builder
    pub fn new(method: crate::types::HttpMethod, url: impl Into<String>) -> Self {
        Self {
            request: HttpRequest::new(method, url),
            retry_policy: None,
            circuit_breaker_enabled: true,
        }
    }

    /// Add header
    #[must_use]
    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.request = self.request.header(name, value);
        self
    }

    /// Set JSON body
    pub fn json<T: serde::Serialize>(mut self, data: &T) -> NetworkResult<Self> {
        self.request = self.request.json_body(data)?;
        Ok(self)
    }

    /// Set body
    #[must_use]
    pub fn body(mut self, body: Vec<u8>) -> Self {
        self.request = self.request.body(body);
        self
    }

    /// Set timeout
    #[must_use]
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request = self.request.timeout(timeout);
        self
    }

    /// Set retry policy
    #[must_use]
    pub fn retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = Some(policy);
        self
    }

    /// Disable circuit breaker for this request
    #[must_use]
    pub fn disable_circuit_breaker(mut self) -> Self {
        self.circuit_breaker_enabled = false;
        self
    }

    /// Send the request
    pub async fn send(self, client: &dyn HttpClientTrait) -> NetworkResult<HttpResponse> {
        if let Some(retry_policy) = self.retry_policy {
            client.send_with_retry(self.request, retry_policy).await
        } else {
            client.send(self.request).await
        }
    }
}

/// Convenience functions for common HTTP methods
impl HttpRequestBuilder {
    /// Create GET request
    pub fn get(url: impl Into<String>) -> Self {
        Self::new(crate::types::HttpMethod::Get, url)
    }

    /// Create POST request
    pub fn post(url: impl Into<String>) -> Self {
        Self::new(crate::types::HttpMethod::Post, url)
    }

    /// Create PUT request
    pub fn put(url: impl Into<String>) -> Self {
        Self::new(crate::types::HttpMethod::Put, url)
    }

    /// Create DELETE request
    pub fn delete(url: impl Into<String>) -> Self {
        Self::new(crate::types::HttpMethod::Delete, url)
    }

    /// Create PATCH request
    pub fn patch(url: impl Into<String>) -> Self {
        Self::new(crate::types::HttpMethod::Patch, url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::HttpMethod;

    #[test]
    fn test_request_builder() -> NetworkResult<()> {
        let builder = HttpRequestBuilder::get("https://api.example.com/data")
            .header("Authorization", "Bearer token")
            .timeout(std::time::Duration::from_secs(30))
            .retry_policy(RetryPolicy::exponential(3, std::time::Duration::from_millis(100)));

        assert_eq!(builder.request.method, HttpMethod::Get);
        assert_eq!(builder.request.url, "https://api.example.com/data");
        assert!(builder.request.headers.contains_key("Authorization"));
        assert!(builder.retry_policy.is_some());

        Ok(())
    }

    #[test]
    fn test_json_request() -> NetworkResult<()> {
        let data = serde_json::json!({"key": "value"});
        let builder = HttpRequestBuilder::post("https://api.example.com/data")
            .json(&data)?;

        assert!(builder.request.body.is_some());
        assert!(builder.request.headers.contains_key("Content-Type"));

        Ok(())
    }

    #[test]
    fn test_stats_default() {
        let stats = HttpClientStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_responses, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.pool_utilization, 0.0_f64);
    }
}
