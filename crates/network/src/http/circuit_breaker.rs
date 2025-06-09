//! Circuit Breaker Implementation
//!
//! Production-ready circuit breaker pattern for preventing cascade failures.
//! Implements three states: Closed, Open, and Half-Open with configurable thresholds.

use crate::error::{NetworkError, NetworkResult};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    /// Circuit is closed - requests are allowed through
    Closed,
    /// Circuit is open - requests are rejected immediately
    Open,
    /// Circuit is half-open - limited requests are allowed to test recovery
    HalfOpen,
}

impl std::fmt::Display for CircuitBreakerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures required to open the circuit
    pub failure_threshold: u32,
    /// Number of successes required to close the circuit from half-open
    pub success_threshold: u32,
    /// Time to wait before transitioning from open to half-open
    pub timeout: Duration,
    /// Maximum number of requests allowed in half-open state
    pub half_open_max_requests: u32,
    /// Time window for failure counting
    pub failure_window: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
            half_open_max_requests: 10,
            failure_window: Duration::from_secs(60),
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Clone)]
pub struct CircuitBreakerStats {
    /// Current state
    pub state: CircuitBreakerState,
    /// Total number of requests
    pub total_requests: u64,
    /// Total number of successful requests
    pub successful_requests: u64,
    /// Total number of failed requests
    pub failed_requests: u64,
    /// Total number of rejected requests (when circuit is open)
    pub rejected_requests: u64,
    /// Current failure count in the current window
    pub current_failures: u32,
    /// Current success count (in half-open state)
    pub current_successes: u32,
    /// Time when circuit was last opened
    pub last_opened_at: Option<Instant>,
    /// Time when circuit was last closed
    pub last_closed_at: Option<Instant>,
}

impl Default for CircuitBreakerStats {
    fn default() -> Self {
        Self {
            state: CircuitBreakerState::Closed,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            rejected_requests: 0,
            current_failures: 0,
            current_successes: 0,
            last_opened_at: None,
            last_closed_at: None,
        }
    }
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    /// Configuration
    config: CircuitBreakerConfig,
    /// Current state
    state: Arc<RwLock<CircuitBreakerState>>,
    /// Statistics
    stats: Arc<RwLock<CircuitBreakerStats>>,
    /// Failure count in current window
    failure_count: Arc<AtomicU32>,
    /// Success count in half-open state
    success_count: Arc<AtomicU32>,
    /// Total request count
    total_requests: Arc<AtomicU64>,
    /// Successful request count
    successful_requests: Arc<AtomicU64>,
    /// Failed request count
    failed_requests: Arc<AtomicU64>,
    /// Rejected request count
    rejected_requests: Arc<AtomicU64>,
    /// Time when circuit was opened
    opened_at: Arc<RwLock<Option<Instant>>>,
    /// Time when failure window started
    failure_window_start: Arc<RwLock<Instant>>,
    /// Number of requests in half-open state
    half_open_requests: Arc<AtomicU32>,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            stats: Arc::new(RwLock::new(CircuitBreakerStats::default())),
            failure_count: Arc::new(AtomicU32::new(0)),
            success_count: Arc::new(AtomicU32::new(0)),
            total_requests: Arc::new(AtomicU64::new(0)),
            successful_requests: Arc::new(AtomicU64::new(0)),
            failed_requests: Arc::new(AtomicU64::new(0)),
            rejected_requests: Arc::new(AtomicU64::new(0)),
            opened_at: Arc::new(RwLock::new(None)),
            failure_window_start: Arc::new(RwLock::new(Instant::now())),
            half_open_requests: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Check if request is allowed through the circuit breaker
    pub async fn is_request_allowed(&self) -> bool {
        let state = *self.state.read().await;
        
        match state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                // Check if timeout has elapsed to transition to half-open
                let value = *self.opened_at.read().await;
                if let Some(opened_time) = value {
                    if opened_time.elapsed() >= self.config.timeout {
                        self.transition_to_half_open().await;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Allow limited requests in half-open state
                let current_requests = self.half_open_requests.load(Ordering::Relaxed);
                current_requests < self.config.half_open_max_requests
            }
        }
    }

    /// Execute operation with circuit breaker protection
    ///
    /// # Errors
    ///
    /// Returns `NetworkError` if circuit breaker is open or operation fails
    pub async fn execute<F, Fut, T>(&self, operation: F) -> NetworkResult<T>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = NetworkResult<T>>,
    {
        // Check if request is allowed
        if !self.is_request_allowed().await {
            self.rejected_requests.fetch_add(1, Ordering::Relaxed);
            return Err(NetworkError::CircuitBreaker {
                state: self.state.read().await.to_string(),
                component: "http_client".to_string(),
                failure_count: self.failure_count.load(Ordering::Relaxed),
            });
        }

        // Increment request counters
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        let state = *self.state.read().await;
        if state == CircuitBreakerState::HalfOpen {
            self.half_open_requests.fetch_add(1, Ordering::Relaxed);
        }

        // Execute operation
        let result = operation().await;

        // Record result
        match &result {
            Ok(_) => self.record_success().await,
            Err(error) => self.record_failure(error).await,
        }

        result
    }

    /// Record successful operation
    async fn record_success(&self) {
        self.successful_requests.fetch_add(1, Ordering::Relaxed);
        
        let state = *self.state.read().await;
        
        match state {
            CircuitBreakerState::Closed => {
                // Reset failure count on success
                self.reset_failure_window().await;
            }
            CircuitBreakerState::HalfOpen => {
                let successes = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if successes >= self.config.success_threshold {
                    self.transition_to_closed().await;
                }
            }
            CircuitBreakerState::Open => {
                // Should not happen, but handle gracefully
            }
        }
    }

    /// Record failed operation
    async fn record_failure(&self, _error: &NetworkError) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        
        let state = *self.state.read().await;
        
        match state {
            CircuitBreakerState::Closed => {
                // Check if failure window needs reset
                let window_start = *self.failure_window_start.read().await;
                if window_start.elapsed() >= self.config.failure_window {
                    self.reset_failure_window().await;
                }
                
                let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                if failures >= self.config.failure_threshold {
                    self.transition_to_open().await;
                }
            }
            CircuitBreakerState::HalfOpen => {
                // Any failure in half-open state transitions back to open
                self.transition_to_open().await;
            }
            CircuitBreakerState::Open => {
                // Already open, no action needed
            }
        }
    }

    /// Transition to closed state
    async fn transition_to_closed(&self) {
        *self.state.write().await = CircuitBreakerState::Closed;
        self.success_count.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);
        self.reset_failure_window().await;
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.state = CircuitBreakerState::Closed;
        stats.last_closed_at = Some(Instant::now());
    }

    /// Transition to open state
    async fn transition_to_open(&self) {
        *self.state.write().await = CircuitBreakerState::Open;
        *self.opened_at.write().await = Some(Instant::now());
        self.success_count.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.state = CircuitBreakerState::Open;
        stats.last_opened_at = Some(Instant::now());
    }

    /// Transition to half-open state
    async fn transition_to_half_open(&self) {
        *self.state.write().await = CircuitBreakerState::HalfOpen;
        self.success_count.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.state = CircuitBreakerState::HalfOpen;
    }

    /// Reset failure counting window
    async fn reset_failure_window(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        *self.failure_window_start.write().await = Instant::now();
    }

    /// Get current circuit breaker state
    pub async fn state(&self) -> CircuitBreakerState {
        *self.state.read().await
    }

    /// Get circuit breaker statistics
    pub async fn stats(&self) -> CircuitBreakerStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update with current atomic values
        stats.state = *self.state.read().await;
        stats.total_requests = self.total_requests.load(Ordering::Relaxed);
        stats.successful_requests = self.successful_requests.load(Ordering::Relaxed);
        stats.failed_requests = self.failed_requests.load(Ordering::Relaxed);
        stats.rejected_requests = self.rejected_requests.load(Ordering::Relaxed);
        stats.current_failures = self.failure_count.load(Ordering::Relaxed);
        stats.current_successes = self.success_count.load(Ordering::Relaxed);
        
        stats
    }

    /// Reset circuit breaker to initial state
    pub async fn reset(&self) {
        *self.state.write().await = CircuitBreakerState::Closed;
        *self.opened_at.write().await = None;
        
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.rejected_requests.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);
        
        self.reset_failure_window().await;
        
        *self.stats.write().await = CircuitBreakerStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_circuit_breaker_closed_state() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        assert_eq!(cb.state().await, CircuitBreakerState::Closed);
        assert!(cb.is_request_allowed().await);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        let counter = Arc::new(AtomicU32::new(0));

        // First failure
        let counter_clone = counter.clone();
        let result = cb.execute(|| async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Err::<(), _>(NetworkError::http(500, "Server Error", None))
        }).await;
        assert!(result.is_err());
        assert_eq!(cb.state().await, CircuitBreakerState::Closed);

        // Second failure - should open circuit
        let counter_clone = counter.clone();
        let result = cb.execute(|| async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Err::<(), _>(NetworkError::http(500, "Server Error", None))
        }).await;
        assert!(result.is_err());
        assert_eq!(cb.state().await, CircuitBreakerState::Open);

        // Third request should be rejected
        let counter_clone = counter.clone();
        let result = cb.execute(|| async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }).await;
        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 2); // Third operation not executed
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_transition() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            timeout: Duration::from_millis(10),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Cause failure to open circuit
        let result = cb.execute(|| async {
            Err::<(), _>(NetworkError::http(500, "Server Error", None))
        }).await;
        assert!(result.is_err());
        assert_eq!(cb.state().await, CircuitBreakerState::Open);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Next request should transition to half-open
        assert!(cb.is_request_allowed().await);
        assert_eq!(cb.state().await, CircuitBreakerState::HalfOpen);
    }

    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout: Duration::from_millis(10),
            ..Default::default()
        };
        let cb = CircuitBreaker::new(config);

        // Open circuit
        let result = cb.execute(|| async {
            Err::<(), _>(NetworkError::http(500, "Server Error", None))
        }).await;
        assert!(result.is_err());

        // Wait and transition to half-open
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(cb.is_request_allowed().await);

        // Two successful requests should close circuit
        let result = cb.execute(|| async { Ok(()) }).await;
        assert!(result.is_ok());
        
        let result = cb.execute(|| async { Ok(()) }).await;
        assert!(result.is_ok());
        
        assert_eq!(cb.state().await, CircuitBreakerState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_stats() {
        let config = CircuitBreakerConfig::default();
        let cb = CircuitBreaker::new(config);

        // Execute some operations
        let _ = cb.execute(|| async { Ok(()) }).await;
        let _ = cb.execute(|| async { 
            Err::<(), _>(NetworkError::http(500, "Server Error", None))
        }).await;

        let stats = cb.stats().await;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.successful_requests, 1);
        assert_eq!(stats.failed_requests, 1);
    }
}
