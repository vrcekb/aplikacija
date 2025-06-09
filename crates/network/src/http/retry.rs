//! Retry Policy Implementation
//!
//! Production-ready retry mechanisms with exponential backoff, jitter, and circuit breaker integration.
//! Optimized for financial applications requiring high reliability.

use crate::error::{NetworkError, NetworkResult};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Enable jitter to avoid thundering herd
    pub enable_jitter: bool,
    /// HTTP status codes that should trigger a retry
    pub retry_on_status_codes: Vec<u16>,
    /// Retry on network errors
    pub retry_on_network_errors: bool,
    /// Retry on timeout errors
    pub retry_on_timeout: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0_f64,
            enable_jitter: true,
            retry_on_status_codes: vec![500, 502, 503, 504, 429], // Server errors and rate limiting
            retry_on_network_errors: true,
            retry_on_timeout: true,
        }
    }
}

impl RetryPolicy {
    /// Create exponential backoff retry policy
    pub fn exponential(max_attempts: u32, initial_delay: Duration) -> Self {
        Self {
            max_attempts,
            initial_delay,
            backoff_multiplier: 2.0_f64,
            enable_jitter: true,
            ..Default::default()
        }
    }

    /// Create linear backoff retry policy
    pub fn linear(max_attempts: u32, delay: Duration) -> Self {
        Self {
            max_attempts,
            initial_delay: delay,
            backoff_multiplier: 1.0_f64,
            enable_jitter: false,
            ..Default::default()
        }
    }

    /// Create fixed delay retry policy
    pub fn fixed(max_attempts: u32, delay: Duration) -> Self {
        Self {
            max_attempts,
            initial_delay: delay,
            max_delay: delay,
            backoff_multiplier: 1.0_f64,
            enable_jitter: false,
            ..Default::default()
        }
    }

    /// Create no-retry policy
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            ..Default::default()
        }
    }

    /// Check if error should trigger a retry
    pub fn should_retry(&self, error: &NetworkError, attempt: u32) -> bool {
        if attempt >= self.max_attempts {
            return false;
        }

        match error {
            NetworkError::Http { status_code, .. } => {
                self.retry_on_status_codes.contains(status_code)
            }
            NetworkError::Connection { .. } => self.retry_on_network_errors,
            NetworkError::Timeout { .. } => self.retry_on_timeout,
            NetworkError::Critical(_) => false, // Never retry critical errors
            NetworkError::CircuitBreaker { .. } => false, // Circuit breaker handles its own logic
            _ => self.retry_on_network_errors,
        }
    }

    /// Calculate delay for next retry attempt
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }

        let base_delay = if self.backoff_multiplier == 1.0_f64 {
            // Linear or fixed backoff
            self.initial_delay
        } else {
            // Exponential backoff
            let multiplier = self.backoff_multiplier.powi(i32::try_from(attempt - 1).unwrap_or(0));
            Duration::from_nanos(
                (self.initial_delay.as_nanos() as f64 * multiplier) as u64
            )
        };

        let delay = base_delay.min(self.max_delay);

        if self.enable_jitter {
            self.add_jitter(delay)
        } else {
            delay
        }
    }

    /// Add jitter to delay to avoid thundering herd
    fn add_jitter(&self, delay: Duration) -> Duration {
        let mut rng = rand::thread_rng();
        let jitter_factor = rng.gen_range(0.5_f64..1.5_f64);
        Duration::from_nanos((delay.as_nanos() as f64 * jitter_factor) as u64)
    }
}

/// Retry state tracker
#[derive(Debug, Clone)]
pub struct RetryState {
    /// Current attempt number (0-based)
    pub attempt: u32,
    /// Total elapsed time
    pub elapsed: Duration,
    /// Start time of retry sequence
    pub start_time: Instant,
    /// Last error encountered
    pub last_error: Option<NetworkError>,
    /// Retry policy being used
    pub policy: RetryPolicy,
}

impl RetryState {
    /// Create new retry state
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            attempt: 0,
            elapsed: Duration::ZERO,
            start_time: Instant::now(),
            last_error: None,
            policy,
        }
    }

    /// Check if more retries are allowed
    pub fn can_retry(&self) -> bool {
        self.attempt < self.policy.max_attempts
    }

    /// Record an error and determine if retry should be attempted
    pub fn record_error(&mut self, error: NetworkError) -> bool {
        self.last_error = Some(error.clone());
        self.elapsed = self.start_time.elapsed();

        if !self.can_retry() {
            return false;
        }

        self.policy.should_retry(&error, self.attempt)
    }

    /// Get delay for next retry attempt
    pub fn next_delay(&mut self) -> Duration {
        self.attempt += 1;
        self.policy.calculate_delay(self.attempt)
    }

    /// Check if retry sequence has timed out
    pub fn is_timed_out(&self, max_total_time: Duration) -> bool {
        self.elapsed >= max_total_time
    }

    /// Get retry statistics
    pub fn stats(&self) -> RetryStats {
        RetryStats {
            total_attempts: self.attempt,
            total_elapsed: self.elapsed,
            success: self.last_error.is_none(),
            last_error: self.last_error.as_ref().map(|e| e.to_string()),
        }
    }
}

/// Retry execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryStats {
    /// Total number of attempts made
    pub total_attempts: u32,
    /// Total time elapsed during retry sequence
    pub total_elapsed: Duration,
    /// Whether the operation ultimately succeeded
    pub success: bool,
    /// Last error message (if any)
    pub last_error: Option<String>,
}

/// Retry executor for async operations
pub struct RetryExecutor<F, Fut, T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = NetworkResult<T>>,
{
    operation: F,
    state: RetryState,
    max_total_time: Option<Duration>,
}

impl<F, Fut, T> RetryExecutor<F, Fut, T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = NetworkResult<T>>,
{
    /// Create new retry executor
    pub fn new(operation: F, policy: RetryPolicy) -> Self {
        Self {
            operation,
            state: RetryState::new(policy),
            max_total_time: None,
        }
    }

    /// Set maximum total time for all retry attempts
    #[must_use]
    pub fn with_max_total_time(mut self, max_time: Duration) -> Self {
        self.max_total_time = Some(max_time);
        self
    }

    /// Execute operation with retry logic
    pub async fn execute(mut self) -> NetworkResult<T> {
        loop {
            // Check total timeout
            if let Some(max_time) = self.max_total_time {
                if self.state.is_timed_out(max_time) {
                    return Err(NetworkError::timeout(
                        "retry_sequence",
                        self.state.elapsed,
                        None,
                    ));
                }
            }

            // Execute operation
            match (self.operation)().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    // Check if we should retry
                    if !self.state.record_error(error.clone()) {
                        return Err(NetworkError::RetryExhausted {
                            attempts: self.state.attempt,
                            operation: "http_request".to_string(),
                            last_error: error.to_string(),
                        });
                    }

                    // Calculate delay and wait
                    let delay = self.state.next_delay();
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
    }

    /// Get current retry statistics
    pub fn stats(&self) -> RetryStats {
        self.state.stats()
    }
}

/// Convenience function to execute operation with retry
pub async fn retry_async<F, Fut, T>(
    operation: F,
    policy: RetryPolicy,
) -> NetworkResult<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = NetworkResult<T>>,
{
    RetryExecutor::new(operation, policy).execute().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_retry_policy_creation() {
        let policy = RetryPolicy::exponential(5, Duration::from_millis(100));
        assert_eq!(policy.max_attempts, 5);
        assert_eq!(policy.initial_delay, Duration::from_millis(100));
        assert_eq!(policy.backoff_multiplier, 2.0_f64);

        let policy = RetryPolicy::linear(3, Duration::from_millis(500));
        assert_eq!(policy.max_attempts, 3);
        assert_eq!(policy.backoff_multiplier, 1.0_f64);

        let policy = RetryPolicy::none();
        assert_eq!(policy.max_attempts, 1);
    }

    #[test]
    fn test_should_retry() {
        let policy = RetryPolicy::default();

        // Should retry on server errors
        let error = NetworkError::http(500, "Internal Server Error", None);
        assert!(policy.should_retry(&error, 1));

        // Should not retry on client errors
        let error = NetworkError::http(404, "Not Found", None);
        assert!(!policy.should_retry(&error, 1));

        // Should not retry if max attempts reached
        let error = NetworkError::http(500, "Internal Server Error", None);
        assert!(!policy.should_retry(&error, 5));
    }

    #[test]
    fn test_delay_calculation() {
        let policy = RetryPolicy::exponential(5, Duration::from_millis(100));

        assert_eq!(policy.calculate_delay(0), Duration::ZERO);
        
        // Note: Due to jitter, we test ranges
        let delay1 = policy.calculate_delay(1);
        assert!(delay1 >= Duration::from_millis(50) && delay1 <= Duration::from_millis(150));

        let policy_no_jitter = RetryPolicy {
            enable_jitter: false,
            ..policy
        };
        assert_eq!(policy_no_jitter.calculate_delay(1), Duration::from_millis(100));
        assert_eq!(policy_no_jitter.calculate_delay(2), Duration::from_millis(200));
    }

    #[tokio::test]
    async fn test_retry_executor() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let operation = move || {
            let counter = counter_clone.clone();
            async move {
                let count = counter.fetch_add(1, Ordering::SeqCst);
                if count < 2 {
                    Err(NetworkError::http(500, "Server Error", None))
                } else {
                    Ok("Success".to_string())
                }
            }
        };

        let policy = RetryPolicy::fixed(5, Duration::from_millis(1));
        let result = RetryExecutor::new(operation, policy).execute().await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Success");
        assert_eq!(counter.load(Ordering::SeqCst), 3); // Failed twice, succeeded on third
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let operation = || async {
            Err::<String, _>(NetworkError::http(500, "Server Error", None))
        };

        let policy = RetryPolicy::fixed(2, Duration::from_millis(1));
        let result = RetryExecutor::new(operation, policy).execute().await;

        assert!(result.is_err());
        if let Err(NetworkError::RetryExhausted { attempts, .. }) = result {
            assert_eq!(attempts, 2);
        } else {
            panic!("Expected RetryExhausted error");
        }
    }
}
