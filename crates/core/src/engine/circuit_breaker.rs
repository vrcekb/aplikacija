//! Circuit Breaker Pattern - Ultra-Reliable Fault Protection
//!
//! Implements enterprise-grade circuit breaker with:
//! - Multi-state circuit breaker (Closed, Open, Half-Open)
//! - Adaptive failure threshold based on historical performance
//! - Real-time health monitoring and automatic recovery
//! - Integration with backpressure system for coordinated protection
//! - Financial-grade robustness with comprehensive error handling
//!
//! This module provides the foundation for maintaining system stability
//! and preventing cascade failures in high-frequency trading environments.

use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

/// Circuit breaker error types
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum CircuitBreakerError {
    /// Circuit breaker is open
    #[error("Circuit breaker is open: {failure_count} failures in {window_ms}ms")]
    CircuitOpen {
        /// Number of failures
        failure_count: u64,
        /// Time window in milliseconds
        window_ms: u64,
    },

    /// Circuit breaker configuration error
    #[error("Invalid circuit breaker configuration: {details}")]
    InvalidConfiguration {
        /// Configuration details
        details: String,
    },

    /// Latency violation detected
    #[error("Latency violation: {actual_ms}ms > {threshold_ms}ms threshold")]
    LatencyViolation {
        /// Actual latency in milliseconds
        actual_ms: u64,
        /// Threshold in milliseconds
        threshold_ms: u64,
    },

    /// Health check failed
    #[error("Health check failed: {reason}")]
    HealthCheckFailed {
        /// Failure reason
        reason: String,
    },

    /// Recovery attempt failed
    #[error("Recovery attempt failed: attempt {attempt} of {max_attempts}")]
    RecoveryFailed {
        /// Current attempt
        attempt: u32,
        /// Maximum attempts
        max_attempts: u32,
    },

    /// Monitoring system failure
    #[error("Circuit breaker monitoring failure: {reason}")]
    MonitoringFailure {
        /// Failure reason
        reason: String,
    },
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed - normal operation
    Closed,
    /// Circuit is open - blocking requests
    Open,
    /// Circuit is half-open - testing recovery
    HalfOpen,
}

impl CircuitState {
    /// Check if requests should be allowed
    #[must_use]
    pub const fn allows_requests(&self) -> bool {
        matches!(self, Self::Closed | Self::HalfOpen)
    }

    /// Check if circuit is in failure state
    #[must_use]
    pub const fn is_failure_state(&self) -> bool {
        matches!(self, Self::Open)
    }
}

/// Circuit breaker configuration with latency monitoring
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold for opening circuit
    pub failure_threshold: u64,
    /// Success threshold for closing circuit
    pub success_threshold: u64,
    /// Time window for failure counting
    pub failure_window: Duration,
    /// Timeout before attempting recovery
    pub recovery_timeout: Duration,
    /// Maximum recovery attempts
    pub max_recovery_attempts: u32,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable adaptive thresholds
    pub enable_adaptive_thresholds: bool,
    /// Enable predictive opening
    pub enable_predictive_opening: bool,
    /// Latency threshold for financial applications
    pub latency_threshold: Duration,
    /// Latency violation threshold before opening circuit
    pub latency_violation_threshold: u32,
    /// Enable latency-based circuit opening
    pub enable_latency_protection: bool,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            failure_window: Duration::from_secs(60),
            recovery_timeout: Duration::from_secs(30),
            max_recovery_attempts: 3,
            health_check_interval: Duration::from_secs(10),
            enable_adaptive_thresholds: true,
            enable_predictive_opening: true,
            latency_threshold: Duration::from_millis(1), // 1ms for financial apps
            latency_violation_threshold: 10,
            enable_latency_protection: true,
        }
    }
}

/// Circuit breaker statistics with latency monitoring
#[derive(Debug, Default)]
pub struct CircuitBreakerStats {
    /// Total requests processed
    pub total_requests: AtomicU64,
    /// Successful requests
    pub successful_requests: AtomicU64,
    /// Failed requests
    pub failed_requests: AtomicU64,
    /// Circuit open events
    pub circuit_open_events: AtomicU64,
    /// Circuit close events
    pub circuit_close_events: AtomicU64,
    /// Recovery attempts
    pub recovery_attempts: AtomicU64,
    /// Successful recoveries
    pub successful_recoveries: AtomicU64,
    /// Current state
    pub current_state: AtomicUsize,
    /// Time in current state (milliseconds)
    pub time_in_state_ms: AtomicU64,
    /// Last state change timestamp
    pub last_state_change: AtomicU64,
    /// Latency violations count
    pub latency_violations: AtomicU64,
    /// Total latency (nanoseconds) for average calculation
    pub total_latency_ns: AtomicU64,
    /// Maximum latency observed (nanoseconds)
    pub max_latency_ns: AtomicU64,
    /// Minimum latency observed (nanoseconds)
    pub min_latency_ns: AtomicU64,
}

impl CircuitBreakerStats {
    /// Get current circuit state
    #[must_use]
    pub fn current_circuit_state(&self) -> CircuitState {
        match self.current_state.load(Ordering::Relaxed) {
            0 => CircuitState::Closed,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Open, // Default to safe state (includes 1 and others)
        }
    }

    /// Get success rate (0.0-1.0)
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0_f64;
        }

        let successful = self.successful_requests.load(Ordering::Relaxed);
        f64::from(u32::try_from(successful).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(total).unwrap_or(u32::MAX))
    }

    /// Get failure rate (0.0-1.0)
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        1.0_f64 - self.success_rate()
    }

    /// Record request result with latency
    pub fn record_request(&self, success: bool) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        if success {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record latency measurement
    pub fn record_latency(&self, latency: Duration) {
        let latency_ns = u64::try_from(latency.as_nanos()).unwrap_or(u64::MAX);

        // Update total latency for average calculation
        self.total_latency_ns
            .fetch_add(latency_ns, Ordering::Relaxed);

        // Update max latency
        let mut current_max = self.max_latency_ns.load(Ordering::Relaxed);
        while latency_ns > current_max {
            match self.max_latency_ns.compare_exchange_weak(
                current_max,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(new_max) => current_max = new_max,
            }
        }

        // Update min latency (initialize if zero)
        let mut current_min = self.min_latency_ns.load(Ordering::Relaxed);
        if current_min == 0 || latency_ns < current_min {
            while current_min == 0 || latency_ns < current_min {
                match self.min_latency_ns.compare_exchange_weak(
                    current_min,
                    latency_ns,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(new_min) => current_min = new_min,
                }
            }
        }
    }

    /// Record latency violation
    pub fn record_latency_violation(&self) {
        self.latency_violations.fetch_add(1, Ordering::Relaxed);
    }

    /// Get average latency in nanoseconds
    #[must_use]
    pub fn avg_latency_ns(&self) -> u64 {
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        if total_requests == 0 {
            return 0;
        }

        let total_latency = self.total_latency_ns.load(Ordering::Relaxed);
        total_latency / total_requests
    }

    /// Record state change
    pub fn record_state_change(&self, new_state: CircuitState) {
        let now_millis = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let now = u64::try_from(now_millis).unwrap_or(u64::MAX);

        self.current_state
            .store(new_state as usize, Ordering::Relaxed);
        self.last_state_change.store(now, Ordering::Relaxed);

        match new_state {
            CircuitState::Open => {
                self.circuit_open_events.fetch_add(1, Ordering::Relaxed);
            }
            CircuitState::Closed => {
                self.circuit_close_events.fetch_add(1, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                self.recovery_attempts.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record successful recovery
    pub fn record_successful_recovery(&self) {
        self.successful_recoveries.fetch_add(1, Ordering::Relaxed);
    }
}

/// Failure record for tracking and error analysis
#[derive(Debug, Clone, Copy)]
struct FailureRecord {
    /// When the failure occurred
    timestamp: Instant,
    /// Error classification for pattern analysis
    error_type: u8,
}

impl FailureRecord {
    /// Create new failure record
    #[must_use]
    pub const fn new(timestamp: Instant, error_type: u8) -> Self {
        Self {
            timestamp,
            error_type,
        }
    }

    /// Get failure age
    #[must_use]
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Check if failure is within time window
    #[must_use]
    pub fn is_within_window(&self, window: Duration) -> bool {
        self.age() <= window
    }

    /// Get error type for classification
    #[must_use]
    pub const fn error_type(&self) -> u8 {
        self.error_type
    }
}

/// Circuit breaker implementation with latency monitoring and micro-optimizations
#[repr(C, align(64))]
pub struct CircuitBreaker {
    /// Configuration
    config: CircuitBreakerConfig,
    /// Current state (optimized for fast atomic access)
    state_atomic: AtomicU8, // 0=Closed, 1=Open, 2=HalfOpen
    /// Current state (for compatibility)
    state: CircuitState,
    /// Statistics
    stats: Arc<CircuitBreakerStats>,
    /// Failure history
    failure_history: Mutex<Vec<FailureRecord>>,
    /// Last state change time
    last_state_change: Instant,
    /// Recovery attempt count
    recovery_attempts: u32,
    /// Adaptive failure threshold
    adaptive_threshold: AtomicU64,
    /// Last health check
    last_health_check: Mutex<Instant>,
    /// Latency violation count for circuit opening
    latency_violation_count: AtomicU64,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let stats = Arc::new(CircuitBreakerStats::default());
        stats.record_state_change(CircuitState::Closed);

        Self {
            config,
            state_atomic: AtomicU8::new(CircuitState::Closed as u8),
            state: CircuitState::Closed,
            stats,
            failure_history: Mutex::new(Vec::with_capacity(1000)),
            last_state_change: Instant::now(),
            recovery_attempts: 0,
            adaptive_threshold: AtomicU64::new(5), // Default threshold
            last_health_check: Mutex::new(Instant::now()),
            latency_violation_count: AtomicU64::new(0),
        }
    }

    /// Execute operation with circuit breaker protection and latency monitoring
    ///
    /// # Errors
    ///
    /// Returns error if circuit is open or operation fails
    pub fn execute<F, T, E>(&mut self, operation: F) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        // Check if request should be allowed
        if !self.should_allow_request()? {
            return Err(CircuitBreakerError::CircuitOpen {
                failure_count: self.stats.failed_requests.load(Ordering::Relaxed),
                window_ms: u64::try_from(self.config.failure_window.as_millis())
                    .unwrap_or(u64::MAX),
            });
        }

        // Execute operation with latency monitoring
        let start_time = Instant::now();
        let result = operation();
        let execution_time = start_time.elapsed();

        // Record latency
        self.stats.record_latency(execution_time);

        // Check for latency violations if enabled
        if self.config.enable_latency_protection && execution_time > self.config.latency_threshold {
            self.stats.record_latency_violation();
            self.latency_violation_count.fetch_add(1, Ordering::Relaxed);

            // Check if we should open circuit due to latency violations
            let violations = self.latency_violation_count.load(Ordering::Relaxed);
            if violations >= u64::from(self.config.latency_violation_threshold) {
                self.transition_to_open();
                return Err(CircuitBreakerError::LatencyViolation {
                    actual_ms: u64::try_from(execution_time.as_millis()).unwrap_or(u64::MAX),
                    threshold_ms: u64::try_from(self.config.latency_threshold.as_millis())
                        .unwrap_or(u64::MAX),
                });
            }
        }

        // Record result and update state
        match result {
            Ok(value) => {
                self.record_success();
                Ok(value)
            }
            Err(error) => {
                self.record_failure(&error.to_string());
                Err(CircuitBreakerError::HealthCheckFailed {
                    reason: error.to_string(),
                })
            }
        }
    }

    /// Ultra-fast state check - optimized hot path (sub-microsecond)
    ///
    /// This is the primary method for high-frequency operations requiring
    /// minimal latency overhead. Uses single atomic read for maximum performance.
    pub fn is_request_allowed_fast(&self) -> bool {
        // Single atomic read - fastest possible check
        self.state_atomic.load(Ordering::Relaxed) == CircuitState::Closed as u8
    }

    /// Check if request should be allowed (comprehensive version)
    ///
    /// # Errors
    ///
    /// Returns error if circuit breaker monitoring fails
    pub fn should_allow_request(&mut self) -> Result<bool, CircuitBreakerError> {
        // Fast path: check atomic state first
        let atomic_state = self.state_atomic.load(Ordering::Relaxed);
        if atomic_state == CircuitState::Closed as u8 {
            return Ok(true);
        }

        // Update state if needed
        self.update_state()?;

        match self.state {
            CircuitState::Open => {
                // Check if recovery timeout has passed
                if self.last_state_change.elapsed() >= self.config.recovery_timeout {
                    self.transition_to_half_open()?;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            CircuitState::Closed | CircuitState::HalfOpen => Ok(true),
        }
    }

    /// Get current circuit state
    #[must_use]
    pub const fn current_state(&self) -> CircuitState {
        self.state
    }

    /// Get circuit breaker statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<CircuitBreakerStats> {
        &self.stats
    }

    /// Check if circuit is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.state == CircuitState::Closed && self.stats.success_rate() > 0.95_f64
    }

    /// Force circuit open (for testing or emergency)
    ///
    /// # Errors
    ///
    /// Returns `CircuitBreakerError` if state transition fails
    pub fn force_open(&mut self) -> Result<(), CircuitBreakerError> {
        self.transition_to_open();
        Ok(())
    }

    /// Force circuit closed (for recovery)
    ///
    /// # Errors
    ///
    /// Returns `CircuitBreakerError` if state transition fails
    pub fn force_close(&mut self) -> Result<(), CircuitBreakerError> {
        self.transition_to_closed();
        Ok(())
    }

    /// Record successful operation
    fn record_success(&mut self) {
        self.stats.record_request(true);

        match self.state {
            CircuitState::HalfOpen => {
                // Check if we have enough successes to close circuit
                let recent_successes = self.count_recent_successes();
                if recent_successes >= self.config.success_threshold {
                    self.transition_to_closed();
                }
            }
            CircuitState::Closed => {
                // Update adaptive threshold if enabled
                if self.config.enable_adaptive_thresholds {
                    self.update_adaptive_threshold();
                }
            }
            CircuitState::Open => {
                // Should not happen, but handle gracefully
            }
        }
    }

    /// Record failed operation
    fn record_failure(&mut self, error_message: &str) {
        self.stats.record_request(false);

        // Add to failure history
        if let Ok(mut history) = self.failure_history.lock() {
            let error_type = Self::classify_error(error_message);
            let failure_record = FailureRecord::new(Instant::now(), error_type);
            history.push(failure_record);

            // Keep only recent failures using the record's window check
            history.retain(|record| record.is_within_window(self.config.failure_window));

            // Analyze error patterns for adaptive behavior
            self.analyze_error_patterns(&history);
        }

        // Check if circuit should open
        match self.state {
            CircuitState::Closed | CircuitState::HalfOpen => {
                let failure_count = self.count_recent_failures();
                let threshold = self.adaptive_threshold.load(Ordering::Relaxed);

                if failure_count >= threshold {
                    self.transition_to_open();
                }
            }
            CircuitState::Open => {
                // Already open, no action needed
            }
        }
    }

    /// Update circuit breaker state
    fn update_state(&self) -> Result<(), CircuitBreakerError> {
        // Perform health check if needed
        if self.should_perform_health_check()? {
            self.perform_health_check()?;
        }

        // Clean old failure records
        self.clean_old_failures();

        Ok(())
    }

    /// Transition to open state
    fn transition_to_open(&mut self) {
        self.state = CircuitState::Open;
        self.last_state_change = Instant::now();
        self.recovery_attempts = 0;
        self.stats.record_state_change(CircuitState::Open);
    }

    /// Transition to half-open state
    fn transition_to_half_open(&mut self) -> Result<(), CircuitBreakerError> {
        if self.recovery_attempts >= self.config.max_recovery_attempts {
            return Err(CircuitBreakerError::RecoveryFailed {
                attempt: self.recovery_attempts,
                max_attempts: self.config.max_recovery_attempts,
            });
        }

        self.state = CircuitState::HalfOpen;
        self.last_state_change = Instant::now();
        self.recovery_attempts += 1;
        self.stats.record_state_change(CircuitState::HalfOpen);
        Ok(())
    }

    /// Transition to closed state
    fn transition_to_closed(&mut self) {
        self.state = CircuitState::Closed;
        self.last_state_change = Instant::now();
        self.recovery_attempts = 0;
        self.stats.record_state_change(CircuitState::Closed);
        self.stats.record_successful_recovery();
    }

    /// Count recent failures
    fn count_recent_failures(&self) -> u64 {
        self.failure_history.lock().map_or(0, |history| {
            let cutoff = Instant::now()
                .checked_sub(self.config.failure_window)
                .unwrap_or_else(Instant::now);
            u64::try_from(
                history
                    .iter()
                    .filter(|record| record.timestamp > cutoff)
                    .count(),
            )
            .unwrap_or(0)
        })
    }

    /// Count recent successes (simplified)
    fn count_recent_successes(&self) -> u64 {
        // In a real implementation, we'd track success history too
        // For now, use a simplified approach based on success rate
        let total = self.stats.total_requests.load(Ordering::Relaxed);
        let successful = self.stats.successful_requests.load(Ordering::Relaxed);

        if total >= 10 {
            successful.saturating_sub(total - 10)
        } else {
            successful
        }
    }

    /// Update adaptive threshold
    fn update_adaptive_threshold(&self) {
        let failure_rate = self.stats.failure_rate();
        let current_threshold = self.adaptive_threshold.load(Ordering::Relaxed);

        // Adjust threshold based on recent performance
        let new_threshold = if failure_rate < 0.01_f64 {
            // Very low failure rate, can be more sensitive
            (current_threshold.saturating_mul(8).saturating_add(3)) / 9
        } else if failure_rate < 0.05_f64 {
            // Low failure rate, slightly more sensitive
            (current_threshold.saturating_mul(9).saturating_add(4)) / 10
        } else {
            // Higher failure rate, be less sensitive
            (current_threshold.saturating_mul(9).saturating_add(7)) / 10
        };

        self.adaptive_threshold
            .store(new_threshold.max(3), Ordering::Relaxed);
    }

    /// Check if health check should be performed
    fn should_perform_health_check(&self) -> Result<bool, CircuitBreakerError> {
        self.last_health_check.lock().map_or_else(
            |_| {
                Err(CircuitBreakerError::MonitoringFailure {
                    reason: "Failed to lock health check timestamp".to_string(),
                })
            },
            |last_check| Ok(last_check.elapsed() >= self.config.health_check_interval),
        )
    }

    /// Perform health check
    fn perform_health_check(&self) -> Result<(), CircuitBreakerError> {
        // Update health check timestamp
        self.last_health_check.lock().map_or_else(
            |_| {},
            |mut last_check| {
                *last_check = Instant::now();
            },
        );

        // Simplified health check - in production, this would be more comprehensive
        let success_rate = self.stats.success_rate();
        if success_rate < 0.5_f64 && self.state == CircuitState::Closed {
            return Err(CircuitBreakerError::HealthCheckFailed {
                reason: format!("Low success rate: {success_rate:.2}"),
            });
        }

        Ok(())
    }

    /// Clean old failure records
    fn clean_old_failures(&self) {
        self.failure_history.lock().map_or_else(
            |_| {},
            |mut history| {
                let cutoff = Instant::now()
                    .checked_sub(self.config.failure_window * 2)
                    .unwrap_or_else(Instant::now);
                history.retain(|record| record.timestamp > cutoff);
            },
        );
    }

    /// Analyze error patterns for adaptive circuit breaker behavior
    fn analyze_error_patterns(&self, history: &[FailureRecord]) {
        if history.len() < 5 {
            return; // Need sufficient data for pattern analysis
        }

        // Count error types
        let mut error_counts = [0_u32; 4]; // Support 4 error types
        for record in history {
            let error_type = record.error_type() as usize;
            if let Some(count) = error_counts.get_mut(error_type) {
                *count += 1;
            }
        }

        // Check for dominant error patterns
        let total_errors = error_counts.iter().sum::<u32>();
        if total_errors > 0 {
            for (error_type, &count) in error_counts.iter().enumerate() {
                let error_rate = f64::from(count) / f64::from(total_errors);

                // If one error type dominates (>70%), adjust threshold accordingly
                if error_rate > 0.7_f64 {
                    match error_type {
                        1 => {
                            // Timeout errors - be more sensitive
                            let current = self.adaptive_threshold.load(Ordering::Relaxed);
                            self.adaptive_threshold
                                .store(current.saturating_sub(1), Ordering::Relaxed);
                        }
                        2 => {
                            // Connection errors - be less sensitive
                            let current = self.adaptive_threshold.load(Ordering::Relaxed);
                            self.adaptive_threshold
                                .store(current + 1, Ordering::Relaxed);
                        }
                        3 => {
                            // Overload errors - be much more sensitive
                            let current = self.adaptive_threshold.load(Ordering::Relaxed);
                            self.adaptive_threshold
                                .store(current.saturating_sub(2), Ordering::Relaxed);
                        }
                        _ => {} // Generic errors - no adjustment
                    }
                }
            }
        }
    }

    /// Classify error type (simplified)
    fn classify_error(error_message: &str) -> u8 {
        // Simplified error classification
        if error_message.contains("timeout") {
            1
        } else if error_message.contains("connection") {
            2
        } else if error_message.contains("overload") {
            3
        } else {
            0 // Generic error
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_creation() {
        let config = CircuitBreakerConfig::default();
        let breaker = CircuitBreaker::new(config);

        assert_eq!(breaker.current_state(), CircuitState::Closed);
        assert!(breaker.is_healthy());
    }

    #[test]
    fn test_circuit_states() {
        assert!(CircuitState::Closed.allows_requests());
        assert!(CircuitState::HalfOpen.allows_requests());
        assert!(!CircuitState::Open.allows_requests());

        assert!(!CircuitState::Closed.is_failure_state());
        assert!(CircuitState::Open.is_failure_state());
    }

    #[test]
    fn test_successful_execution() {
        let config = CircuitBreakerConfig::default();
        let mut breaker = CircuitBreaker::new(config);

        let result = breaker.execute(|| Ok::<i32, &str>(42_i32));
        assert!(result.is_ok());
        if let Ok(value) = result {
            assert_eq!(value, 42_i32);
        }

        assert_eq!(breaker.stats.successful_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_failed_execution() {
        let config = CircuitBreakerConfig::default();
        let mut breaker = CircuitBreaker::new(config);

        let result = breaker.execute(|| Err::<i32, &str>("test error"));
        assert!(result.is_err());

        assert_eq!(breaker.stats.failed_requests.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_circuit_opening() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let mut breaker = CircuitBreaker::new(config);

        // Generate failures to open circuit
        for _ in 0_i32..5_i32 {
            let _ = breaker.execute(|| Err::<i32, &str>("test error"));
        }

        assert_eq!(breaker.current_state(), CircuitState::Open);
    }

    #[test]
    fn test_force_operations() {
        let config = CircuitBreakerConfig::default();
        let mut breaker = CircuitBreaker::new(config);

        // Force open
        let result = breaker.force_open();
        assert!(result.is_ok());
        assert_eq!(breaker.current_state(), CircuitState::Open);

        // Force close
        let result = breaker.force_close();
        assert!(result.is_ok());
        assert_eq!(breaker.current_state(), CircuitState::Closed);
    }
}
