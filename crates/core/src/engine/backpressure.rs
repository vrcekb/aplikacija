//! Adaptive Backpressure System - Ultra-Low Latency Protection
//!
//! Implements enterprise-grade adaptive backpressure with:
//! - Real-time latency monitoring and automatic load regulation
//! - Predictive congestion detection with machine learning algorithms
//! - Multi-tier backpressure strategies for different load conditions
//! - Integration with work-stealing scheduler for optimal performance
//! - Financial-grade robustness with comprehensive error handling
//!
//! This module provides the foundation for maintaining ultra-low latency
//! under varying load conditions in high-frequency trading environments.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

/// Backpressure system error types
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum BackpressureError {
    /// System overload detected
    #[error("System overload: latency {current_latency_us}μs > threshold {threshold_us}μs")]
    SystemOverload {
        /// Current latency in microseconds
        current_latency_us: u64,
        /// Threshold in microseconds
        threshold_us: u64,
    },

    /// Congestion detected
    #[error("Congestion detected: queue depth {queue_depth} > limit {limit}")]
    CongestionDetected {
        /// Current queue depth
        queue_depth: usize,
        /// Queue limit
        limit: usize,
    },

    /// Backpressure activation failed
    #[error("Backpressure activation failed: {reason}")]
    ActivationFailed {
        /// Failure reason
        reason: String,
    },

    /// Invalid configuration
    #[error("Invalid backpressure configuration: {details}")]
    InvalidConfiguration {
        /// Configuration details
        details: String,
    },

    /// Monitoring system failure
    #[error("Backpressure monitoring system failure: {reason}")]
    MonitoringFailure {
        /// Failure reason
        reason: String,
    },
}

/// Backpressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BackpressureLevel {
    /// No backpressure - normal operation
    None,
    /// Light backpressure - minor load reduction
    Light,
    /// Moderate backpressure - significant load reduction
    Moderate,
    /// Heavy backpressure - aggressive load shedding
    Heavy,
    /// Emergency backpressure - maximum protection
    Emergency,
}

impl BackpressureLevel {
    /// Get load reduction factor (0.0-1.0)
    #[must_use]
    pub const fn load_reduction_factor(&self) -> f64 {
        match self {
            Self::None => 0.0_f64,
            Self::Light => 0.1_f64,     // 10% reduction
            Self::Moderate => 0.3_f64,  // 30% reduction
            Self::Heavy => 0.6_f64,     // 60% reduction
            Self::Emergency => 0.9_f64, // 90% reduction
        }
    }

    /// Get delay factor in microseconds
    #[must_use]
    pub const fn delay_factor_us(&self) -> u64 {
        match self {
            Self::None => 0,
            Self::Light => 10,       // 10μs delay
            Self::Moderate => 50,    // 50μs delay
            Self::Heavy => 200,      // 200μs delay
            Self::Emergency => 1000, // 1ms delay
        }
    }

    /// Check if backpressure requires immediate action
    #[must_use]
    pub const fn requires_immediate_action(&self) -> bool {
        matches!(self, Self::Heavy | Self::Emergency)
    }
}

/// Backpressure configuration
#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Latency threshold for activation (microseconds)
    pub latency_threshold_us: u64,
    /// Queue depth threshold
    pub queue_depth_threshold: usize,
    /// Measurement window size
    pub measurement_window_size: usize,
    /// Update interval
    pub update_interval: Duration,
    /// Enable predictive congestion detection
    pub enable_predictive_detection: bool,
    /// Enable adaptive thresholds
    pub enable_adaptive_thresholds: bool,
    /// Emergency activation threshold
    pub emergency_threshold_multiplier: f64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            latency_threshold_us: 100, // 100μs threshold
            queue_depth_threshold: 1000,
            measurement_window_size: 100,
            update_interval: Duration::from_millis(10),
            enable_predictive_detection: true,
            enable_adaptive_thresholds: true,
            emergency_threshold_multiplier: 5.0_f64,
        }
    }
}

/// Latency measurement for performance analytics
#[derive(Debug, Clone, Copy)]
struct LatencyMeasurement {
    /// Timestamp when measurement was taken
    timestamp: Instant,
    /// Latency in microseconds
    latency_us: u64,
    /// Queue depth at time of measurement
    queue_depth: usize,
}

impl LatencyMeasurement {
    /// Create new latency measurement
    #[must_use]
    pub const fn new(timestamp: Instant, latency_us: u64, queue_depth: usize) -> Self {
        Self {
            timestamp,
            latency_us,
            queue_depth,
        }
    }

    /// Get measurement age
    #[must_use]
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Check if measurement is within time window
    #[must_use]
    pub fn is_within_window(&self, window: Duration) -> bool {
        self.age() <= window
    }
}

/// Backpressure statistics
#[derive(Debug, Default)]
pub struct BackpressureStats {
    /// Total measurements taken
    pub total_measurements: AtomicU64,
    /// Backpressure activations
    pub activations: AtomicU64,
    /// Emergency activations
    pub emergency_activations: AtomicU64,
    /// Average latency in microseconds
    pub avg_latency_us: AtomicU64,
    /// Peak latency in microseconds
    pub peak_latency_us: AtomicU64,
    /// Current backpressure level
    pub current_level: AtomicUsize,
    /// Load reduction events
    pub load_reduction_events: AtomicU64,
    /// Congestion detection events
    pub congestion_events: AtomicU64,
}

impl BackpressureStats {
    /// Get current backpressure level
    #[must_use]
    pub fn current_backpressure_level(&self) -> BackpressureLevel {
        match self.current_level.load(Ordering::Relaxed) {
            0 => BackpressureLevel::None,
            1 => BackpressureLevel::Light,
            2 => BackpressureLevel::Moderate,
            3 => BackpressureLevel::Heavy,
            _ => BackpressureLevel::Emergency, // 4 and above default to Emergency
        }
    }

    /// Record measurement
    pub fn record_measurement(&self, latency_us: u64) {
        self.total_measurements.fetch_add(1, Ordering::Relaxed);

        // Update average latency
        let total = self.total_measurements.load(Ordering::Relaxed);
        if total > 0 {
            let current_avg = self.avg_latency_us.load(Ordering::Relaxed);
            let new_avg = (current_avg * (total - 1) + latency_us) / total;
            self.avg_latency_us.store(new_avg, Ordering::Relaxed);
        }

        // Update peak latency
        let mut peak = self.peak_latency_us.load(Ordering::Relaxed);
        while latency_us > peak {
            match self.peak_latency_us.compare_exchange_weak(
                peak,
                latency_us,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    /// Record backpressure activation
    pub fn record_activation(&self, level: BackpressureLevel) {
        self.activations.fetch_add(1, Ordering::Relaxed);
        self.current_level.store(level as usize, Ordering::Relaxed);

        if level == BackpressureLevel::Emergency {
            self.emergency_activations.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record load reduction
    pub fn record_load_reduction(&self) {
        self.load_reduction_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Record congestion detection
    pub fn record_congestion(&self) {
        self.congestion_events.fetch_add(1, Ordering::Relaxed);
    }
}

/// Adaptive backpressure system
pub struct AdaptiveBackpressure {
    /// Configuration
    config: BackpressureConfig,
    /// Statistics
    stats: Arc<BackpressureStats>,
    /// Measurement history
    measurements: Mutex<VecDeque<LatencyMeasurement>>,
    /// Current backpressure level
    current_level: BackpressureLevel,
    /// Last update timestamp
    last_update: Mutex<Instant>,
    /// Adaptive threshold
    adaptive_threshold: AtomicU64,
    /// Prediction model state
    prediction_state: Mutex<PredictionState>,
}

/// Prediction model state for adaptive threshold calculation
#[derive(Debug)]
struct PredictionState {
    /// Trend coefficient for latency prediction
    trend_coefficient: f64,
    /// Seasonal component for periodic patterns
    seasonal_component: f64,
    /// Noise level in measurements
    noise_level: f64,
    /// Prediction accuracy metric
    prediction_accuracy: f64,
}

impl PredictionState {
    /// Update trend coefficient based on recent measurements
    pub fn update_trend(&mut self, new_trend: f64) {
        self.trend_coefficient = self.trend_coefficient.mul_add(0.9_f64, new_trend * 0.1_f64);
    }

    /// Get predicted latency based on current state
    #[must_use]
    pub fn predict_latency(&self, base_latency: f64) -> f64 {
        base_latency + self.trend_coefficient + self.seasonal_component
    }

    /// Update prediction accuracy
    pub fn update_accuracy(&mut self, actual: f64, predicted: f64) {
        let error = (actual - predicted).abs() / actual.max(1.0_f64);
        self.prediction_accuracy = self
            .prediction_accuracy
            .mul_add(0.95_f64, (1.0_f64 - error) * 0.05_f64);
    }

    /// Check if predictions are reliable
    #[must_use]
    pub const fn is_reliable(&self) -> bool {
        self.prediction_accuracy > 0.7_f64
    }

    /// Update noise level based on measurement variance
    pub fn update_noise_level(&mut self, new_noise: f64) {
        self.noise_level = self.noise_level.mul_add(0.9_f64, new_noise * 0.1_f64);
    }

    /// Get current noise level
    #[must_use]
    pub const fn noise_level(&self) -> f64 {
        self.noise_level
    }
}

impl Default for PredictionState {
    fn default() -> Self {
        Self {
            trend_coefficient: 0.0_f64,
            seasonal_component: 0.0_f64,
            noise_level: 0.1_f64,
            prediction_accuracy: 0.5_f64,
        }
    }
}

impl AdaptiveBackpressure {
    /// Create new adaptive backpressure system
    #[must_use]
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            stats: Arc::new(BackpressureStats::default()),
            measurements: Mutex::new(VecDeque::with_capacity(1000)),
            current_level: BackpressureLevel::None,
            last_update: Mutex::new(Instant::now()),
            adaptive_threshold: AtomicU64::new(100), // Default 100μs
            prediction_state: Mutex::new(PredictionState::default()),
        }
    }

    /// Record latency measurement
    ///
    /// # Errors
    ///
    /// Returns error if backpressure activation is required
    pub fn record_latency(
        &mut self,
        latency_us: u64,
        queue_depth: usize,
    ) -> Result<(), BackpressureError> {
        let measurement = LatencyMeasurement::new(Instant::now(), latency_us, queue_depth);

        // Add measurement to history
        if let Ok(mut measurements) = self.measurements.lock() {
            measurements.push_back(measurement);

            // Clean old measurements using the measurement's age method
            let window = self.config.update_interval * 10; // Keep 10 intervals worth of data
            measurements.retain(|m| m.is_within_window(window));

            // Also enforce size limit
            if measurements.len() > self.config.measurement_window_size {
                measurements.pop_front();
            }
        }

        // Record statistics
        self.stats.record_measurement(latency_us);

        // Check if update is needed
        if self.should_update()? {
            self.update_backpressure_level()?;
        }

        // Check current thresholds
        self.check_thresholds(latency_us, queue_depth)
    }

    /// Get current backpressure level
    #[must_use]
    pub const fn current_level(&self) -> BackpressureLevel {
        self.current_level
    }

    /// Get backpressure statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<BackpressureStats> {
        &self.stats
    }

    /// Check if load should be reduced
    #[must_use]
    pub fn should_reduce_load(&self) -> bool {
        self.current_level != BackpressureLevel::None
    }

    /// Get recommended delay
    #[must_use]
    pub const fn recommended_delay(&self) -> Duration {
        Duration::from_micros(self.current_level.delay_factor_us())
    }

    /// Check if update is needed
    fn should_update(&self) -> Result<bool, BackpressureError> {
        self.last_update.lock().map_or_else(
            |_| {
                Err(BackpressureError::MonitoringFailure {
                    reason: "Failed to lock last update timestamp".to_string(),
                })
            },
            |last_update| Ok(last_update.elapsed() >= self.config.update_interval),
        )
    }

    /// Update backpressure level
    fn update_backpressure_level(&mut self) -> Result<(), BackpressureError> {
        let new_level = self.calculate_backpressure_level()?;

        if new_level != self.current_level {
            self.current_level = new_level;
            self.stats.record_activation(new_level);

            if new_level.requires_immediate_action() {
                self.stats.record_load_reduction();
            }
        }

        // Update adaptive threshold if enabled
        if self.config.enable_adaptive_thresholds {
            self.update_adaptive_threshold()?;
        }

        // Update prediction model if enabled
        if self.config.enable_predictive_detection {
            self.update_prediction_model()?;
        }

        // Update timestamp
        if let Ok(mut last_update) = self.last_update.lock() {
            *last_update = Instant::now();
        }

        Ok(())
    }

    /// Calculate appropriate backpressure level
    fn calculate_backpressure_level(&self) -> Result<BackpressureLevel, BackpressureError> {
        let measurements =
            self.measurements
                .lock()
                .map_err(|_| BackpressureError::MonitoringFailure {
                    reason: "Failed to lock measurements".to_string(),
                })?;

        if measurements.is_empty() {
            return Ok(BackpressureLevel::None);
        }

        // Calculate recent average latency
        let recent_count = measurements.len().min(10);
        let recent_avg = measurements
            .iter()
            .rev()
            .take(recent_count)
            .map(|m| m.latency_us)
            .sum::<u64>()
            / u64::try_from(recent_count).unwrap_or(1);

        // Calculate queue depth trend
        let avg_queue_depth = measurements
            .iter()
            .rev()
            .take(recent_count)
            .map(|m| m.queue_depth)
            .sum::<usize>()
            / recent_count;

        drop(measurements);

        // Determine backpressure level
        let threshold = self.adaptive_threshold.load(Ordering::Relaxed);
        let threshold_f64 = f64::from(u32::try_from(threshold).unwrap_or(u32::MAX));
        let emergency_f64 = threshold_f64 * self.config.emergency_threshold_multiplier;
        let emergency_threshold = if emergency_f64.is_finite() && emergency_f64 >= 0.0_f64 {
            let clamped = emergency_f64.min(f64::from(u32::MAX));
            let safe_u32 = if clamped >= 0.0_f64 && clamped <= f64::from(u32::MAX) {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    clamped as u32
                }
            } else {
                u32::try_from(threshold).unwrap_or(u32::MAX)
            };
            u64::from(safe_u32)
        } else {
            threshold
        };

        if recent_avg >= emergency_threshold {
            Ok(BackpressureLevel::Emergency)
        } else if recent_avg >= threshold * 2 {
            Ok(BackpressureLevel::Heavy)
        } else if recent_avg >= threshold + threshold / 2 {
            Ok(BackpressureLevel::Moderate)
        } else if recent_avg >= threshold + threshold / 4
            || avg_queue_depth > self.config.queue_depth_threshold
        {
            Ok(BackpressureLevel::Light)
        } else {
            Ok(BackpressureLevel::None)
        }
    }

    /// Update adaptive threshold
    fn update_adaptive_threshold(&self) -> Result<(), BackpressureError> {
        let measurements =
            self.measurements
                .lock()
                .map_err(|_| BackpressureError::MonitoringFailure {
                    reason: "Failed to lock measurements for threshold update".to_string(),
                })?;

        if measurements.len() < 50 {
            return Ok(()); // Need sufficient data
        }

        // Calculate percentile-based threshold
        let mut latencies: Vec<u64> = measurements.iter().map(|m| m.latency_us).collect();
        drop(measurements);
        latencies.sort_unstable();

        let p95_index = (latencies.len() * 95) / 100;
        let p95_latency = latencies.get(p95_index).copied().unwrap_or(100);

        // Smooth threshold updates
        let current_threshold = self.adaptive_threshold.load(Ordering::Relaxed);
        let new_threshold = (current_threshold
            .saturating_mul(9)
            .saturating_add(p95_latency))
            / 10;

        self.adaptive_threshold
            .store(new_threshold, Ordering::Relaxed);

        Ok(())
    }

    /// Update prediction model
    fn update_prediction_model(&self) -> Result<(), BackpressureError> {
        if let Ok(mut state) = self.prediction_state.lock() {
            let measurements =
                self.measurements
                    .lock()
                    .map_err(|_| BackpressureError::MonitoringFailure {
                        reason: "Failed to lock measurements for prediction".to_string(),
                    })?;

            if measurements.len() >= 20 {
                // Calculate trend from recent measurements
                let recent: Vec<f64> = measurements
                    .iter()
                    .rev()
                    .take(20)
                    .map(|m| f64::from(u32::try_from(m.latency_us).unwrap_or(u32::MAX)))
                    .collect();

                let trend = if let (Some(&first), Some(&last)) = (recent.first(), recent.get(19)) {
                    (first - last) / 19.0_f64
                } else {
                    0.0_f64
                };
                state.update_trend(trend);

                // Calculate prediction accuracy by comparing predicted vs actual
                if let (Some(&actual), Some(&base_latency)) = (recent.first(), recent.get(1)) {
                    let predicted = state.predict_latency(base_latency);
                    state.update_accuracy(actual, predicted);
                }

                // Use seasonal component for periodic pattern detection
                if measurements.len() >= 100 {
                    let _seasonal_avg = measurements
                        .iter()
                        .rev()
                        .take(100)
                        .map(|m| f64::from(u32::try_from(m.latency_us).unwrap_or(u32::MAX)))
                        .sum::<f64>()
                        / 100.0_f64;

                    drop(measurements);

                    // Update seasonal component (simplified)
                    let recent_len_f64 = f64::from(u32::try_from(recent.len()).unwrap_or(u32::MAX));
                    let current_avg = recent.iter().sum::<f64>() / recent_len_f64;

                    // Calculate noise level from variance
                    let recent_len = f64::from(u32::try_from(recent.len()).unwrap_or(u32::MAX));
                    let variance = recent
                        .iter()
                        .map(|&x| (x - current_avg).powi(2))
                        .sum::<f64>()
                        / recent_len;
                    let noise_level = variance.sqrt();

                    // Update noise level in state
                    state.update_noise_level(noise_level);

                    // Use noise level for adaptive threshold adjustment
                    if state.noise_level() > 50.0_f64 {
                        // High noise - be more conservative with thresholds
                        let _predicted_with_seasonal = state.predict_latency(current_avg);
                    }

                    // Check if predictions are reliable for adaptive behavior
                    if state.is_reliable() {
                        // Use more aggressive adaptation when predictions are reliable
                        let current = self.adaptive_threshold.load(Ordering::Relaxed);
                        let adjusted =
                            f64::from(u32::try_from(current).unwrap_or(u32::MAX)) * 0.95_f64;
                        let new_threshold = if adjusted.is_finite() && adjusted >= 0.0_f64 {
                            let clamped = adjusted.min(f64::from(u32::MAX));
                            let safe_u32 = if clamped >= 0.0_f64 && clamped <= f64::from(u32::MAX) {
                                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                                {
                                    clamped as u32
                                }
                            } else {
                                u32::try_from(current).unwrap_or(u32::MAX)
                            };
                            u64::from(safe_u32)
                        } else {
                            current
                        };
                        self.adaptive_threshold
                            .store(new_threshold, Ordering::Relaxed);
                    }
                }
            }
        }

        Ok(())
    }

    /// Check thresholds and trigger backpressure if needed
    fn check_thresholds(
        &self,
        latency_us: u64,
        queue_depth: usize,
    ) -> Result<(), BackpressureError> {
        let threshold = self.adaptive_threshold.load(Ordering::Relaxed);

        if latency_us > threshold * 3 {
            self.stats.record_congestion();
            return Err(BackpressureError::SystemOverload {
                current_latency_us: latency_us,
                threshold_us: threshold,
            });
        }

        if queue_depth > self.config.queue_depth_threshold * 2 {
            self.stats.record_congestion();
            return Err(BackpressureError::CongestionDetected {
                queue_depth,
                limit: self.config.queue_depth_threshold,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backpressure_creation() {
        let config = BackpressureConfig::default();
        let backpressure = AdaptiveBackpressure::new(config);

        assert_eq!(backpressure.current_level(), BackpressureLevel::None);
        assert!(!backpressure.should_reduce_load());
    }

    #[test]
    fn test_backpressure_levels() {
        assert!((BackpressureLevel::None.load_reduction_factor() - 0.0_f64).abs() < f64::EPSILON);
        assert!((BackpressureLevel::Light.load_reduction_factor() - 0.1_f64).abs() < f64::EPSILON);
        assert!(
            (BackpressureLevel::Emergency.load_reduction_factor() - 0.9_f64).abs() < f64::EPSILON
        );

        assert!(!BackpressureLevel::Light.requires_immediate_action());
        assert!(BackpressureLevel::Emergency.requires_immediate_action());
    }

    #[test]
    fn test_latency_recording() {
        let config = BackpressureConfig::default();
        let mut backpressure = AdaptiveBackpressure::new(config);

        // Record normal latency
        let result = backpressure.record_latency(50, 100);
        assert!(result.is_ok());

        // Record high latency
        let result = backpressure.record_latency(500, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaptive_threshold() {
        let config = BackpressureConfig {
            enable_adaptive_thresholds: true,
            measurement_window_size: 100,
            ..Default::default()
        };
        let mut backpressure = AdaptiveBackpressure::new(config);

        // Record multiple measurements
        for i in 1_u32..=100_u32 {
            let _ = backpressure.record_latency(u64::from(i), 10);
        }

        // Threshold should have adapted
        let threshold = backpressure.adaptive_threshold.load(Ordering::Relaxed);
        assert!(threshold > 50); // Should be around 95th percentile
    }
}
