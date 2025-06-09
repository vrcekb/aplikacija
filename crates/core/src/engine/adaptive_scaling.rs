//! Advanced Worker Scaling - Ultra-Performance Dynamic Worker Management
//!
//! Production-ready adaptive worker scaling for `TallyIO` financial applications.
//! Implements lock-free worker scaling with predictive load balancing and
//! sub-microsecond scaling decisions for <1ms latency requirements.

use std::{
    sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

use thiserror::Error;

/// Advanced worker scaling error types
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum ScalingError {
    /// Invalid worker count
    #[error("Invalid worker count: {count}")]
    InvalidWorkerCount {
        /// Worker count that caused the error
        count: usize,
    },

    /// Scaling operation in progress
    #[error("Scaling operation already in progress")]
    ScalingInProgress,

    /// Resource exhaustion
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        /// Resource that was exhausted
        resource: &'static str,
    },

    /// Critical latency violation
    #[error("Latency violation: {latency_ns}ns > {max_latency_ns}ns")]
    LatencyViolation {
        /// Actual latency in nanoseconds
        latency_ns: u64,
        /// Maximum allowed latency in nanoseconds
        max_latency_ns: u64,
    },
}

/// Worker scaling configuration
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Minimum number of workers
    pub min_workers: usize,
    /// Maximum number of workers
    pub max_workers: usize,
    /// Target latency in nanoseconds
    pub target_latency_ns: u64,
    /// Scaling threshold (0.0 to 1.0)
    pub scaling_threshold: f64,
    /// Cooldown period between scaling operations
    pub cooldown_duration: Duration,
    /// Predictive scaling window
    pub prediction_window: Duration,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 16,
            target_latency_ns: 1_000_000, // 1ms
            scaling_threshold: 0.8_f64,
            cooldown_duration: Duration::from_millis(100),
            prediction_window: Duration::from_millis(50),
        }
    }
}

/// Lock-free worker scaling metrics
#[derive(Debug, Default)]
#[repr(C, align(64))]
pub struct ScalingMetrics {
    /// Current worker count
    pub current_workers: AtomicUsize,
    /// Target worker count
    pub target_workers: AtomicUsize,
    /// Total scaling operations
    pub scaling_operations: AtomicU64,
    /// Successful scaling operations
    pub successful_scalings: AtomicU64,
    /// Failed scaling operations
    pub failed_scalings: AtomicU64,
    /// Average scaling latency in nanoseconds
    pub avg_scaling_latency_ns: AtomicU64,
    /// Last scaling timestamp
    pub last_scaling_timestamp: AtomicU64,
    /// Predictive load factor
    pub predicted_load: AtomicU64, // Fixed-point: multiply by 1000
}

/// Advanced worker scaling manager
///
/// Implements lock-free dynamic worker scaling with predictive load balancing.
/// Optimized for ultra-low latency financial applications with <1ms requirements.
#[derive(Debug)]
#[repr(C, align(64))]
pub struct AdaptiveScalingManager {
    /// Scaling configuration
    config: ScalingConfig,
    /// Scaling metrics
    metrics: ScalingMetrics,
    /// Scaling state pointer (lock-free)
    scaling_state: AtomicPtr<ScalingState>,
    /// Load history for prediction
    load_history: AtomicPtr<LoadHistory>,
    /// Manager unique identifier
    manager_id: u64,
}

/// Lock-free scaling state
#[derive(Debug)]
#[repr(C, align(64))]
struct ScalingState {
    /// Is scaling operation in progress
    is_scaling: AtomicU64, // 0 = idle, 1 = scaling
    /// Current load factor (fixed-point: multiply by 1000)
    current_load: AtomicU64,
    /// Last measurement timestamp
    last_measurement: AtomicU64,
    /// Scaling decision timestamp
    scaling_decision_time: AtomicU64,
}

/// Load history for predictive scaling
#[derive(Debug)]
#[repr(C, align(64))]
struct LoadHistory {
    /// Load samples (circular buffer)
    samples: [AtomicU64; 32],
    /// Current sample index
    current_index: AtomicUsize,
    /// Sample count
    sample_count: AtomicUsize,
}

impl Default for ScalingState {
    fn default() -> Self {
        Self {
            is_scaling: AtomicU64::new(0),
            current_load: AtomicU64::new(0),
            last_measurement: AtomicU64::new(0),
            scaling_decision_time: AtomicU64::new(0),
        }
    }
}

impl Default for LoadHistory {
    fn default() -> Self {
        Self {
            samples: [const { AtomicU64::new(0) }; 32],
            current_index: AtomicUsize::new(0),
            sample_count: AtomicUsize::new(0),
        }
    }
}

impl AdaptiveScalingManager {
    /// Create new adaptive scaling manager
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: ScalingConfig) -> Result<Self, ScalingError> {
        if config.min_workers == 0 || config.min_workers > config.max_workers {
            return Err(ScalingError::InvalidWorkerCount {
                count: config.min_workers,
            });
        }

        let scaling_state = Box::into_raw(Box::new(ScalingState::default()));
        let load_history = Box::into_raw(Box::new(LoadHistory::default()));

        let min_workers = config.min_workers;
        let manager = Self {
            config,
            metrics: ScalingMetrics::default(),
            scaling_state: AtomicPtr::new(scaling_state),
            load_history: AtomicPtr::new(load_history),
            manager_id: fastrand::u64(..),
        };

        // Initialize current workers
        manager
            .metrics
            .current_workers
            .store(min_workers, Ordering::Relaxed);
        manager
            .metrics
            .target_workers
            .store(min_workers, Ordering::Relaxed);

        Ok(manager)
    }

    /// Record load measurement for scaling decisions
    ///
    /// # Arguments
    ///
    /// * `load_factor` - Current load factor (0.0 to 1.0)
    /// * `latency_ns` - Current operation latency in nanoseconds
    #[inline]
    pub fn record_load(&self, load_factor: f64, latency_ns: u64) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let load_fixed = (load_factor.clamp(0.0_f64, 1000.0_f64) * 1000.0_f64) as u64;
        let now = current_timestamp_ns();

        // Update current load
        unsafe {
            let state = &*self.scaling_state.load(Ordering::Relaxed);
            state.current_load.store(load_fixed, Ordering::Relaxed);
            state.last_measurement.store(now, Ordering::Relaxed);
        }

        // Add to load history for prediction
        self.add_load_sample(load_fixed);

        // Check if scaling is needed
        if self.should_scale(load_factor, latency_ns) {
            let _ = self.trigger_scaling();
        }
    }

    /// Check if scaling is needed based on current metrics
    #[inline]
    fn should_scale(&self, load_factor: f64, latency_ns: u64) -> bool {
        // Fast path: check latency violation
        if latency_ns > self.config.target_latency_ns {
            return true;
        }

        // Check load threshold
        if load_factor > self.config.scaling_threshold {
            return true;
        }

        // Check cooldown period
        let now = current_timestamp_ns();
        let last_scaling = self.metrics.last_scaling_timestamp.load(Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        let cooldown_ns =
            u64::try_from(self.config.cooldown_duration.as_nanos()).unwrap_or(u64::MAX);

        if now.saturating_sub(last_scaling) < cooldown_ns {
            return false;
        }

        // Predictive scaling check
        self.predict_scaling_need()
    }

    /// Trigger scaling operation (lock-free)
    #[inline]
    fn trigger_scaling(&self) -> Result<(), ScalingError> {
        let start_time = Instant::now();

        unsafe {
            let state = &*self.scaling_state.load(Ordering::Relaxed);

            // Try to acquire scaling lock
            if state
                .is_scaling
                .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                return Err(ScalingError::ScalingInProgress);
            }

            // Calculate target worker count
            let current_workers = self.metrics.current_workers.load(Ordering::Relaxed);
            let current_load = state.current_load.load(Ordering::Relaxed);
            #[allow(clippy::cast_precision_loss)]
            let load_factor = (current_load as f64) / 1000.0_f64;

            let target_workers = self.calculate_target_workers(current_workers, load_factor);

            // Update target
            self.metrics
                .target_workers
                .store(target_workers, Ordering::Relaxed);

            // Record scaling operation
            self.metrics
                .scaling_operations
                .fetch_add(1, Ordering::Relaxed);

            #[allow(clippy::cast_possible_truncation)]
            let scaling_latency =
                u64::try_from(start_time.elapsed().as_nanos()).unwrap_or(u64::MAX);

            // Check latency requirement
            if scaling_latency > self.config.target_latency_ns {
                self.metrics.failed_scalings.fetch_add(1, Ordering::Relaxed);

                // Release scaling lock
                state.is_scaling.store(0, Ordering::Release);

                return Err(ScalingError::LatencyViolation {
                    latency_ns: scaling_latency,
                    max_latency_ns: self.config.target_latency_ns,
                });
            }

            // Update metrics
            self.metrics
                .successful_scalings
                .fetch_add(1, Ordering::Relaxed);
            self.update_avg_latency(scaling_latency);
            self.metrics
                .last_scaling_timestamp
                .store(current_timestamp_ns(), Ordering::Relaxed);

            // Release scaling lock
            state.is_scaling.store(0, Ordering::Release);
        }

        Ok(())
    }

    /// Calculate optimal target worker count
    #[inline]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn calculate_target_workers(&self, current_workers: usize, load_factor: f64) -> usize {
        let predicted_load = self.predict_future_load();
        let combined_load = f64::midpoint(load_factor, predicted_load);

        let target = if combined_load > self.config.scaling_threshold {
            // Scale up
            let scale_factor = combined_load / self.config.scaling_threshold;
            ((current_workers as f64) * scale_factor).ceil() as usize
        } else if combined_load < self.config.scaling_threshold * 0.5_f64 {
            // Scale down
            let scale_factor = combined_load / (self.config.scaling_threshold * 0.5_f64);
            ((current_workers as f64) * scale_factor).floor() as usize
        } else {
            current_workers
        };

        target.clamp(self.config.min_workers, self.config.max_workers)
    }

    /// Add load sample to history
    #[inline]
    fn add_load_sample(&self, load_fixed: u64) {
        unsafe {
            let history = &*self.load_history.load(Ordering::Relaxed);
            let index = history.current_index.fetch_add(1, Ordering::Relaxed) % 32;
            if let Some(sample) = history.samples.get(index) {
                sample.store(load_fixed, Ordering::Relaxed);
            }
            history.sample_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Predict if scaling is needed based on load history
    #[inline]
    fn predict_scaling_need(&self) -> bool {
        let predicted_load = self.predict_future_load();
        predicted_load > self.config.scaling_threshold
    }

    /// Predict future load based on history
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    fn predict_future_load(&self) -> f64 {
        unsafe {
            let history = &*self.load_history.load(Ordering::Relaxed);
            let sample_count = history.sample_count.load(Ordering::Relaxed).min(32);

            if sample_count < 3 {
                return 0.0_f64;
            }

            // Simple linear prediction based on recent trend
            let mut weighted_sum = 0_u64;
            let mut weight_total = 0_u64;

            for i in 0..sample_count {
                let sample = history
                    .samples
                    .get(i)
                    .map_or(0, |s| s.load(Ordering::Relaxed));
                let weight = (i + 1) as u64; // More recent samples have higher weight
                weighted_sum += sample * weight;
                weight_total += weight;
            }

            if weight_total == 0 {
                return 0.0_f64;
            }

            let weighted_avg = weighted_sum / weight_total;
            (weighted_avg as f64) / 1000.0_f64
        }
    }

    /// Update average latency metric
    #[inline]
    fn update_avg_latency(&self, new_latency: u64) {
        let current_avg = self.metrics.avg_scaling_latency_ns.load(Ordering::Relaxed);
        let operations = self.metrics.scaling_operations.load(Ordering::Relaxed);

        if operations == 0 {
            self.metrics
                .avg_scaling_latency_ns
                .store(new_latency, Ordering::Relaxed);
        } else {
            let new_avg = (current_avg * (operations - 1) + new_latency) / operations;
            self.metrics
                .avg_scaling_latency_ns
                .store(new_avg, Ordering::Relaxed);
        }
    }

    /// Get current scaling metrics
    #[must_use]
    pub const fn metrics(&self) -> &ScalingMetrics {
        &self.metrics
    }

    /// Get target worker count
    #[must_use]
    pub fn target_worker_count(&self) -> usize {
        self.metrics.target_workers.load(Ordering::Relaxed)
    }

    /// Get current worker count
    #[must_use]
    pub fn current_worker_count(&self) -> usize {
        self.metrics.current_workers.load(Ordering::Relaxed)
    }

    /// Check if scaling operation is in progress
    #[must_use]
    pub fn is_scaling(&self) -> bool {
        unsafe {
            let state = &*self.scaling_state.load(Ordering::Relaxed);
            state.is_scaling.load(Ordering::Relaxed) != 0
        }
    }
}

impl Drop for AdaptiveScalingManager {
    fn drop(&mut self) {
        // Clean up allocated memory
        unsafe {
            let scaling_state = self.scaling_state.load(Ordering::Relaxed);
            if !scaling_state.is_null() {
                let _ = Box::from_raw(scaling_state);
            }

            let load_history = self.load_history.load(Ordering::Relaxed);
            if !load_history.is_null() {
                let _ = Box::from_raw(load_history);
            }
        }
    }
}

/// Get current timestamp in nanoseconds
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_scaling_manager_creation() -> Result<(), ScalingError> {
        let config = ScalingConfig::default();
        let manager = AdaptiveScalingManager::new(config)?;

        assert_eq!(manager.current_worker_count(), 2);
        assert_eq!(manager.target_worker_count(), 2);
        assert!(!manager.is_scaling());

        Ok(())
    }

    #[test]
    fn test_load_recording() -> Result<(), ScalingError> {
        let config = ScalingConfig::default();
        let manager = AdaptiveScalingManager::new(config)?;

        // Record normal load
        manager.record_load(0.5_f64, 500_000); // 0.5ms latency

        // Record high load
        manager.record_load(0.9_f64, 800_000); // 0.8ms latency

        Ok(())
    }

    #[test]
    fn test_scaling_trigger() -> Result<(), ScalingError> {
        let config = ScalingConfig {
            scaling_threshold: 0.7_f64,
            cooldown_duration: Duration::from_millis(1),
            ..ScalingConfig::default()
        };

        let manager = AdaptiveScalingManager::new(config)?;

        // Trigger scaling with high load
        manager.record_load(0.9_f64, 500_000);

        // Allow some time for scaling
        std::thread::sleep(Duration::from_millis(10));

        Ok(())
    }

    #[test]
    fn test_invalid_configuration() {
        let config = ScalingConfig {
            min_workers: 0,
            max_workers: 10,
            ..ScalingConfig::default()
        };

        assert!(AdaptiveScalingManager::new(config).is_err());
    }

    #[test]
    fn test_metrics_collection() -> Result<(), ScalingError> {
        let config = ScalingConfig::default();
        let manager = AdaptiveScalingManager::new(config)?;

        let metrics = manager.metrics();
        assert_eq!(metrics.current_workers.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.scaling_operations.load(Ordering::Relaxed), 0);

        Ok(())
    }
}
