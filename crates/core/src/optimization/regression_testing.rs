//! Performance regression testing framework for `TallyIO`
//!
//! Provides automated performance monitoring and regression detection
//! to ensure consistent ultra-low latency performance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Performance regression testing errors
#[derive(Error, Debug, Clone)]
pub enum RegressionTestError {
    /// Performance regression detected
    #[error("Performance regression detected in {test_name}: current {current_ms}ms > baseline {baseline_ms}ms (threshold: {threshold_percent}%)")]
    PerformanceRegression {
        /// Test name
        test_name: String,
        /// Current performance in milliseconds
        current_ms: f64,
        /// Baseline performance in milliseconds
        baseline_ms: f64,
        /// Regression threshold percentage
        threshold_percent: f64,
    },

    /// Baseline not found
    #[error("Baseline not found for test: {test_name}")]
    BaselineNotFound {
        /// Test name
        test_name: String,
    },

    /// Test execution failed
    #[error("Test execution failed: {reason}")]
    TestExecutionFailed {
        /// Error reason
        reason: String,
    },

    /// Invalid configuration
    #[error("Invalid configuration: {field}")]
    InvalidConfiguration {
        /// Configuration field
        field: String,
    },
}

/// Performance regression testing result type
pub type RegressionTestResult<T> = Result<T, RegressionTestError>;

/// Performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    /// Test name
    pub test_name: String,
    /// Duration in nanoseconds
    pub duration_ns: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl PerformanceMeasurement {
    /// Create new performance measurement
    #[must_use]
    pub fn new(test_name: &str, duration: Duration) -> Self {
        Self {
            test_name: test_name.to_string(),
            duration_ns: u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            metadata: HashMap::new(),
        }
    }

    /// Get duration in milliseconds
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ns as f64 / 1_000_000.0_f64
    }

    /// Get duration in microseconds
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_us(&self) -> f64 {
        self.duration_ns as f64 / 1_000.0_f64
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Performance baseline storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Test name
    pub test_name: String,
    /// Baseline duration in nanoseconds
    pub baseline_duration_ns: u64,
    /// Number of samples used for baseline
    pub sample_count: u32,
    /// Standard deviation
    pub std_deviation_ns: f64,
    /// Creation timestamp
    pub created_at: u64,
}

impl PerformanceBaseline {
    /// Create new baseline from measurements
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn from_measurements(test_name: &str, measurements: &[PerformanceMeasurement]) -> Self {
        let durations: Vec<u64> = measurements.iter().map(|m| m.duration_ns).collect();
        let sum = durations.iter().sum::<u64>();
        let len = durations.len();
        let mean = sum as f64 / len as f64;

        let variance = durations
            .iter()
            .map(|&d| (d as f64 - mean).powi(2_i32))
            .sum::<f64>()
            / len as f64;
        let std_deviation = variance.sqrt();

        Self {
            test_name: test_name.to_string(),
            baseline_duration_ns: mean.round() as u64,
            sample_count: u32::try_from(len).unwrap_or(u32::MAX),
            std_deviation_ns: std_deviation,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }

    /// Get baseline duration in milliseconds
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn baseline_duration_ms(&self) -> f64 {
        self.baseline_duration_ns as f64 / 1_000_000.0_f64
    }
}

/// Regression testing configuration
#[derive(Debug, Clone)]
pub struct RegressionTestConfig {
    /// Regression threshold as percentage (e.g., 10.0 for 10%)
    pub regression_threshold_percent: f64,
    /// Minimum number of samples for baseline
    pub min_baseline_samples: u32,
    /// Maximum age of baseline in seconds
    pub max_baseline_age_seconds: u64,
    /// Enable automatic baseline updates
    pub auto_update_baseline: bool,
    /// Warmup iterations before measurement
    pub warmup_iterations: u32,
    /// Number of measurement iterations
    pub measurement_iterations: u32,
}

impl Default for RegressionTestConfig {
    fn default() -> Self {
        Self {
            regression_threshold_percent: 10.0,
            min_baseline_samples: 10,
            max_baseline_age_seconds: 7 * 24 * 3600, // 1 week
            auto_update_baseline: false,
            warmup_iterations: 5,
            measurement_iterations: 10,
        }
    }
}

impl RegressionTestConfig {
    /// Validate configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn validate(&self) -> RegressionTestResult<()> {
        if self.regression_threshold_percent <= 0.0_f64 {
            return Err(RegressionTestError::InvalidConfiguration {
                field: "regression_threshold_percent must be > 0".to_string(),
            });
        }

        if self.min_baseline_samples == 0 {
            return Err(RegressionTestError::InvalidConfiguration {
                field: "min_baseline_samples must be > 0".to_string(),
            });
        }

        if self.measurement_iterations == 0 {
            return Err(RegressionTestError::InvalidConfiguration {
                field: "measurement_iterations must be > 0".to_string(),
            });
        }

        Ok(())
    }
}

/// Performance regression tester
pub struct RegressionTester {
    /// Configuration
    config: RegressionTestConfig,
    /// Stored baselines
    baselines: HashMap<String, PerformanceBaseline>,
    /// Recent measurements
    measurements: HashMap<String, Vec<PerformanceMeasurement>>,
}

impl RegressionTester {
    /// Create new regression tester
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: RegressionTestConfig) -> RegressionTestResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            baselines: HashMap::new(),
            measurements: HashMap::new(),
        })
    }

    /// Run performance test with regression checking
    ///
    /// # Errors
    ///
    /// Returns error if test fails or regression is detected
    pub fn run_test<F, R>(&mut self, test_name: &str, test_fn: F) -> RegressionTestResult<R>
    where
        F: Fn() -> R,
    {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let _ = test_fn();
        }

        // Measure performance
        let mut durations = Vec::new();
        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            let _result = test_fn();
            let duration = start.elapsed();
            durations.push(duration);
        }

        // Calculate average duration
        let total_duration: Duration = durations.iter().sum();
        let avg_duration = total_duration / u32::try_from(durations.len()).unwrap_or(1_u32);

        // Create measurement
        let measurement = PerformanceMeasurement::new(test_name, avg_duration);

        // Store measurement
        self.measurements
            .entry(test_name.to_string())
            .or_default()
            .push(measurement.clone());

        // Check for regression
        self.check_regression(&measurement)?;

        // Run test one more time to return result
        Ok(test_fn())
    }

    /// Check for performance regression
    ///
    /// # Errors
    ///
    /// Returns error if regression is detected
    pub fn check_regression(
        &self,
        measurement: &PerformanceMeasurement,
    ) -> RegressionTestResult<()> {
        if let Some(baseline) = self.baselines.get(&measurement.test_name) {
            let current_ms = measurement.duration_ms();
            let baseline_ms = baseline.baseline_duration_ms();

            let regression_percent = ((current_ms - baseline_ms) / baseline_ms) * 100.0_f64;

            if regression_percent > self.config.regression_threshold_percent {
                return Err(RegressionTestError::PerformanceRegression {
                    test_name: measurement.test_name.clone(),
                    current_ms,
                    baseline_ms,
                    threshold_percent: self.config.regression_threshold_percent,
                });
            }
        }

        Ok(())
    }

    /// Set baseline for test
    pub fn set_baseline(&mut self, test_name: &str, baseline: PerformanceBaseline) {
        self.baselines.insert(test_name.to_string(), baseline);
    }

    /// Create baseline from recent measurements
    ///
    /// # Errors
    ///
    /// Returns error if insufficient measurements
    pub fn create_baseline_from_measurements(
        &mut self,
        test_name: &str,
    ) -> RegressionTestResult<()> {
        let measurements = self.measurements.get(test_name).ok_or_else(|| {
            RegressionTestError::BaselineNotFound {
                test_name: test_name.to_string(),
            }
        })?;

        if measurements.len() < self.config.min_baseline_samples as usize {
            return Err(RegressionTestError::InvalidConfiguration {
                field: format!(
                    "Insufficient measurements for baseline: {} < {}",
                    measurements.len(),
                    self.config.min_baseline_samples
                ),
            });
        }

        let baseline = PerformanceBaseline::from_measurements(test_name, measurements);
        self.set_baseline(test_name, baseline);

        Ok(())
    }

    /// Get baseline for test
    #[must_use]
    pub fn get_baseline(&self, test_name: &str) -> Option<&PerformanceBaseline> {
        self.baselines.get(test_name)
    }

    /// Get recent measurements for test
    #[must_use]
    pub fn get_measurements(&self, test_name: &str) -> Option<&Vec<PerformanceMeasurement>> {
        self.measurements.get(test_name)
    }

    /// Clear old measurements
    pub fn clear_old_measurements(&mut self, max_age_seconds: u64) {
        let cutoff = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs())
            - max_age_seconds;

        for measurements in self.measurements.values_mut() {
            measurements.retain(|m| m.timestamp > cutoff);
        }
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &RegressionTestConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_performance_measurement() {
        let duration = Duration::from_millis(5);
        let measurement = PerformanceMeasurement::new("test", duration);

        assert_eq!(measurement.test_name, "test");
        assert!((measurement.duration_ms() - 5.0_f64).abs() < f64::EPSILON);
        assert!((measurement.duration_us() - 5_000.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn test_regression_tester_creation() -> RegressionTestResult<()> {
        let config = RegressionTestConfig::default();
        let _tester = RegressionTester::new(config)?;
        Ok(())
    }

    #[test]
    fn test_simple_performance_test() -> RegressionTestResult<()> {
        let config = RegressionTestConfig {
            measurement_iterations: 3_u32,
            warmup_iterations: 1_u32,
            ..Default::default()
        };
        let mut tester = RegressionTester::new(config)?;

        let result = tester.run_test("simple_test", || {
            thread::sleep(Duration::from_millis(1_u64));
            42_i32
        })?;

        assert_eq!(result, 42_i32);

        let measurements = tester.get_measurements("simple_test").ok_or_else(|| {
            RegressionTestError::BaselineNotFound {
                test_name: "simple_test".to_string(),
            }
        })?;
        assert_eq!(measurements.len(), 1_usize);

        Ok(())
    }
}
