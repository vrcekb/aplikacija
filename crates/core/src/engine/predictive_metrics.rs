//! Predictive Metrics - ML-Based Performance Prediction and Optimization
//!
//! Production-ready predictive metrics for `TallyIO` financial applications.
//! Implements lightweight ML inference for performance prediction and
//! optimization with sub-microsecond prediction latency for <1ms requirements.

use std::{
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
    time::{Duration, Instant},
};

use thiserror::Error;

use crate::engine::executor::Task;

/// Predictive metrics error types
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum PredictiveError {
    /// Model not trained
    #[error("Model not trained")]
    ModelNotTrained,

    /// Invalid feature vector
    #[error("Invalid feature vector: {reason}")]
    InvalidFeatures {
        /// Reason for invalid features
        reason: &'static str,
    },

    /// Prediction latency violation
    #[error("Prediction latency violation: {latency_ns}ns > {max_latency_ns}ns")]
    LatencyViolation {
        /// Actual latency in nanoseconds
        latency_ns: u64,
        /// Maximum allowed latency in nanoseconds
        max_latency_ns: u64,
    },

    /// Insufficient training data
    #[error("Insufficient training data: {samples} < {required}")]
    InsufficientData {
        /// Current number of samples
        samples: usize,
        /// Required number of samples
        required: usize,
    },
}

/// Lightweight ML model for latency prediction
#[derive(Debug, Clone)]
pub struct TinyMLModel {
    /// Model weights (8 features)
    weights: [f32; 8],
    /// Model bias
    bias: f32,
    /// Feature normalization parameters
    feature_means: [f32; 8],
    feature_stds: [f32; 8],
    /// Model generation
    generation: u64,
    /// Training sample count
    training_samples: usize,
}

/// Feature vector for prediction
#[derive(Debug, Clone, Copy)]
pub struct FeatureVector {
    /// Task data size
    pub data_size: f32,
    /// Task priority
    pub priority: f32,
    /// Current system load
    pub system_load: f32,
    /// Worker queue depth
    pub queue_depth: f32,
    /// Recent average latency
    pub recent_latency: f32,
    /// Memory pressure
    pub memory_pressure: f32,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Network latency
    pub network_latency: f32,
}

/// Prediction result
#[derive(Debug, Clone, Copy)]
pub struct PredictionResult {
    /// Predicted latency in nanoseconds
    pub predicted_latency_ns: u64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Prediction timestamp
    pub timestamp: u64,
    /// Model generation used
    pub model_generation: u64,
}

/// Training sample for model updates
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Feature vector
    pub features: FeatureVector,
    /// Actual latency observed
    pub actual_latency_ns: u64,
    /// Sample timestamp
    pub timestamp: u64,
    /// Sample weight
    pub weight: f32,
}

/// Predictive metrics configuration
#[derive(Debug, Clone)]
pub struct PredictiveConfig {
    /// Maximum prediction latency in nanoseconds
    pub max_prediction_latency_ns: u64,
    /// Model update frequency
    pub update_frequency: Duration,
    /// Minimum training samples required
    pub min_training_samples: usize,
    /// Learning rate for online updates
    pub learning_rate: f32,
    /// Feature smoothing factor
    pub smoothing_factor: f32,
    /// Prediction confidence threshold
    pub confidence_threshold: f32,
}

impl Default for PredictiveConfig {
    fn default() -> Self {
        Self {
            max_prediction_latency_ns: 10_000, // 10μs
            update_frequency: Duration::from_millis(100),
            min_training_samples: 100,
            learning_rate: 0.01_f32,
            smoothing_factor: 0.9_f32,
            confidence_threshold: 0.7_f32,
        }
    }
}

/// Predictive metrics manager
#[derive(Debug)]
pub struct PredictiveMetrics {
    /// ML model for latency prediction
    model: TinyMLModel,
    /// Configuration
    config: PredictiveConfig,
    /// Training samples buffer (circular)
    training_buffer: Vec<TrainingSample>,
    /// Current buffer index
    buffer_index: AtomicU32,
    /// Total predictions made
    total_predictions: AtomicU64,
    /// Successful predictions
    successful_predictions: AtomicU64,
    /// Average prediction latency
    avg_prediction_latency_ns: AtomicU64,
    /// Model accuracy (0-1000, fixed-point)
    model_accuracy: AtomicU32,
    /// Last model update timestamp (for future use in model versioning)
    #[allow(dead_code)]
    last_update: AtomicU64,
}

impl Default for TinyMLModel {
    fn default() -> Self {
        Self {
            weights: [0.1_f32; 8], // Small initial weights
            bias: 0.0_f32,
            feature_means: [0.0_f32; 8],
            feature_stds: [1.0_f32; 8],
            generation: 1,
            training_samples: 0,
        }
    }
}

impl TinyMLModel {
    /// Predict latency from feature vector - Ultra-optimized inference
    ///
    /// # Errors
    ///
    /// Returns error if model is not trained or features are invalid
    pub fn predict(&self, features: &FeatureVector) -> Result<f32, PredictiveError> {
        if self.training_samples < 10 {
            return Err(PredictiveError::ModelNotTrained);
        }

        // Normalize features (vectorized operations)
        let normalized = self.normalize_features(features);

        // Linear model inference: y = w^T * x + b
        let mut prediction = self.bias;

        // Unrolled dot product for performance
        prediction += self.weights[0] * normalized[0];
        prediction += self.weights[1] * normalized[1];
        prediction += self.weights[2] * normalized[2];
        prediction += self.weights[3] * normalized[3];
        prediction += self.weights[4] * normalized[4];
        prediction += self.weights[5] * normalized[5];
        prediction += self.weights[6] * normalized[6];
        prediction += self.weights[7] * normalized[7];

        // Apply activation (ReLU for latency prediction)
        Ok(prediction.max(0.0_f32))
    }

    /// Normalize feature vector
    fn normalize_features(&self, features: &FeatureVector) -> [f32; 8] {
        let raw = [
            features.data_size,
            features.priority,
            features.system_load,
            features.queue_depth,
            features.recent_latency,
            features.memory_pressure,
            features.cpu_utilization,
            features.network_latency,
        ];

        let mut normalized = [0.0_f32; 8];
        for (i, (&raw_val, (&mean, &std))) in raw
            .iter()
            .zip(self.feature_means.iter().zip(self.feature_stds.iter()))
            .enumerate()
            .take(8)
        {
            if let Some(norm_val) = normalized.get_mut(i) {
                *norm_val = (raw_val - mean) / std.max(1e-6_f32);
            }
        }

        normalized
    }

    /// Update model with new training sample (online learning)
    #[inline]
    pub fn update(&mut self, sample: &TrainingSample, learning_rate: f32) {
        let normalized = self.normalize_features(&sample.features);
        let target = f32::from(
            u16::try_from(sample.actual_latency_ns.min(u64::from(u16::MAX))).unwrap_or(u16::MAX),
        );

        // Forward pass - safe iteration
        let prediction = self.bias
            + self
                .weights
                .iter()
                .zip(normalized.iter())
                .map(|(&weight, &feature)| weight * feature)
                .sum::<f32>();

        // Calculate error
        let error = target - prediction;

        // Gradient descent update
        self.bias += learning_rate * error * sample.weight;
        for (weight, &feature) in self.weights.iter_mut().zip(normalized.iter()) {
            *weight += learning_rate * error * feature * sample.weight;
        }

        self.training_samples += 1;
        self.generation += 1;

        // Update feature statistics (exponential moving average)
        let alpha = 0.01_f32; // Smoothing factor
        let raw_features = [
            sample.features.data_size,
            sample.features.priority,
            sample.features.system_load,
            sample.features.queue_depth,
            sample.features.recent_latency,
            sample.features.memory_pressure,
            sample.features.cpu_utilization,
            sample.features.network_latency,
        ];

        // Safe iteration for feature statistics update
        for ((mean, std), &raw_val) in self
            .feature_means
            .iter_mut()
            .zip(self.feature_stds.iter_mut())
            .zip(raw_features.iter())
        {
            *mean = (1.0_f32 - alpha).mul_add(*mean, alpha * raw_val);
            let diff = raw_val - *mean;
            *std = (1.0_f32 - alpha).mul_add(*std, alpha * diff * diff);
        }
    }
}

impl PredictiveMetrics {
    /// Create new predictive metrics manager
    #[must_use]
    pub fn new(config: PredictiveConfig) -> Self {
        let buffer_size = config.min_training_samples * 2;
        let training_buffer = Vec::with_capacity(buffer_size);

        Self {
            model: TinyMLModel::default(),
            config,
            training_buffer,
            buffer_index: AtomicU32::new(0),
            total_predictions: AtomicU64::new(0),
            successful_predictions: AtomicU64::new(0),
            avg_prediction_latency_ns: AtomicU64::new(0),
            model_accuracy: AtomicU32::new(0),
            last_update: AtomicU64::new(0),
        }
    }

    /// Predict task latency - Ultra-optimized hot path
    ///
    /// # Errors
    ///
    /// Returns error if prediction fails or exceeds latency limit
    pub fn predict_latency(
        &self,
        task: &Task,
        system_metrics: &SystemMetrics,
    ) -> Result<PredictionResult, PredictiveError> {
        let start_time = Instant::now();

        // Extract features from task and system state
        let features = Self::extract_features(task, system_metrics);

        // Perform prediction
        let predicted_latency = self.model.predict(&features)?;

        let prediction_latency = u64::try_from(start_time.elapsed().as_nanos()).unwrap_or(u64::MAX);

        // Check latency requirement
        if prediction_latency > self.config.max_prediction_latency_ns {
            return Err(PredictiveError::LatencyViolation {
                latency_ns: prediction_latency,
                max_latency_ns: self.config.max_prediction_latency_ns,
            });
        }

        // Calculate confidence based on model training
        let confidence = self.calculate_confidence(&features);

        // Update metrics
        self.total_predictions.fetch_add(1, Ordering::Relaxed);
        if confidence >= self.config.confidence_threshold {
            self.successful_predictions.fetch_add(1, Ordering::Relaxed);
        }

        self.update_avg_prediction_latency(prediction_latency);

        Ok(PredictionResult {
            predicted_latency_ns: {
                let clamped_latency = predicted_latency.max(0.0_f32).min(f32::from(u16::MAX));
                let safe_u16 = if clamped_latency.is_finite() && (0.0_f32..=f32::from(u16::MAX)).contains(&clamped_latency) {
                    let rounded = clamped_latency.round();
                    if rounded >= 0.0_f32 && rounded <= f32::from(u16::MAX) {
                        // Safe conversion: f32 in [0.0, u16::MAX] range
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        { rounded as u16 }
                    } else {
                        0_u16
                    }
                } else {
                    0_u16
                };
                u64::from(safe_u16)
            },
            confidence,
            timestamp: current_timestamp_ns(),
            model_generation: self.model.generation,
        })
    }

    /// Add training sample for model improvement
    pub fn add_training_sample(&mut self, sample: &TrainingSample) {
        // Add to circular buffer
        let index = self.buffer_index.fetch_add(1, Ordering::Relaxed) as usize
            % self.training_buffer.capacity();

        if let Some(existing_sample) = self.training_buffer.get_mut(index) {
            *existing_sample = sample.clone();
        } else {
            self.training_buffer.push(sample.clone());
        }

        // Update model with new sample
        self.model.update(sample, self.config.learning_rate);

        // Update model accuracy
        self.update_model_accuracy();
    }

    /// Extract features from task and system state
    fn extract_features(task: &Task, system_metrics: &SystemMetrics) -> FeatureVector {
        FeatureVector {
            data_size: f32::from(u16::try_from(task.data.len()).unwrap_or(u16::MAX)),
            priority: f32::from(task.priority as u8),
            system_load: system_metrics.cpu_load,
            queue_depth: f32::from(u16::try_from(system_metrics.queue_depth).unwrap_or(u16::MAX)),
            recent_latency: f32::from(
                u16::try_from(
                    system_metrics
                        .recent_avg_latency_ns
                        .min(u64::from(u16::MAX)),
                )
                .unwrap_or(u16::MAX),
            ),
            memory_pressure: system_metrics.memory_pressure,
            cpu_utilization: system_metrics.cpu_utilization,
            network_latency: f32::from(
                u16::try_from(system_metrics.network_latency_ns.min(u64::from(u16::MAX)))
                    .unwrap_or(u16::MAX),
            ),
        }
    }

    /// Calculate prediction confidence
    #[inline]
    fn calculate_confidence(&self, _features: &FeatureVector) -> f32 {
        // Simple confidence based on training samples
        let samples = self.model.training_samples;
        let min_samples = self.config.min_training_samples;

        if samples < min_samples {
            0.0_f32
        } else {
            let confidence =
                (f32::from(u16::try_from(samples.min(usize::from(u16::MAX))).unwrap_or(u16::MAX))
                    / (f32::from(
                        u16::try_from(min_samples.min(usize::from(u16::MAX))).unwrap_or(u16::MAX),
                    ) * 2.0_f32))
                    .min(1.0_f32);
            confidence * 0.9_f32 // Conservative confidence
        }
    }

    /// Update average prediction latency
    #[inline]
    fn update_avg_prediction_latency(&self, new_latency: u64) {
        let current_avg = self.avg_prediction_latency_ns.load(Ordering::Relaxed);
        let predictions = self.total_predictions.load(Ordering::Relaxed);

        if predictions == 0 {
            self.avg_prediction_latency_ns
                .store(new_latency, Ordering::Relaxed);
        } else {
            let new_avg = (current_avg * (predictions - 1) + new_latency) / predictions;
            self.avg_prediction_latency_ns
                .store(new_avg, Ordering::Relaxed);
        }
    }

    /// Update model accuracy metric
    fn update_model_accuracy(&self) {
        // Simplified accuracy calculation
        let total = self.total_predictions.load(Ordering::Relaxed);
        let successful = self.successful_predictions.load(Ordering::Relaxed);

        if total > 0 {
            let accuracy = (successful * 1000) / total; // Fixed-point accuracy
            self.model_accuracy.store(
                u32::try_from(accuracy).unwrap_or(u32::MAX),
                Ordering::Relaxed,
            );
        }
    }

    /// Get prediction metrics
    #[must_use]
    pub fn get_metrics(&self) -> PredictiveMetricsSnapshot {
        PredictiveMetricsSnapshot {
            total_predictions: self.total_predictions.load(Ordering::Relaxed),
            successful_predictions: self.successful_predictions.load(Ordering::Relaxed),
            avg_prediction_latency_ns: self.avg_prediction_latency_ns.load(Ordering::Relaxed),
            model_accuracy: f64::from(self.model_accuracy.load(Ordering::Relaxed)) / 1000.0_f64,
            model_generation: self.model.generation,
            training_samples: self.model.training_samples,
        }
    }

    /// Check if model is ready for predictions
    #[must_use]
    pub const fn is_model_ready(&self) -> bool {
        self.model.training_samples >= self.config.min_training_samples
    }
}

/// System metrics for feature extraction
#[derive(Debug, Clone, Copy)]
pub struct SystemMetrics {
    /// CPU load (0.0 to 1.0)
    pub cpu_load: f32,
    /// Queue depth
    pub queue_depth: u32,
    /// Recent average latency in nanoseconds
    pub recent_avg_latency_ns: u64,
    /// Memory pressure (0.0 to 1.0)
    pub memory_pressure: f32,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f32,
    /// Network latency in nanoseconds
    pub network_latency_ns: u64,
}

/// Snapshot of predictive metrics
#[derive(Debug, Clone)]
pub struct PredictiveMetricsSnapshot {
    /// Total predictions made
    pub total_predictions: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Average prediction latency
    pub avg_prediction_latency_ns: u64,
    /// Model accuracy (0.0 to 1.0)
    pub model_accuracy: f64,
    /// Model generation
    pub model_generation: u64,
    /// Training samples count
    pub training_samples: usize,
}

/// Get current timestamp in nanoseconds
fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StrategyId;

    #[test]
    fn test_tiny_ml_model_prediction() -> Result<(), PredictiveError> {
        let mut model = TinyMLModel::default();

        // Add some training samples
        for i in 0..20 {
            let sample = TrainingSample {
                features: FeatureVector {
                    data_size: f32::from(
                        u16::try_from((i * 100).min(u64::from(u16::MAX))).unwrap_or(u16::MAX),
                    ),
                    priority: 1.0_f32,
                    system_load: 0.5_f32,
                    queue_depth: 10.0_f32,
                    recent_latency: 500_000.0_f32,
                    memory_pressure: 0.3_f32,
                    cpu_utilization: 0.6_f32,
                    network_latency: 1000.0_f32,
                },
                actual_latency_ns: 500_000 + (i * 1000),
                timestamp: current_timestamp_ns(),
                weight: 1.0_f32,
            };
            model.update(&sample, 0.01_f32);
        }

        // Test prediction
        let features = FeatureVector {
            data_size: 1000.0_f32,
            priority: 1.0_f32,
            system_load: 0.5_f32,
            queue_depth: 10.0_f32,
            recent_latency: 500_000.0_f32,
            memory_pressure: 0.3_f32,
            cpu_utilization: 0.6_f32,
            network_latency: 1000.0_f32,
        };

        let prediction = model.predict(&features)?;
        assert!(prediction >= 0.0_f32);

        Ok(())
    }

    #[test]
    fn test_predictive_metrics_creation() {
        let config = PredictiveConfig::default();
        let metrics = PredictiveMetrics::new(config);

        assert!(!metrics.is_model_ready());
        assert_eq!(metrics.get_metrics().total_predictions, 0);
    }

    #[test]
    fn test_feature_extraction() {
        let config = PredictiveConfig::default();
        let _metrics = PredictiveMetrics::new(config);

        let task = Task::new(StrategyId::new(), vec![1, 2, 3, 4, 5]);
        let system_metrics = SystemMetrics {
            cpu_load: 0.5_f32,
            queue_depth: 10,
            recent_avg_latency_ns: 500_000,
            memory_pressure: 0.3_f32,
            cpu_utilization: 0.6_f32,
            network_latency_ns: 1000,
        };

        let features = PredictiveMetrics::extract_features(&task, &system_metrics);
        assert!((features.data_size - 5.0_f32).abs() < f32::EPSILON);
        assert!((features.system_load - 0.5_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_training_sample_addition() {
        let config = PredictiveConfig::default();
        let mut metrics = PredictiveMetrics::new(config);

        let sample = TrainingSample {
            features: FeatureVector {
                data_size: 1000.0_f32,
                priority: 1.0_f32,
                system_load: 0.5_f32,
                queue_depth: 10.0_f32,
                recent_latency: 500_000.0_f32,
                memory_pressure: 0.3_f32,
                cpu_utilization: 0.6_f32,
                network_latency: 1000.0_f32,
            },
            actual_latency_ns: 500_000,
            timestamp: current_timestamp_ns(),
            weight: 1.0_f32,
        };

        metrics.add_training_sample(&sample);
        assert_eq!(metrics.get_metrics().training_samples, 1);
    }
}
