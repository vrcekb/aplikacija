//! `TallyIO` Core Prelude
//!
//! Common imports for `TallyIO` core functionality.
//! Import this module to get access to the most commonly used types and traits.

// Re-export core types
pub use crate::config::{CoreConfig, EngineConfig, MempoolConfig, OptimizationConfig, StateConfig};
pub use crate::error::{
    CoreError, CoreResult, CriticalError, EngineError, MempoolError, OptimizationError, StateError,
};
pub use crate::types::{Address, BlockNumber, Gas, Price, PrivateKey, TxHash};

// Re-export commonly used external types
pub use async_trait::async_trait;
pub use futures::prelude::*;
pub use serde::{Deserialize, Serialize};
pub use std::sync::Arc;
pub use std::time::{Duration, Instant, SystemTime};
pub use tokio::sync::{Mutex, RwLock};
pub use tracing::{debug, error, info, instrument, warn};
pub use uuid::Uuid;

// Re-export performance types
pub use dashmap::DashMap;
pub use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

// Re-export validation
pub use garde::Validate;

/// Common result type alias
pub type Result<T> = CoreResult<T>;

/// Common trait for components that can be started and stopped
#[async_trait]
pub trait Lifecycle: Send + Sync {
    /// Start the component
    ///
    /// # Errors
    /// Returns error if component fails to start
    async fn start(&mut self) -> CoreResult<()>;

    /// Stop the component
    ///
    /// # Errors
    /// Returns error if component fails to stop gracefully
    async fn stop(&mut self) -> CoreResult<()>;

    /// Check if component is running
    fn is_running(&self) -> bool;

    /// Get component status
    fn status(&self) -> ComponentStatus;
}

/// Component status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComponentStatus {
    /// Component is not initialized
    Uninitialized,
    /// Component is starting up
    Starting,
    /// Component is running normally
    Running,
    /// Component is stopping
    Stopping,
    /// Component is stopped
    Stopped,
    /// Component has failed
    Failed,
}

impl ComponentStatus {
    /// Check if component is operational
    #[must_use]
    pub const fn is_operational(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// Check if component is transitioning
    #[must_use]
    pub const fn is_transitioning(&self) -> bool {
        matches!(self, Self::Starting | Self::Stopping)
    }

    /// Check if component has failed
    #[must_use]
    pub const fn has_failed(&self) -> bool {
        matches!(self, Self::Failed)
    }
}

/// Common trait for components that provide health checks
pub trait HealthCheck: Send + Sync {
    /// Perform health check
    ///
    /// # Errors
    /// Returns error if health check fails
    fn health_check(&self) -> CoreResult<HealthStatus>;
}

/// Health status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Component name
    pub component: String,
    /// Overall health status
    pub status: HealthLevel,
    /// Health check timestamp
    pub timestamp: SystemTime,
    /// Additional details
    pub details: Option<String>,
    /// Performance metrics
    pub metrics: Option<HealthMetrics>,
}

/// Health level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthLevel {
    /// Component is healthy
    Healthy,
    /// Component is degraded but functional
    Degraded,
    /// Component is unhealthy
    Unhealthy,
    /// Component status is unknown
    Unknown,
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// CPU usage percentage (0-100)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Number of active connections
    pub active_connections: u32,
    /// Request rate (requests per second)
    pub request_rate: f64,
    /// Error rate (errors per second)
    pub error_rate: f64,
    /// Average response time in microseconds
    pub avg_response_time_us: u64,
}

/// Common trait for configurable components
pub trait Configurable<T> {
    /// Update component configuration
    ///
    /// # Errors
    /// Returns error if configuration is invalid or update fails
    fn update_config(&mut self, config: T) -> CoreResult<()>;

    /// Get current configuration
    fn config(&self) -> &T;
}

/// Common trait for components that provide metrics
pub trait MetricsProvider: Send + Sync {
    /// Get component metrics
    fn metrics(&self) -> ComponentMetrics;
}

/// Component metrics
#[derive(Debug, Clone)]
pub struct ComponentMetrics {
    /// Component name
    pub component: String,
    /// Metrics collection timestamp
    pub timestamp: SystemTime,
    /// Counter metrics
    pub counters: DashMap<String, u64>,
    /// Gauge metrics
    pub gauges: DashMap<String, f64>,
    /// Histogram metrics
    pub histograms: DashMap<String, Vec<f64>>,
}

impl ComponentMetrics {
    /// Create new metrics for component
    pub fn new(component: impl Into<String>) -> Self {
        Self {
            component: component.into(),
            timestamp: SystemTime::now(),
            counters: DashMap::new(),
            gauges: DashMap::new(),
            histograms: DashMap::new(),
        }
    }

    /// Increment counter
    pub fn increment_counter(&self, name: &str, value: u64) {
        self.counters
            .entry(name.to_string())
            .and_modify(|v| *v += value)
            .or_insert(value);
    }

    /// Set gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        self.gauges.insert(name.to_string(), value);
    }

    /// Record histogram value
    pub fn record_histogram(&self, name: &str, value: f64) {
        self.histograms
            .entry(name.to_string())
            .and_modify(|v| v.push(value))
            .or_insert_with(|| vec![value]);
    }
}

/// Macro for creating component with standard traits
#[macro_export]
macro_rules! component {
    ($name:ident, $config:ty) => {
        impl $crate::prelude::Configurable<$config> for $name {
            fn update_config(&mut self, config: $config) -> $crate::prelude::CoreResult<()> {
                config.validate(&())?;
                self.config = config;
                Ok(())
            }

            fn config(&self) -> &$config {
                &self.config
            }
        }

        impl $crate::prelude::MetricsProvider for $name {
            fn metrics(&self) -> $crate::prelude::ComponentMetrics {
                self.metrics.clone()
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_status() {
        assert!(ComponentStatus::Running.is_operational());
        assert!(!ComponentStatus::Failed.is_operational());
        assert!(ComponentStatus::Starting.is_transitioning());
        assert!(ComponentStatus::Failed.has_failed());
    }

    #[test]
    fn test_component_metrics() {
        let metrics = ComponentMetrics::new("test_component");

        metrics.increment_counter("requests", 1);
        metrics.increment_counter("requests", 2);

        metrics.set_gauge("cpu_usage", 45.5_f64);
        metrics.record_histogram("latency", 1.5_f64);

        // Test counter - production-ready error handling without unwrap
        if let Some(counter) = metrics.counters.get("requests") {
            assert_eq!(counter.value(), &3);
        } else {
            #[allow(clippy::panic)]
            {
                panic!("Counter should exist in test");
            }
        }

        // Use approximate comparison for floating point without unwrap
        if let Some(gauge) = metrics.gauges.get("cpu_usage") {
            assert!((gauge.value() - 45.5_f64).abs() < f64::EPSILON);
        } else {
            #[allow(clippy::panic)]
            {
                panic!("Gauge should exist in test");
            }
        }

        // Test histogram - production-ready pattern without unwrap
        assert!(metrics.histograms.contains_key("latency"));
        {
            let histogram = metrics.histograms.get("latency");
            if let Some(histogram) = histogram {
                let histogram_value = histogram.value().clone();
                assert_eq!(histogram_value, vec![1.5_f64]);
            } else {
                #[allow(clippy::panic)]
                {
                    panic!("Histogram should exist in test");
                }
            }
        }
    }
}
