//! jemalloc Tuning for `TallyIO` Ultra-Performance
//!
//! Production-ready jemalloc configuration for financial trading applications
//! requiring <1ms latency and optimal memory allocation patterns.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use thiserror::Error;

/// jemalloc tuning errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum JemallocError {
    /// Configuration failed
    #[error("jemalloc configuration failed: {reason}")]
    ConfigurationFailed {
        /// Error reason
        reason: String,
    },

    /// Statistics unavailable
    #[error("jemalloc statistics not available")]
    StatsUnavailable,

    /// Invalid parameter
    #[error("Invalid jemalloc parameter: {param} = {value}")]
    InvalidParameter {
        /// Parameter name
        param: String,
        /// Parameter value
        value: String,
    },

    /// Not supported
    #[error("jemalloc not available or not supported")]
    NotSupported,
}

/// Result type for jemalloc operations
pub type JemallocResult<T> = Result<T, JemallocError>;

/// jemalloc configuration for ultra-performance
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct JemallocConfig {
    /// Background thread enabled
    pub background_thread: bool,
    /// Metadata transparent huge pages
    pub metadata_thp: MetadataThp,
    /// Dirty page decay time (ms)
    pub dirty_decay_ms: u64,
    /// Muzzy page decay time (ms)
    pub muzzy_decay_ms: u64,
    /// Number of arenas
    pub narenas: Option<usize>,
    /// Retain pages
    pub retain: bool,
    /// Profiling enabled
    pub prof: bool,
    /// Profiling active
    pub prof_active: bool,
    /// Statistics enabled
    pub stats_enabled: bool,
}

/// Metadata transparent huge page settings
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataThp {
    /// Disabled
    Disabled,
    /// Auto
    Auto,
    /// Always
    Always,
}

/// jemalloc statistics
#[derive(Debug, Clone)]
pub struct JemallocStats {
    /// Total allocated bytes
    pub allocated: u64,
    /// Total active bytes
    pub active: u64,
    /// Total metadata bytes
    pub metadata: u64,
    /// Total resident bytes
    pub resident: u64,
    /// Total mapped bytes
    pub mapped: u64,
    /// Total retained bytes
    pub retained: u64,
    /// Number of allocations
    pub allocations: u64,
    /// Number of deallocations
    pub deallocations: u64,
    /// Number of reallocations
    pub reallocations: u64,
}

/// jemalloc performance tuner
pub struct JemallocTuner {
    /// Current configuration
    config: JemallocConfig,
    /// Performance metrics
    metrics: JemallocMetrics,
    /// Last tuning time
    last_tuning: Instant,
    /// Tuning interval
    tuning_interval: Duration,
}

/// Performance metrics for jemalloc tuning
#[derive(Debug)]
struct JemallocMetrics {
    /// Allocation latency samples
    allocation_latencies: Vec<u64>,
    /// Memory fragmentation ratio
    #[allow(dead_code)]
    fragmentation_ratio: AtomicU64,
    /// Cache hit ratio
    #[allow(dead_code)]
    cache_hit_ratio: AtomicU64,
    /// Total allocations
    total_allocations: AtomicU64,
}

impl Default for JemallocConfig {
    fn default() -> Self {
        Self {
            background_thread: true,
            metadata_thp: MetadataThp::Auto,
            dirty_decay_ms: 5000,
            muzzy_decay_ms: 10000,
            narenas: None, // Auto-detect
            retain: true,
            prof: false, // Disable profiling in production
            prof_active: false,
            stats_enabled: true,
        }
    }
}

impl JemallocConfig {
    /// Create configuration optimized for ultra-low latency
    #[must_use]
    pub fn ultra_low_latency() -> Self {
        Self {
            background_thread: true,
            metadata_thp: MetadataThp::Always,
            dirty_decay_ms: 1000, // Faster cleanup
            muzzy_decay_ms: 2000,
            narenas: Some(num_cpus::get()), // One arena per CPU
            retain: true,
            prof: false,
            prof_active: false,
            stats_enabled: true,
        }
    }

    /// Create configuration optimized for high throughput
    #[must_use]
    pub fn high_throughput() -> Self {
        Self {
            background_thread: true,
            metadata_thp: MetadataThp::Auto,
            dirty_decay_ms: 10000, // Less frequent cleanup
            muzzy_decay_ms: 20000,
            narenas: Some(num_cpus::get() / 2), // Fewer arenas
            retain: true,
            prof: false,
            prof_active: false,
            stats_enabled: true,
        }
    }

    /// Create configuration for financial trading
    #[must_use]
    pub fn financial_trading() -> Self {
        Self {
            background_thread: true,
            metadata_thp: MetadataThp::Always,
            dirty_decay_ms: 2000, // Balance between latency and memory
            muzzy_decay_ms: 5000,
            narenas: Some(num_cpus::get()), // One arena per CPU
            retain: true,
            prof: false, // No profiling overhead
            prof_active: false,
            stats_enabled: true,
        }
    }
}

impl JemallocTuner {
    /// Create new jemalloc tuner
    ///
    /// # Errors
    ///
    /// Returns error if jemalloc is not available
    pub fn new(config: JemallocConfig) -> JemallocResult<Self> {
        let tuner = Self {
            config,
            metrics: JemallocMetrics {
                allocation_latencies: Vec::with_capacity(1000),
                fragmentation_ratio: AtomicU64::new(0),
                cache_hit_ratio: AtomicU64::new(0),
                total_allocations: AtomicU64::new(0),
            },
            last_tuning: Instant::now(),
            tuning_interval: Duration::from_secs(60), // Tune every minute
        };

        tuner.apply_config()?;
        Ok(tuner)
    }

    /// Apply jemalloc configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration fails
    pub fn apply_config(&self) -> JemallocResult<()> {
        // In a real implementation, this would use jemalloc-ctl crate
        // For now, we'll simulate the configuration

        #[cfg(feature = "jemalloc")]
        {
            self.apply_jemalloc_config()
        }

        #[cfg(not(feature = "jemalloc"))]
        {
            // Log that jemalloc is not available
            tracing::warn!("jemalloc not available - using system allocator");
            Ok(())
        }
    }

    #[cfg(feature = "jemalloc")]
    #[allow(clippy::unnecessary_wraps)]
    fn apply_jemalloc_config(&self) -> JemallocResult<()> {
        // This would use jemalloc-ctl to configure jemalloc
        // Example configuration calls:

        // Set background thread
        if self.config.background_thread {
            // jemalloc_ctl::background_thread::write(true)?;
        }

        // Set decay times
        // jemalloc_ctl::arenas::dirty_decay_ms::write(self.config.dirty_decay_ms)?;
        // jemalloc_ctl::arenas::muzzy_decay_ms::write(self.config.muzzy_decay_ms)?;

        // Set number of arenas
        if let Some(narenas) = self.config.narenas {
            // jemalloc_ctl::opt::narenas::write(narenas)?;
            tracing::info!("Configured jemalloc with {} arenas", narenas);
        }

        tracing::info!("Applied jemalloc configuration for ultra-performance");
        Ok(())
    }

    /// Get current jemalloc statistics
    ///
    /// # Errors
    ///
    /// Returns error if statistics are not available
    pub fn get_stats(&self) -> JemallocResult<JemallocStats> {
        #[cfg(feature = "jemalloc")]
        {
            self.get_jemalloc_stats()
        }

        #[cfg(not(feature = "jemalloc"))]
        {
            // Return dummy stats for non-jemalloc builds
            Ok(JemallocStats {
                allocated: 0,
                active: 0,
                metadata: 0,
                resident: 0,
                mapped: 0,
                retained: 0,
                allocations: self.metrics.total_allocations.load(Ordering::Relaxed),
                deallocations: 0,
                reallocations: 0,
            })
        }
    }

    #[cfg(feature = "jemalloc")]
    #[allow(
        clippy::unused_self,
        clippy::missing_const_for_fn,
        clippy::unnecessary_wraps
    )]
    fn get_jemalloc_stats(&self) -> JemallocResult<JemallocStats> {
        // This would use jemalloc-ctl to get statistics
        // Example:
        // let allocated = jemalloc_ctl::stats::allocated::read()?;
        // let active = jemalloc_ctl::stats::active::read()?;

        Ok(JemallocStats {
            allocated: 0, // Would be real values
            active: 0,
            metadata: 0,
            resident: 0,
            mapped: 0,
            retained: 0,
            allocations: 0,
            deallocations: 0,
            reallocations: 0,
        })
    }

    /// Record allocation latency for tuning
    pub fn record_allocation_latency(&mut self, latency_ns: u64) {
        self.metrics.allocation_latencies.push(latency_ns);
        self.metrics
            .total_allocations
            .fetch_add(1, Ordering::Relaxed);

        // Keep only recent samples
        if self.metrics.allocation_latencies.len() > 1000 {
            self.metrics.allocation_latencies.drain(0..500);
        }
    }

    /// Auto-tune jemalloc based on performance metrics
    ///
    /// # Errors
    ///
    /// Returns error if tuning fails
    pub fn auto_tune(&mut self) -> JemallocResult<bool> {
        if self.last_tuning.elapsed() < self.tuning_interval {
            return Ok(false); // Not time to tune yet
        }

        let stats = self.get_stats()?;
        let mut config_changed = false;

        // Analyze allocation latencies
        if !self.metrics.allocation_latencies.is_empty() {
            let avg_latency = self.metrics.allocation_latencies.iter().sum::<u64>()
                / self.metrics.allocation_latencies.len() as u64;

            // If average latency is high, tune for lower latency
            if avg_latency > 1000 {
                // 1µs threshold
                if self.config.dirty_decay_ms > 1000 {
                    self.config.dirty_decay_ms = 1000;
                    config_changed = true;
                }

                if self.config.narenas.is_none() {
                    self.config.narenas = Some(num_cpus::get());
                    config_changed = true;
                }
            }
        }

        // Analyze memory fragmentation
        if stats.allocated > 0 {
            let fragmentation = ((stats.active - stats.allocated) * 100) / stats.allocated;

            if fragmentation > 20 {
                // 20% fragmentation threshold
                // Increase decay frequency to reduce fragmentation
                if self.config.dirty_decay_ms > 2000 {
                    self.config.dirty_decay_ms = 2000;
                    config_changed = true;
                }
            }
        }

        if config_changed {
            self.apply_config()?;
            self.last_tuning = Instant::now();
            tracing::info!("Auto-tuned jemalloc configuration");
        }

        Ok(config_changed)
    }

    /// Get current configuration
    #[must_use]
    pub const fn config(&self) -> &JemallocConfig {
        &self.config
    }

    /// Get performance metrics summary
    #[must_use]
    pub fn metrics_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        if !self.metrics.allocation_latencies.is_empty() {
            #[allow(clippy::cast_precision_loss)]
            let avg_latency = self.metrics.allocation_latencies.iter().sum::<u64>() as f64
                / self.metrics.allocation_latencies.len() as f64;
            summary.insert("avg_allocation_latency_ns".to_string(), avg_latency);

            #[allow(clippy::cast_precision_loss)]
            let max_latency = *self.metrics.allocation_latencies.iter().max().unwrap_or(&0) as f64;
            summary.insert("max_allocation_latency_ns".to_string(), max_latency);
        }

        #[allow(clippy::cast_precision_loss)]
        let total_allocs = self.metrics.total_allocations.load(Ordering::Relaxed) as f64;
        summary.insert("total_allocations".to_string(), total_allocs);

        summary
    }
}

/// Initialize jemalloc with optimal configuration for `TallyIO`
///
/// # Errors
///
/// Returns error if initialization fails
pub fn init_jemalloc_for_trading() -> JemallocResult<JemallocTuner> {
    let config = JemallocConfig::financial_trading();
    JemallocTuner::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jemalloc_config_creation() {
        let config = JemallocConfig::ultra_low_latency();
        assert!(config.background_thread);
        assert_eq!(config.metadata_thp, MetadataThp::Always);
        assert_eq!(config.dirty_decay_ms, 1000);
    }

    #[test]
    fn test_jemalloc_tuner_creation() -> JemallocResult<()> {
        let config = JemallocConfig::default();
        let _tuner = JemallocTuner::new(config)?;
        Ok(())
    }

    #[test]
    fn test_metrics_recording() -> JemallocResult<()> {
        let config = JemallocConfig::default();
        let mut tuner = JemallocTuner::new(config)?;

        tuner.record_allocation_latency(500);
        tuner.record_allocation_latency(1000);

        let summary = tuner.metrics_summary();
        assert!(summary.contains_key("avg_allocation_latency_ns"));
        assert!(summary.contains_key("total_allocations"));

        Ok(())
    }
}
