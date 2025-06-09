//! Performance Monitoring and Validation
//!
//! Real-time performance monitoring and validation for ultra-optimized components.
//! Ensures performance targets are met in production environments.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use thiserror::Error;

/// Performance monitoring errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum PerformanceError {
    /// Performance target exceeded
    #[error("Performance target exceeded: {actual_ns}ns > {target_ns}ns for {operation}")]
    TargetExceeded {
        /// Operation name
        operation: String,
        /// Actual time in nanoseconds
        actual_ns: u128,
        /// Target time in nanoseconds
        target_ns: u128,
    },

    /// Performance degradation detected
    #[error("Performance degradation: {current_ns}ns vs baseline {baseline_ns}ns ({degradation_percent:.1}%)")]
    PerformanceDegradation {
        /// Current performance in nanoseconds
        current_ns: u128,
        /// Baseline performance in nanoseconds
        baseline_ns: u128,
        /// Degradation percentage
        degradation_percent: f64,
    },
}

/// Result type for performance operations
pub type PerformanceResult<T> = Result<T, PerformanceError>;

/// Performance targets for different operations
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    /// Ring buffer operation target (ns)
    pub ring_buffer_ns: u128,
    /// SPSC queue operation target (ns)
    pub spsc_queue_ns: u128,
    /// Memory allocation target (ns)
    pub memory_alloc_ns: u128,
    /// SIMD operation target (ns per element)
    pub simd_per_element_ns: u128,
    /// Typed queue operation target (ns)
    pub typed_queue_ns: u128,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            ring_buffer_ns: 100,
            spsc_queue_ns: 100,
            memory_alloc_ns: 50,
            simd_per_element_ns: 1,
            typed_queue_ns: 200,
        }
    }
}

/// Performance statistics
#[derive(Debug, Default)]
pub struct PerformanceStats {
    /// Total operations measured
    pub total_operations: AtomicU64,
    /// Operations within target
    pub operations_within_target: AtomicU64,
    /// Total time spent (nanoseconds)
    pub total_time_ns: AtomicU64,
    /// Minimum time observed (nanoseconds)
    pub min_time_ns: AtomicU64,
    /// Maximum time observed (nanoseconds)
    pub max_time_ns: AtomicU64,
}

impl PerformanceStats {
    /// Get success rate (operations within target)
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 1.0_f64;
        }

        let within_target = self.operations_within_target.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        {
            within_target as f64 / total as f64
        }
    }

    /// Get average time (nanoseconds)
    #[must_use]
    pub fn average_time_ns(&self) -> f64 {
        let total = self.total_operations.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0_f64;
        }

        let total_time = self.total_time_ns.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        {
            total_time as f64 / total as f64
        }
    }

    /// Get operations per second
    #[must_use]
    pub fn operations_per_second(&self, elapsed_seconds: f64) -> f64 {
        if elapsed_seconds <= 0.0_f64 {
            return 0.0_f64;
        }

        let total = self.total_operations.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        {
            total as f64 / elapsed_seconds
        }
    }
}

/// Performance monitor for real-time validation
pub struct PerformanceMonitor {
    /// Performance targets
    targets: PerformanceTargets,
    /// Statistics by operation type
    ring_buffer_stats: Arc<PerformanceStats>,
    spsc_queue_stats: Arc<PerformanceStats>,
    memory_alloc_stats: Arc<PerformanceStats>,
    simd_stats: Arc<PerformanceStats>,
    typed_queue_stats: Arc<PerformanceStats>,
    /// Monitor start time
    start_time: Instant,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    #[must_use]
    pub fn new() -> Self {
        Self::with_targets(PerformanceTargets::default())
    }

    /// Create performance monitor with custom targets
    #[must_use]
    pub fn with_targets(targets: PerformanceTargets) -> Self {
        Self {
            targets,
            ring_buffer_stats: Arc::new(PerformanceStats::default()),
            spsc_queue_stats: Arc::new(PerformanceStats::default()),
            memory_alloc_stats: Arc::new(PerformanceStats::default()),
            simd_stats: Arc::new(PerformanceStats::default()),
            typed_queue_stats: Arc::new(PerformanceStats::default()),
            start_time: Instant::now(),
        }
    }

    /// Measure and validate ring buffer operation
    ///
    /// # Errors
    ///
    /// Returns error if operation exceeds performance target
    pub fn measure_ring_buffer<F, R>(&self, operation: F) -> PerformanceResult<R>
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();

        Self::record_measurement(
            &self.ring_buffer_stats,
            elapsed,
            self.targets.ring_buffer_ns,
            "ring_buffer",
        )?;

        Ok(result)
    }

    /// Measure and validate SPSC queue operation
    ///
    /// # Errors
    ///
    /// Returns error if operation exceeds performance target
    pub fn measure_spsc_queue<F, R>(&self, operation: F) -> PerformanceResult<R>
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();

        Self::record_measurement(
            &self.spsc_queue_stats,
            elapsed,
            self.targets.spsc_queue_ns,
            "spsc_queue",
        )?;

        Ok(result)
    }

    /// Measure and validate memory allocation
    ///
    /// # Errors
    ///
    /// Returns error if operation exceeds performance target
    pub fn measure_memory_alloc<F, R>(&self, operation: F) -> PerformanceResult<R>
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();

        Self::record_measurement(
            &self.memory_alloc_stats,
            elapsed,
            self.targets.memory_alloc_ns,
            "memory_alloc",
        )?;

        Ok(result)
    }

    /// Measure and validate SIMD operation
    ///
    /// # Errors
    ///
    /// Returns error if operation exceeds performance target
    pub fn measure_simd<F, R>(&self, operation: F, element_count: usize) -> PerformanceResult<R>
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();

        let target_ns = self.targets.simd_per_element_ns * element_count as u128;

        Self::record_measurement(&self.simd_stats, elapsed, target_ns, "simd")?;

        Ok(result)
    }

    /// Measure and validate typed queue operation
    ///
    /// # Errors
    ///
    /// Returns error if operation exceeds performance target
    pub fn measure_typed_queue<F, R>(&self, operation: F) -> PerformanceResult<R>
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let elapsed = start.elapsed();

        Self::record_measurement(
            &self.typed_queue_stats,
            elapsed,
            self.targets.typed_queue_ns,
            "typed_queue",
        )?;

        Ok(result)
    }

    /// Record measurement and validate against target
    ///
    /// # Errors
    ///
    /// Returns error if operation exceeds performance target
    fn record_measurement(
        stats: &Arc<PerformanceStats>,
        elapsed: Duration,
        target_ns: u128,
        operation: &str,
    ) -> PerformanceResult<()> {
        let elapsed_ns = elapsed.as_nanos();

        // Update statistics
        stats.total_operations.fetch_add(1, Ordering::Relaxed);

        // Safely convert u128 to u64, clamping to max value if needed
        let elapsed_ns_u64 = u64::try_from(elapsed_ns).unwrap_or(u64::MAX);
        stats
            .total_time_ns
            .fetch_add(elapsed_ns_u64, Ordering::Relaxed);

        // Update min/max
        let current_min = stats.min_time_ns.load(Ordering::Relaxed);
        if current_min == 0 || elapsed_ns < u128::from(current_min) {
            stats.min_time_ns.store(elapsed_ns_u64, Ordering::Relaxed);
        }

        let current_max = stats.max_time_ns.load(Ordering::Relaxed);
        if elapsed_ns > u128::from(current_max) {
            stats.max_time_ns.store(elapsed_ns_u64, Ordering::Relaxed);
        }

        // Check if within target
        if elapsed_ns <= target_ns {
            stats
                .operations_within_target
                .fetch_add(1, Ordering::Relaxed);
            Ok(())
        } else {
            Err(PerformanceError::TargetExceeded {
                operation: operation.to_string(),
                actual_ns: elapsed_ns,
                target_ns,
            })
        }
    }

    /// Get comprehensive performance report
    #[must_use]
    pub fn get_performance_report(&self) -> PerformanceReport {
        let elapsed_seconds = self.start_time.elapsed().as_secs_f64();

        PerformanceReport {
            elapsed_seconds,
            ring_buffer: Self::create_operation_report(&self.ring_buffer_stats, elapsed_seconds),
            spsc_queue: Self::create_operation_report(&self.spsc_queue_stats, elapsed_seconds),
            memory_alloc: Self::create_operation_report(&self.memory_alloc_stats, elapsed_seconds),
            simd: Self::create_operation_report(&self.simd_stats, elapsed_seconds),
            typed_queue: Self::create_operation_report(&self.typed_queue_stats, elapsed_seconds),
        }
    }

    /// Create operation report
    fn create_operation_report(
        stats: &Arc<PerformanceStats>,
        elapsed_seconds: f64,
    ) -> OperationReport {
        OperationReport {
            total_operations: stats.total_operations.load(Ordering::Relaxed),
            success_rate: stats.success_rate(),
            average_time_ns: stats.average_time_ns(),
            min_time_ns: stats.min_time_ns.load(Ordering::Relaxed),
            max_time_ns: stats.max_time_ns.load(Ordering::Relaxed),
            operations_per_second: stats.operations_per_second(elapsed_seconds),
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance report for all operations
#[derive(Debug)]
pub struct PerformanceReport {
    /// Total elapsed time
    pub elapsed_seconds: f64,
    /// Ring buffer performance
    pub ring_buffer: OperationReport,
    /// SPSC queue performance
    pub spsc_queue: OperationReport,
    /// Memory allocation performance
    pub memory_alloc: OperationReport,
    /// SIMD performance
    pub simd: OperationReport,
    /// Typed queue performance
    pub typed_queue: OperationReport,
}

/// Performance report for a specific operation
#[derive(Debug)]
pub struct OperationReport {
    /// Total operations performed
    pub total_operations: u64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average time in nanoseconds
    pub average_time_ns: f64,
    /// Minimum time in nanoseconds
    pub min_time_ns: u64,
    /// Maximum time in nanoseconds
    pub max_time_ns: u64,
    /// Operations per second
    pub operations_per_second: f64,
}
