//! `TallyIO` Engine - Ultra-Performance MEV/Liquidation Engine
//!
//! Production-ready execution engine for crypto MEV bot and position liquidator.
//! Designed for <1ms latency with zero-panic guarantee.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Engine Core                               │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
//! │  │  Executor   │  │  Scheduler  │  │   Worker    │         │
//! │  │             │  │             │  │   Pool      │         │
//! │  └─────────────┘  └─────────────┘  └─────────────┘         │
//! └─────────────────────────────────────────────────────────────┘
//!           │                    │                    │
//! ┌─────────▼────────┐  ┌────────▼────────┐  ┌────────▼────────┐
//! │   MEV Strategy   │  │   Liquidation   │  │   Risk Engine   │
//! │   Execution      │  │   Strategy      │  │                 │
//! └──────────────────┘  └─────────────────┘  └─────────────────┘
//! ```

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use thiserror::Error;

use crate::types::{StrategyId, TaskId, WorkerId};

pub mod adaptive_scaling;
pub mod backpressure;
pub mod batch_processor;
pub mod circuit_breaker;
pub mod enhanced_load_balancer;
pub mod executor;
pub mod load_testing;
pub mod numa_scheduler;
pub mod predictive_metrics;
pub mod scheduler;
pub mod ultra;
pub mod work_stealing;
pub mod worker;

#[cfg(feature = "numa")]
pub mod numa;

pub use adaptive_scaling::{AdaptiveScalingManager, ScalingConfig, ScalingError, ScalingMetrics};
pub use backpressure::{AdaptiveBackpressure, BackpressureError, BackpressureLevel};
pub use batch_processor::{BatchError, BatchProcessor, BatchStats};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerError, CircuitState};
pub use enhanced_load_balancer::{
    EnhancedLoadBalancer, LoadBalancerConfig, LoadBalancerError, LoadBalancingStrategy,
};
pub use executor::*;
pub use load_testing::{LoadPattern, LoadTestError, LoadTestFramework};
pub use numa_scheduler::{
    MemoryPattern, NumaScheduler as TallyNumaScheduler, NumaSchedulerError, NumaTask,
};
pub use predictive_metrics::{
    PredictiveConfig, PredictiveError, PredictiveMetrics, PredictiveMetricsSnapshot, SystemMetrics,
};
pub use scheduler::*;
pub use work_stealing::{WorkStealingError, WorkStealingScheduler};
pub use worker::*;

#[cfg(feature = "numa")]
pub use numa::{NumaError, NumaScheduler, NumaTopology};

/// Engine-specific error types
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// Engine is not running
    #[error("Engine is not running")]
    NotRunning,

    /// Engine is already running
    #[error("Engine is already running")]
    AlreadyRunning,

    /// Worker pool is full
    #[error("Worker pool is full (capacity: {capacity})")]
    WorkerPoolFull {
        /// Pool capacity
        capacity: usize,
    },

    /// Strategy not found
    #[error("Strategy not found: {strategy_id}")]
    StrategyNotFound {
        /// Strategy identifier
        strategy_id: StrategyId,
    },

    /// Task execution failed
    #[error("Task execution failed: {task_id}, reason: {reason}")]
    TaskExecutionFailed {
        /// Task identifier
        task_id: TaskId,
        /// Failure reason
        reason: String,
    },

    /// Scheduler error
    #[error("Scheduler error: {reason}")]
    SchedulerError {
        /// Error reason
        reason: String,
    },

    /// Worker error
    #[error("Worker error: {worker_id}, reason: {reason}")]
    WorkerError {
        /// Worker identifier
        worker_id: WorkerId,
        /// Error reason
        reason: String,
    },

    /// Configuration error
    #[error("Configuration error: {field}")]
    ConfigError {
        /// Field name
        field: String,
    },

    /// Resource exhausted
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        /// Resource name
        resource: String,
    },

    /// Timeout error
    #[error("Operation timeout after {duration_ms}ms")]
    Timeout {
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// Task not found
    #[error("Task not found: {task_id}")]
    TaskNotFound {
        /// Task identifier
        task_id: TaskId,
    },

    /// Invalid state
    #[error("Invalid state in {component}: expected {expected}, got {actual}")]
    InvalidState {
        /// Component name
        component: String,
        /// Expected state
        expected: String,
        /// Actual state
        actual: String,
    },
}

/// Engine result type
pub type EngineResult<T> = Result<T, EngineError>;

/// Engine configuration
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Maximum number of worker threads
    pub max_workers: usize,

    /// Task queue capacity
    pub task_queue_capacity: usize,

    /// Worker idle timeout
    pub worker_idle_timeout: Duration,

    /// Task execution timeout
    pub task_timeout: Duration,

    /// Enable performance monitoring
    pub enable_monitoring: bool,

    /// CPU affinity for workers
    pub cpu_affinity: Option<Vec<usize>>,

    /// Memory pool size for zero-allocation paths
    pub memory_pool_size: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_workers: num_cpus::get().min(16),
            task_queue_capacity: 10_000,
            worker_idle_timeout: Duration::from_secs(30),
            task_timeout: Duration::from_millis(100),
            enable_monitoring: true,
            cpu_affinity: None,
            memory_pool_size: 1024 * 1024, // 1MB
        }
    }
}

impl EngineConfig {
    /// Validate configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn validate(&self) -> EngineResult<()> {
        if self.max_workers == 0 {
            return Err(EngineError::ConfigError {
                field: "max_workers must be > 0".to_string(),
            });
        }

        if self.task_queue_capacity == 0 {
            return Err(EngineError::ConfigError {
                field: "task_queue_capacity must be > 0".to_string(),
            });
        }

        if self.memory_pool_size == 0 {
            return Err(EngineError::ConfigError {
                field: "memory_pool_size must be > 0".to_string(),
            });
        }

        Ok(())
    }
}

/// Engine performance metrics
#[derive(Debug, Default)]
pub struct EngineMetrics {
    /// Total tasks processed
    pub tasks_processed: AtomicU64,

    /// Total tasks failed
    pub tasks_failed: AtomicU64,

    /// Total execution time in nanoseconds
    pub total_execution_time_ns: AtomicU64,

    /// Active workers count
    pub active_workers: AtomicU64,

    /// Queue depth
    pub queue_depth: AtomicU64,

    /// Engine start time (nanoseconds since epoch)
    pub start_time_ns: AtomicU64,
}

impl EngineMetrics {
    /// Get average task execution time in microseconds
    #[must_use]
    pub fn average_execution_time_us(&self) -> f64 {
        let total_tasks = self.tasks_processed.load(Ordering::Relaxed);
        if total_tasks == 0 {
            return 0.0_f64;
        }

        let total_time_ns = self.total_execution_time_ns.load(Ordering::Relaxed);
        f64::from(u32::try_from(total_time_ns / total_tasks).unwrap_or(u32::MAX)) / 1000.0_f64
    }

    /// Get tasks per second
    #[must_use]
    pub fn tasks_per_second(&self) -> f64 {
        let start_time_ns = self.start_time_ns.load(Ordering::Relaxed);
        if start_time_ns == 0 {
            return 0.0_f64;
        }

        let current_time_ns = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos(),
        )
        .unwrap_or(u64::MAX);

        let elapsed_ns = current_time_ns.saturating_sub(start_time_ns);
        #[allow(clippy::cast_precision_loss)]
        let elapsed_secs = elapsed_ns as f64 / 1_000_000_000.0_f64;

        if elapsed_secs > 0.0_f64 {
            let total_tasks = self.tasks_processed.load(Ordering::Relaxed);
            f64::from(u32::try_from(total_tasks).unwrap_or(u32::MAX)) / elapsed_secs
        } else {
            0.0_f64
        }
    }

    /// Get error rate (0.0 to 1.0)
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total_tasks = self.tasks_processed.load(Ordering::Relaxed);
        if total_tasks == 0 {
            return 0.0_f64;
        }

        let failed_tasks = self.tasks_failed.load(Ordering::Relaxed);
        f64::from(u32::try_from(failed_tasks).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(total_tasks).unwrap_or(u32::MAX))
    }
}

/// Aggregated metrics from all engine components
#[derive(Debug, Clone)]
pub struct AggregatedMetrics {
    /// Timestamp when metrics were collected
    pub timestamp: Instant,

    /// Total tasks processed across all components
    pub total_tasks_processed: u64,

    /// Total tasks failed across all components
    pub total_tasks_failed: u64,

    /// Average execution time in microseconds
    pub average_execution_time_us: f64,

    /// Tasks per second throughput
    pub tasks_per_second: f64,

    /// Overall error rate (0.0 to 1.0)
    pub error_rate: f64,

    /// Active workers count
    pub active_workers: u64,

    /// Queue depth
    pub queue_depth: u64,

    /// Memory usage statistics
    pub memory_usage_bytes: u64,

    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
}

impl AggregatedMetrics {
    /// Create new aggregated metrics
    #[must_use]
    pub fn new() -> Self {
        Self {
            timestamp: Instant::now(),
            total_tasks_processed: 0,
            total_tasks_failed: 0,
            average_execution_time_us: 0.0_f64,
            tasks_per_second: 0.0_f64,
            error_rate: 0.0_f64,
            active_workers: 0,
            queue_depth: 0,
            memory_usage_bytes: 0,
            cpu_utilization: 0.0_f64,
        }
    }

    /// Add engine metrics to aggregation
    pub fn add_engine_metrics(&mut self, metrics: &EngineMetrics) {
        self.total_tasks_processed += metrics.tasks_processed.load(Ordering::Relaxed);
        self.total_tasks_failed += metrics.tasks_failed.load(Ordering::Relaxed);
        self.active_workers = metrics.active_workers.load(Ordering::Relaxed);
        self.queue_depth = metrics.queue_depth.load(Ordering::Relaxed);

        // Calculate aggregated rates
        self.average_execution_time_us = metrics.average_execution_time_us();
        self.tasks_per_second = metrics.tasks_per_second();
        self.error_rate = metrics.error_rate();
    }

    /// Add worker pool metrics to aggregation
    pub fn add_worker_pool_metrics(&mut self, worker_pool: &WorkerPool) {
        let stats = worker_pool.stats();
        self.total_tasks_processed += stats.total_tasks_processed.load(Ordering::Relaxed);
        // Additional worker pool specific metrics can be added here
    }

    /// Add scheduler metrics to aggregation
    pub fn add_scheduler_metrics(&mut self, scheduler: &TaskScheduler) {
        let stats = scheduler.stats();
        let tasks_scheduled = stats.tasks_scheduled.load(Ordering::Relaxed);
        let tasks_dispatched = stats.tasks_dispatched.load(Ordering::Relaxed);

        // Update queue efficiency metrics
        if tasks_scheduled > 0 {
            let dispatch_rate = f64::from(u32::try_from(tasks_dispatched).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(tasks_scheduled).unwrap_or(u32::MAX));
            // Could add dispatch_rate to metrics if needed
            let _ = dispatch_rate; // Suppress unused warning for now
        }
    }

    /// Add executor metrics to aggregation
    pub fn add_executor_metrics(&mut self, executor: &TaskExecutor) {
        let stats = executor.stats();
        let executed = stats.tasks_executed.load(Ordering::Relaxed);
        let successful = stats.successful_executions.load(Ordering::Relaxed);

        // Update execution success metrics
        if executed > 0 {
            let success_rate = f64::from(u32::try_from(successful).unwrap_or(u32::MAX))
                / f64::from(u32::try_from(executed).unwrap_or(u32::MAX));
            // Could add success_rate to metrics if needed
            let _ = success_rate; // Suppress unused warning for now
        }
    }
}

impl Default for AggregatedMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Engine health status for monitoring and diagnostics
#[derive(Debug, Clone)]
pub struct EngineHealthStatus {
    /// Whether engine is currently running
    pub is_running: bool,

    /// Current number of worker threads
    pub worker_count: usize,

    /// Number of registered strategies
    pub strategy_count: usize,

    /// Current throughput in tasks per second
    pub tasks_per_second: f64,

    /// Current error rate (0.0 to 1.0)
    pub error_rate: f64,

    /// Average task execution latency in microseconds
    pub average_latency_us: f64,

    /// Current queue depth
    pub queue_depth: u64,

    /// Memory usage in megabytes
    pub memory_usage_mb: u64,

    /// Engine uptime in seconds
    pub uptime_seconds: u64,
}

impl EngineHealthStatus {
    /// Check if engine is healthy based on performance thresholds
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.is_running
            && self.error_rate < 0.01_f64  // Less than 1% error rate
            && self.average_latency_us < 1000.0_f64  // Less than 1ms average latency
            && self.worker_count > 0
    }

    /// Get health score (0.0 to 1.0, where 1.0 is perfect health)
    #[must_use]
    pub fn health_score(&self) -> f64 {
        if !self.is_running {
            return 0.0_f64;
        }

        let mut score = 1.0_f64;

        // Penalize high error rates
        score *= 1.0_f64 - self.error_rate.min(1.0_f64);

        // Penalize high latency (target: <1ms)
        if self.average_latency_us > 1000.0_f64 {
            score *= 1000.0_f64 / self.average_latency_us.max(1000.0_f64);
        }

        // Penalize if no workers
        if self.worker_count == 0 {
            score *= 0.5_f64;
        }

        score.clamp(0.0_f64, 1.0_f64)
    }
}

/// Main `TallyIO` Engine with Advanced Performance Optimizations
#[repr(C, align(64))]
pub struct Engine {
    /// Engine configuration
    config: Arc<EngineConfig>,

    /// Engine running state (cache-aligned for hot path access)
    is_running: AtomicBool,

    /// Task executor
    executor: Option<TaskExecutor>,

    /// Task scheduler
    scheduler: Option<TaskScheduler>,

    /// Worker pool
    worker_pool: Option<Arc<WorkerPool>>,

    /// Performance metrics
    metrics: Arc<EngineMetrics>,

    /// Active strategies
    strategies: Arc<DashMap<StrategyId, Arc<dyn Strategy + Send + Sync>>>,

    /// Advanced adaptive scaling manager
    scaling_manager: Option<AdaptiveScalingManager>,

    /// Enhanced load balancer
    load_balancer: Option<EnhancedLoadBalancer>,

    /// Predictive metrics system
    predictive_metrics: Option<PredictiveMetrics>,

    /// Advanced circuit breaker
    circuit_breaker: Option<CircuitBreaker>,
}

/// Strategy trait for MEV and liquidation strategies
pub trait Strategy: Send + Sync {
    /// Strategy identifier
    fn id(&self) -> StrategyId;

    /// Strategy name
    fn name(&self) -> &str;

    /// Execute strategy
    ///
    /// # Errors
    ///
    /// Returns error if strategy execution fails
    fn execute(&self, context: &ExecutionContext) -> EngineResult<ExecutionResult>;

    /// Validate strategy configuration
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    fn validate(&self) -> EngineResult<()>;

    /// Get strategy priority (higher = more important)
    fn priority(&self) -> u8;

    /// Check if strategy can execute in current context
    fn can_execute(&self, context: &ExecutionContext) -> bool;
}

/// Execution context for strategies
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Task identifier
    pub task_id: TaskId,

    /// Worker identifier
    pub worker_id: WorkerId,

    /// Execution timestamp
    pub timestamp: Instant,

    /// Available gas limit
    pub gas_limit: u64,

    /// Maximum execution time
    pub max_execution_time: Duration,

    /// Strategy-specific data
    pub data: Vec<u8>,
}

/// Strategy execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Success flag
    pub success: bool,

    /// Gas used
    pub gas_used: u64,

    /// Execution time
    pub execution_time: Duration,

    /// Result data
    pub data: Vec<u8>,

    /// Error message if failed
    pub error: Option<String>,
}

impl Engine {
    /// Create new engine instance
    ///
    /// # Errors
    ///
    /// Returns error if configuration validation fails
    pub fn new(config: EngineConfig) -> EngineResult<Self> {
        config.validate()?;

        Ok(Self {
            config: Arc::new(config),
            is_running: AtomicBool::new(false),
            executor: None,
            scheduler: None,
            worker_pool: None,
            metrics: Arc::new(EngineMetrics::default()),
            strategies: Arc::new(DashMap::new()),
            scaling_manager: None,
            load_balancer: None,
            predictive_metrics: None,
            circuit_breaker: None,
        })
    }

    /// Start the engine
    ///
    /// # Errors
    ///
    /// Returns error if engine is already running or initialization fails
    pub fn start(&mut self) -> EngineResult<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Err(EngineError::AlreadyRunning);
        }

        // Initialize components
        let mut worker_pool = WorkerPool::new(self.config.clone(), self.metrics.clone())?;
        let mut scheduler = TaskScheduler::new(self.config.clone(), self.metrics.clone())?;
        let executor = TaskExecutor::new(
            self.config.clone(),
            self.metrics.clone(),
            self.strategies.clone(),
        )?;

        // Start worker pool first with strategies
        worker_pool.start_with_strategies(self.strategies.clone())?;

        // Create dispatcher to connect scheduler with worker pool
        let worker_pool_arc = Arc::new(worker_pool);
        let dispatcher = Arc::new(WorkerPoolDispatcher::new(worker_pool_arc.clone()));

        // Set dispatcher in scheduler before starting
        scheduler.set_dispatcher(dispatcher)?;

        // Start scheduler
        scheduler.start()?;

        // Initialize advanced components with safe worker configuration
        let min_workers = 1_usize;
        let max_workers = self.config.max_workers.max(min_workers + 1); // Ensure max > min
        let scaling_config = ScalingConfig {
            min_workers,
            max_workers,
            target_latency_ns: 1_000_000, // 1ms
            scaling_threshold: 0.8_f64,
            cooldown_duration: Duration::from_millis(100),
            prediction_window: Duration::from_millis(50),
        };
        let scaling_manager =
            AdaptiveScalingManager::new(scaling_config).map_err(|e| EngineError::ConfigError {
                field: format!("Scaling manager initialization failed: {e}"),
            })?;

        let load_balancer_config = LoadBalancerConfig {
            strategy: LoadBalancingStrategy::Predictive,
            max_workers,
            target_latency_ns: 1_000_000, // 1ms
            measurement_window: Duration::from_millis(10),
            cache_optimized: true,
            numa_aware: true,
        };
        let load_balancer = EnhancedLoadBalancer::new(load_balancer_config).map_err(|e| {
            EngineError::ConfigError {
                field: format!("Load balancer initialization failed: {e}"),
            }
        })?;

        let predictive_config = PredictiveConfig {
            max_prediction_latency_ns: 10_000, // 10μs
            update_frequency: Duration::from_millis(100),
            min_training_samples: 100,
            learning_rate: 0.01_f32,
            smoothing_factor: 0.9_f32,
            confidence_threshold: 0.7_f32,
        };
        let predictive_metrics = PredictiveMetrics::new(predictive_config);

        let circuit_breaker_config = circuit_breaker::CircuitBreakerConfig::default();
        let circuit_breaker = CircuitBreaker::new(circuit_breaker_config);

        self.worker_pool = Some(worker_pool_arc);
        self.scheduler = Some(scheduler);
        self.executor = Some(executor);
        self.scaling_manager = Some(scaling_manager);
        self.load_balancer = Some(load_balancer);
        self.predictive_metrics = Some(predictive_metrics);
        self.circuit_breaker = Some(circuit_breaker);

        // Mark as running and record start time
        self.is_running.store(true, Ordering::Release);
        let start_time_ns = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos(),
        )
        .unwrap_or(u64::MAX);
        self.metrics
            .start_time_ns
            .store(start_time_ns, Ordering::Relaxed);

        Ok(())
    }

    /// Stop the engine
    ///
    /// # Errors
    ///
    /// Returns error if engine is not running or shutdown fails
    pub fn stop(&mut self) -> EngineResult<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(EngineError::NotRunning);
        }

        // Stop components in reverse order
        if let Some(ref mut scheduler) = self.scheduler {
            scheduler.stop()?;
        }

        // Signal worker pool to shutdown gracefully
        if let Some(ref pool_arc) = self.worker_pool {
            pool_arc.signal_shutdown();
        }

        // Clear components
        self.executor = None;
        self.scheduler = None;
        self.worker_pool = None;

        // Mark as stopped
        self.is_running.store(false, Ordering::Release);

        Ok(())
    }

    /// Check if engine is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Register a strategy
    ///
    /// # Errors
    ///
    /// Returns error if strategy validation fails
    pub fn register_strategy(&self, strategy: Arc<dyn Strategy + Send + Sync>) -> EngineResult<()> {
        strategy.validate()?;

        let strategy_id = strategy.id();
        self.strategies.insert(strategy_id, strategy);

        // If engine is running, update worker pool strategies
        if self.is_running() {
            self.update_worker_strategies()?;
        }

        Ok(())
    }

    /// Unregister a strategy
    ///
    /// # Errors
    ///
    /// Returns error if strategy is not found
    pub fn unregister_strategy(&self, strategy_id: StrategyId) -> EngineResult<()> {
        self.strategies
            .remove(&strategy_id)
            .ok_or(EngineError::StrategyNotFound { strategy_id })?;

        // If engine is running, update worker pool strategies
        if self.is_running() {
            self.update_worker_strategies()?;
        }

        Ok(())
    }

    /// Update strategies in all running workers
    ///
    /// # Errors
    ///
    /// Returns error if strategy update fails
    pub fn update_strategies(
        &self,
        new_strategies: &Arc<DashMap<StrategyId, Arc<dyn Strategy + Send + Sync>>>,
    ) -> EngineResult<()> {
        // Validate all new strategies
        for strategy in new_strategies.iter() {
            strategy.value().validate()?;
        }

        // Replace strategies atomically
        self.strategies.clear();
        for entry in new_strategies.iter() {
            self.strategies
                .insert(*entry.key(), Arc::clone(entry.value()));
        }

        // Update worker pool if running
        if self.is_running() {
            self.update_worker_strategies()?;
        }

        Ok(())
    }

    /// Internal method to update worker pool strategies
    fn update_worker_strategies(&self) -> EngineResult<()> {
        if let Some(ref worker_pool) = self.worker_pool {
            worker_pool.update_strategies(self.strategies.clone())?;
        }
        Ok(())
    }

    /// Get engine configuration
    #[must_use]
    pub const fn config(&self) -> &Arc<EngineConfig> {
        &self.config
    }

    /// Get engine metrics
    #[must_use]
    pub const fn metrics(&self) -> &Arc<EngineMetrics> {
        &self.metrics
    }

    /// Get aggregated metrics from all components
    #[must_use]
    pub fn get_aggregated_metrics(&self) -> AggregatedMetrics {
        let mut aggregated = AggregatedMetrics::new();

        // Engine metrics
        aggregated.add_engine_metrics(&self.metrics);

        // Worker pool metrics
        if let Some(ref worker_pool) = self.worker_pool {
            aggregated.add_worker_pool_metrics(worker_pool);
        }

        // Scheduler metrics
        if let Some(ref scheduler) = self.scheduler {
            aggregated.add_scheduler_metrics(scheduler);
        }

        // Executor metrics
        if let Some(ref executor) = self.executor {
            aggregated.add_executor_metrics(executor);
        }

        aggregated
    }

    /// Get number of registered strategies
    #[must_use]
    pub fn strategy_count(&self) -> usize {
        self.strategies.len()
    }

    /// Submit a task for execution - Ultra-optimized hot path
    ///
    /// # Errors
    ///
    /// Returns error if engine is not running or task submission fails
    pub fn submit_task(&self, task: &Task) -> EngineResult<TaskId> {
        // Ultra-fast path: Single atomic check with relaxed ordering
        if !self.is_running.load(Ordering::Relaxed) {
            return Err(EngineError::NotRunning);
        }

        // Direct scheduler access - eliminate Option overhead
        // SAFETY: scheduler is guaranteed to be Some when is_running is true
        let scheduler = unsafe { self.scheduler.as_ref().unwrap_unchecked() };

        // Direct submission with optimized path
        scheduler.submit_task(task)
    }

    /// Submit a task for execution (fallback safe version)
    ///
    /// # Errors
    ///
    /// Returns error if engine is not running or task submission fails
    pub fn submit_task_safe(&self, task: &Task) -> EngineResult<TaskId> {
        if !self.is_running() {
            return Err(EngineError::NotRunning);
        }

        let scheduler = self
            .scheduler
            .as_ref()
            .ok_or_else(|| EngineError::SchedulerError {
                reason: "Scheduler not initialized".to_string(),
            })?;

        scheduler.submit_task(task)
    }

    /// Scale worker pool to new size
    ///
    /// # Errors
    ///
    /// Returns error if scaling fails or engine is not running
    pub fn scale_workers(&self, new_worker_count: usize) -> EngineResult<()> {
        if !self.is_running() {
            return Err(EngineError::NotRunning);
        }

        if new_worker_count == 0 {
            return Err(EngineError::ConfigError {
                field: "Worker count must be > 0".to_string(),
            });
        }

        // For now, log the scaling request
        // In a full implementation, this would:
        // 1. Create new workers if scaling up
        // 2. Gracefully shutdown excess workers if scaling down
        // 3. Update worker pool configuration

        tracing::info!(
            "Worker scaling requested: current={}, target={}",
            self.worker_count(),
            new_worker_count
        );

        // This is a placeholder implementation
        // Full implementation would require significant worker pool refactoring
        Ok(())
    }

    /// Get current worker count
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.worker_pool
            .as_ref()
            .map_or(0, |pool| pool.worker_count())
    }

    /// Get engine health status
    #[must_use]
    pub fn health_status(&self) -> EngineHealthStatus {
        let metrics = self.get_aggregated_metrics();

        EngineHealthStatus {
            is_running: self.is_running(),
            worker_count: self.worker_count(),
            strategy_count: self.strategy_count(),
            tasks_per_second: metrics.tasks_per_second,
            error_rate: metrics.error_rate,
            average_latency_us: metrics.average_execution_time_us,
            queue_depth: metrics.queue_depth,
            memory_usage_mb: metrics.memory_usage_bytes / 1_024_u64 / 1_024_u64,
            uptime_seconds: self.uptime_seconds(),
        }
    }

    /// Get engine uptime in seconds
    #[must_use]
    pub fn uptime_seconds(&self) -> u64 {
        let start_time_ns = self.metrics.start_time_ns.load(Ordering::Relaxed);
        if start_time_ns == 0 {
            return 0;
        }

        let current_time_ns = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos(),
        )
        .unwrap_or(u64::MAX);

        current_time_ns.saturating_sub(start_time_ns) / 1_000_000_000_u64
    }

    /// Get adaptive scaling manager
    #[must_use]
    pub const fn scaling_manager(&self) -> &Option<AdaptiveScalingManager> {
        &self.scaling_manager
    }

    /// Get enhanced load balancer
    #[must_use]
    pub const fn load_balancer(&self) -> &Option<EnhancedLoadBalancer> {
        &self.load_balancer
    }

    /// Get predictive metrics system
    #[must_use]
    pub const fn predictive_metrics(&self) -> &Option<PredictiveMetrics> {
        &self.predictive_metrics
    }

    /// Get circuit breaker
    #[must_use]
    pub const fn circuit_breaker(&self) -> &Option<CircuitBreaker> {
        &self.circuit_breaker
    }

    /// Record load measurement for adaptive scaling
    pub fn record_load_measurement(&self, load_factor: f64, latency_ns: u64) {
        if let Some(ref scaling_manager) = self.scaling_manager {
            scaling_manager.record_load(load_factor, latency_ns);
        }
    }

    /// Update worker load in load balancer
    pub fn update_worker_load(
        &self,
        worker_id: WorkerId,
        load: f64,
        task_count: u32,
        avg_latency_ns: u64,
    ) {
        if let Some(ref load_balancer) = self.load_balancer {
            load_balancer.update_worker_load(worker_id, load, task_count, avg_latency_ns);
        }
    }

    /// Predict task latency using ML model
    ///
    /// # Errors
    ///
    /// Returns error if prediction fails
    pub fn predict_task_latency(
        &self,
        task: &Task,
        system_metrics: &SystemMetrics,
    ) -> Result<Option<u64>, PredictiveError> {
        self.predictive_metrics
            .as_ref()
            .map_or(Ok(None), |predictive_metrics| {
                match predictive_metrics.predict_latency(task, system_metrics) {
                    Ok(result) => Ok(Some(result.predicted_latency_ns)),
                    Err(e) => Err(e),
                }
            })
    }

    /// Check if circuit breaker allows requests (ultra-fast)
    #[must_use]
    pub fn is_circuit_breaker_open(&self) -> bool {
        self.circuit_breaker
            .as_ref()
            .is_some_and(|cb| !cb.is_request_allowed_fast())
    }

    /// Get comprehensive system metrics for ML prediction
    #[must_use]
    pub fn get_system_metrics(&self) -> SystemMetrics {
        let aggregated = self.get_aggregated_metrics();

        SystemMetrics {
            cpu_load: {
                let clamped = aggregated.cpu_utilization.clamp(0.0_f64, 1.0_f64);
                if clamped.is_finite() && (0.0_f64..=1.0_f64).contains(&clamped) {
                    // Safe conversion: f64 in [0.0, 1.0] always fits in f32
                    #[allow(clippy::cast_possible_truncation)]
                    { clamped as f32 }
                } else {
                    0.0_f32
                }
            },
            queue_depth: u32::try_from(aggregated.queue_depth).unwrap_or(u32::MAX),
            recent_avg_latency_ns: {
                let clamped_time = aggregated.average_execution_time_us.clamp(0.0_f64, f64::from(u32::MAX));
                let safe_u32 = if clamped_time.is_finite() && (0.0_f64..=f64::from(u32::MAX)).contains(&clamped_time) {
                    // Safe conversion: f64 in [0.0, u32::MAX] range
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    { clamped_time as u32 }
                } else {
                    0_u32
                };
                u64::from(safe_u32).saturating_mul(1000)
            },
            memory_pressure: {
                let clamped_bytes = aggregated.memory_usage_bytes.min(u64::from(u32::MAX));
                let gb_usage = f64::from(
                    u32::try_from(clamped_bytes).unwrap_or(u32::MAX)
                ) / (1024.0_f64 * 1024.0_f64 * 1024.0_f64);
                let clamped_usage = gb_usage.clamp(0.0_f64, 1.0_f64);
                if clamped_usage.is_finite() && (0.0_f64..=1.0_f64).contains(&clamped_usage) {
                    // Safe conversion: f64 in [0.0, 1.0] always fits in f32
                    #[allow(clippy::cast_possible_truncation)]
                    { clamped_usage as f32 }
                } else {
                    0.0_f32
                }
            },
            cpu_utilization: {
                let clamped = aggregated.cpu_utilization.clamp(0.0_f64, 1.0_f64);
                if clamped.is_finite() && (0.0_f64..=1.0_f64).contains(&clamped) {
                    // Safe conversion: f64 in [0.0, 1.0] always fits in f32
                    #[allow(clippy::cast_possible_truncation)]
                    { clamped as f32 }
                } else {
                    0.0_f32
                }
            },
            network_latency_ns: 1000, // Placeholder - would be measured in real implementation
        }
    }
}
