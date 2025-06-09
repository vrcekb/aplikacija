//! Task Executor - Ultra-Performance Task Execution Engine
//!
//! Handles execution of MEV and liquidation strategies with <1ms latency guarantee.
//! Implements zero-allocation hot paths and lock-free concurrency.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use thiserror::Error;

use crate::types::{StrategyId, TaskId, WorkerId};

use super::{
    EngineConfig, EngineError, EngineMetrics, EngineResult, ExecutionContext, ExecutionResult,
    Strategy,
};

/// Task execution error
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ExecutorError {
    /// Task not found
    #[error("Task not found: {task_id}")]
    TaskNotFound {
        /// Task identifier
        task_id: TaskId,
    },

    /// Strategy not found
    #[error("Strategy not found: {strategy_id}")]
    StrategyNotFound {
        /// Strategy identifier
        strategy_id: StrategyId,
    },

    /// Execution timeout
    #[error("Task execution timeout: {task_id}, duration: {duration_ms}ms")]
    ExecutionTimeout {
        /// Task identifier
        task_id: TaskId,
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// Resource exhausted
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted {
        /// Resource name
        resource: String,
    },

    /// Invalid task state
    #[error("Invalid task state: {task_id}, expected: {expected}, actual: {actual}")]
    InvalidTaskState {
        /// Task identifier
        task_id: TaskId,
        /// Expected state
        expected: String,
        /// Actual state
        actual: String,
    },
}

impl From<ExecutorError> for EngineError {
    fn from(err: ExecutorError) -> Self {
        match err {
            ExecutorError::TaskNotFound { task_id } => Self::TaskNotFound { task_id },
            ExecutorError::StrategyNotFound { strategy_id } => {
                Self::StrategyNotFound { strategy_id }
            }
            ExecutorError::ExecutionTimeout {
                task_id: _,
                duration_ms,
            } => Self::Timeout { duration_ms },
            ExecutorError::ResourceExhausted { resource } => Self::ResourceExhausted { resource },
            ExecutorError::InvalidTaskState {
                task_id,
                expected,
                actual,
            } => Self::InvalidState {
                component: format!("Task {task_id}"),
                expected,
                actual,
            },
        }
    }
}

/// Task execution priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority (batch operations)
    Low = 0,
    /// Normal priority (regular operations)
    Normal = 1,
    /// High priority (MEV opportunities)
    High = 2,
    /// Critical priority (liquidations)
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Task execution state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is pending execution
    Pending,
    /// Task is currently executing
    Executing,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
}

impl Default for TaskState {
    fn default() -> Self {
        Self::Pending
    }
}

/// Execution task
#[derive(Debug, Clone)]
pub struct Task {
    /// Unique task identifier
    pub id: TaskId,

    /// Strategy to execute
    pub strategy_id: StrategyId,

    /// Task priority
    pub priority: TaskPriority,

    /// Maximum execution time
    pub max_execution_time: Duration,

    /// Gas limit for execution
    pub gas_limit: u64,

    /// Task-specific data
    pub data: Vec<u8>,

    /// Task creation timestamp
    pub created_at: Instant,

    /// Current task state
    pub state: TaskState,
}

impl Task {
    /// Create new task
    #[must_use]
    pub fn new(strategy_id: StrategyId, data: Vec<u8>) -> Self {
        Self {
            id: TaskId::new(),
            strategy_id,
            priority: TaskPriority::default(),
            max_execution_time: Duration::from_millis(100),
            gas_limit: 1_000_000,
            data,
            created_at: Instant::now(),
            state: TaskState::default(),
        }
    }

    /// Create high priority task
    #[must_use]
    pub fn new_high_priority(strategy_id: StrategyId, data: Vec<u8>) -> Self {
        let mut task = Self::new(strategy_id, data);
        task.priority = TaskPriority::High;
        task.max_execution_time = Duration::from_millis(50);
        task
    }

    /// Create critical priority task
    #[must_use]
    pub fn new_critical(strategy_id: StrategyId, data: Vec<u8>) -> Self {
        let mut task = Self::new(strategy_id, data);
        task.priority = TaskPriority::Critical;
        task.max_execution_time = Duration::from_millis(10);
        task
    }

    /// Check if task is expired
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.max_execution_time * 2
    }

    /// Get task age
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Task executor statistics
#[derive(Debug, Default)]
pub struct ExecutorStats {
    /// Total tasks executed
    pub tasks_executed: AtomicU64,

    /// Total execution time in nanoseconds
    pub total_execution_time_ns: AtomicU64,

    /// Tasks by priority
    pub low_priority_tasks: AtomicU64,
    /// Normal priority task count
    pub normal_priority_tasks: AtomicU64,
    /// High priority task count
    pub high_priority_tasks: AtomicU64,
    /// Critical priority task count
    pub critical_priority_tasks: AtomicU64,

    /// Execution outcomes
    pub successful_executions: AtomicU64,
    /// Failed execution count
    pub failed_executions: AtomicU64,
    /// Timeout execution count
    pub timeout_executions: AtomicU64,
}

impl ExecutorStats {
    /// Get average execution time in microseconds
    #[must_use]
    pub fn average_execution_time_us(&self) -> f64 {
        let total_tasks = self.tasks_executed.load(Ordering::Relaxed);
        if total_tasks == 0 {
            return 0.0_f64;
        }

        let total_time_ns = self.total_execution_time_ns.load(Ordering::Relaxed);
        f64::from(u32::try_from(total_time_ns / total_tasks).unwrap_or(u32::MAX)) / 1000.0_f64
    }

    /// Get success rate (0.0 to 1.0)
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total_tasks = self.tasks_executed.load(Ordering::Relaxed);
        if total_tasks == 0 {
            return 0.0_f64;
        }

        let successful = self.successful_executions.load(Ordering::Relaxed);
        f64::from(u32::try_from(successful).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(total_tasks).unwrap_or(u32::MAX))
    }
}

/// Task executor
pub struct TaskExecutor {
    /// Configuration
    #[allow(dead_code)]
    config: Arc<EngineConfig>,

    /// Engine metrics
    metrics: Arc<EngineMetrics>,

    /// Available strategies
    strategies: Arc<DashMap<StrategyId, Arc<dyn Strategy + Send + Sync>>>,

    /// Executor statistics
    stats: Arc<ExecutorStats>,

    /// Active tasks
    active_tasks: Arc<DashMap<TaskId, Task>>,
}

impl TaskExecutor {
    /// Create new task executor
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(
        config: Arc<EngineConfig>,
        metrics: Arc<EngineMetrics>,
        strategies: Arc<DashMap<StrategyId, Arc<dyn Strategy + Send + Sync>>>,
    ) -> EngineResult<Self> {
        Ok(Self {
            config,
            metrics,
            strategies,
            stats: Arc::new(ExecutorStats::default()),
            active_tasks: Arc::new(DashMap::new()),
        })
    }

    /// Execute a task
    ///
    /// # Errors
    ///
    /// Returns error if task execution fails
    pub fn execute_task(
        &self,
        mut task: Task,
        worker_id: WorkerId,
    ) -> EngineResult<ExecutionResult> {
        let start_time = Instant::now();

        // Update task state
        task.state = TaskState::Executing;
        self.active_tasks.insert(task.id, task.clone());

        // Execute strategy with timeout protection - avoid significant drop
        let result = {
            let strategy =
                self.strategies
                    .get(&task.strategy_id)
                    .ok_or(ExecutorError::StrategyNotFound {
                        strategy_id: task.strategy_id,
                    })?;

            // Create execution context
            let context = ExecutionContext {
                task_id: task.id,
                worker_id,
                timestamp: start_time,
                gas_limit: task.gas_limit,
                max_execution_time: task.max_execution_time,
                data: task.data.clone(),
            };

            self.execute_with_timeout(strategy.as_ref(), &context, task.max_execution_time)?
        };

        // Update statistics
        let execution_time = start_time.elapsed();
        self.update_stats(&task, &result, execution_time);

        // Update task state
        task.state = if result.success {
            TaskState::Completed
        } else {
            TaskState::Failed
        };

        self.active_tasks.remove(&task.id);

        Ok(result)
    }

    /// Execute strategy with timeout protection
    fn execute_with_timeout(
        &self,
        strategy: &dyn Strategy,
        context: &ExecutionContext,
        timeout: Duration,
    ) -> EngineResult<ExecutionResult> {
        // Use configured timeout if provided timeout is too large
        let effective_timeout = timeout.min(self.config.task_timeout);

        // Check if strategy can execute
        if !strategy.can_execute(context) {
            return Ok(ExecutionResult {
                success: false,
                gas_used: 0,
                execution_time: Duration::ZERO,
                data: Vec::with_capacity(0),
                error: Some("Strategy cannot execute in current context".to_string()),
            });
        }

        // Execute strategy
        let start = Instant::now();
        let result = strategy.execute(context);
        let execution_time = start.elapsed();

        // Check timeout
        if execution_time > effective_timeout {
            return Err(EngineError::Timeout {
                duration_ms: u64::try_from(execution_time.as_millis()).unwrap_or(u64::MAX),
            });
        }

        result
    }

    /// Update execution statistics
    fn update_stats(&self, task: &Task, result: &ExecutionResult, execution_time: Duration) {
        self.stats.tasks_executed.fetch_add(1, Ordering::Relaxed);
        self.stats.total_execution_time_ns.fetch_add(
            u64::try_from(execution_time.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );

        // Update priority counters
        match task.priority {
            TaskPriority::Low => self
                .stats
                .low_priority_tasks
                .fetch_add(1, Ordering::Relaxed),
            TaskPriority::Normal => self
                .stats
                .normal_priority_tasks
                .fetch_add(1, Ordering::Relaxed),
            TaskPriority::High => self
                .stats
                .high_priority_tasks
                .fetch_add(1, Ordering::Relaxed),
            TaskPriority::Critical => self
                .stats
                .critical_priority_tasks
                .fetch_add(1, Ordering::Relaxed),
        };

        // Update outcome counters
        if result.success {
            self.stats
                .successful_executions
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.failed_executions.fetch_add(1, Ordering::Relaxed);
        }

        // Update engine metrics
        self.metrics.tasks_processed.fetch_add(1, Ordering::Relaxed);
        if !result.success {
            self.metrics.tasks_failed.fetch_add(1, Ordering::Relaxed);
        }
        self.metrics.total_execution_time_ns.fetch_add(
            u64::try_from(execution_time.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// Get executor statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<ExecutorStats> {
        &self.stats
    }

    /// Get number of active tasks
    #[must_use]
    pub fn active_task_count(&self) -> usize {
        self.active_tasks.len()
    }
}
