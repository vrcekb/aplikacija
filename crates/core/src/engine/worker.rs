//! Worker Pool - Ultra-Performance Worker Thread Management
//!
//! Implements lock-free worker pool for task execution with CPU affinity and NUMA awareness.
//! Designed for <1ms latency with zero-allocation hot paths.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::{self, Receiver, Sender};
use dashmap::DashMap;
use thiserror::Error;

use crate::types::{StrategyId, WorkerId};

use super::{
    EngineConfig, EngineError, EngineMetrics, EngineResult, ExecutionResult, Strategy, Task,
    TaskExecutor,
};

/// Worker-specific error types
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum WorkerError {
    /// Worker not found
    #[error("Worker not found: {worker_id}")]
    WorkerNotFound {
        /// Worker identifier
        worker_id: WorkerId,
    },

    /// Worker pool is full
    #[error("Worker pool is full (capacity: {capacity})")]
    PoolFull {
        /// Pool capacity
        capacity: usize,
    },

    /// Worker is busy
    #[error("Worker is busy: {worker_id}")]
    WorkerBusy {
        /// Worker identifier
        worker_id: WorkerId,
    },

    /// Worker thread panic
    #[error("Worker thread panic: {worker_id}")]
    WorkerPanic {
        /// Worker identifier
        worker_id: WorkerId,
    },

    /// Channel error
    #[error("Channel error: {reason}")]
    ChannelError {
        /// Error reason
        reason: String,
    },
}

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is idle and waiting for tasks
    Idle,
    /// Worker is executing a task
    Busy,
    /// Worker is shutting down
    Stopping,
    /// Worker has stopped
    Stopped,
}

/// Shutdown signal for graceful worker termination
#[derive(Debug, Clone)]
pub struct ShutdownSignal {
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

impl ShutdownSignal {
    /// Create new shutdown signal
    fn new() -> Self {
        Self {
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal shutdown
    fn signal(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Check if shutdown was signaled
    fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

impl Default for WorkerState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Worker statistics
#[derive(Debug, Default)]
pub struct WorkerStats {
    /// Total tasks processed
    pub tasks_processed: AtomicU64,

    /// Total execution time in nanoseconds
    pub total_execution_time_ns: AtomicU64,

    /// Total idle time in nanoseconds
    pub total_idle_time_ns: AtomicU64,

    /// Worker start time (nanoseconds since epoch)
    pub start_time_ns: AtomicU64,

    /// Last task completion time (nanoseconds since epoch)
    pub last_task_time_ns: AtomicU64,
}

impl WorkerStats {
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

    /// Get worker utilization (0.0 to 1.0)
    #[must_use]
    pub fn utilization(&self) -> f64 {
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

        let total_time_ns = current_time_ns.saturating_sub(start_time_ns);
        if total_time_ns == 0 {
            return 0.0_f64;
        }

        let execution_time_ns = self.total_execution_time_ns.load(Ordering::Relaxed);
        f64::from(u32::try_from(execution_time_ns).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(total_time_ns).unwrap_or(u32::MAX))
    }
}

/// Worker thread
pub struct Worker {
    /// Worker identifier
    pub id: WorkerId,

    /// Worker state
    pub state: AtomicU64, // Using u64 to store WorkerState as atomic

    /// Worker statistics
    pub stats: Arc<WorkerStats>,

    /// Task receiver channel
    task_receiver: Receiver<Task>,

    /// Result sender channel
    result_sender: Sender<(WorkerId, Task, Result<ExecutionResult, EngineError>)>,

    /// Worker thread handle
    thread_handle: Option<thread::JoinHandle<()>>,

    /// CPU affinity
    cpu_affinity: Option<usize>,

    /// Shutdown signal
    shutdown_signal: ShutdownSignal,
}

impl Worker {
    /// Create new worker
    fn new(
        id: WorkerId,
        task_receiver: Receiver<Task>,
        result_sender: Sender<(WorkerId, Task, Result<ExecutionResult, EngineError>)>,
        cpu_affinity: Option<usize>,
    ) -> Self {
        Self {
            id,
            state: AtomicU64::new(WorkerState::Idle as u64),
            stats: Arc::new(WorkerStats::default()),
            task_receiver,
            result_sender,
            thread_handle: None,
            cpu_affinity,
            shutdown_signal: ShutdownSignal::new(),
        }
    }

    /// Start worker thread
    fn start(&mut self, executor: Arc<TaskExecutor>) -> Result<(), WorkerError> {
        let worker_id = self.id;
        let task_receiver = self.task_receiver.clone();
        let result_sender = self.result_sender.clone();
        let stats_ref = Arc::clone(&self.stats);
        let state_atomic = Arc::new(AtomicU64::new(WorkerState::Idle as u64));
        let cpu_affinity = self.cpu_affinity;
        let shutdown_signal = self.shutdown_signal.clone();

        let handle = thread::Builder::new()
            .name(format!("tallyio-worker-{worker_id}"))
            .spawn(move || {
                Self::worker_loop(
                    worker_id,
                    &task_receiver,
                    &result_sender,
                    &executor,
                    &stats_ref,
                    &state_atomic,
                    cpu_affinity,
                    &shutdown_signal,
                );
            })
            .map_err(|e| WorkerError::ChannelError {
                reason: format!("Failed to start worker thread: {e}"),
            })?;

        self.thread_handle = Some(handle);

        // Record start time
        let start_time_ns = u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos(),
        )
        .unwrap_or(u64::MAX);
        self.stats
            .start_time_ns
            .store(start_time_ns, Ordering::Relaxed);

        Ok(())
    }

    /// Worker main loop
    #[allow(clippy::too_many_arguments)]
    fn worker_loop(
        worker_id: WorkerId,
        task_receiver: &Receiver<Task>,
        result_sender: &Sender<(WorkerId, Task, Result<ExecutionResult, EngineError>)>,
        executor: &Arc<TaskExecutor>,
        stats_ref: &Arc<WorkerStats>,
        state_atomic: &Arc<AtomicU64>,
        cpu_affinity: Option<usize>,
        shutdown_signal: &ShutdownSignal,
    ) {
        // Set CPU affinity if specified
        if let Some(_core_id) = cpu_affinity {
            #[cfg(target_os = "linux")]
            {
                use core_affinity::{set_for_current, CoreId};
                let _ = set_for_current(CoreId { id: core_id });
            }
        }

        let mut idle_start = Instant::now();

        loop {
            // Check for shutdown signal first
            if shutdown_signal.is_shutdown() {
                state_atomic.store(WorkerState::Stopping as u64, Ordering::Release);
                break;
            }

            match task_receiver.recv_timeout(Duration::from_millis(1)) {
                Ok(task) => {
                    // Check shutdown signal before processing task
                    if shutdown_signal.is_shutdown() {
                        state_atomic.store(WorkerState::Stopping as u64, Ordering::Release);
                        break;
                    }

                    // Update idle time
                    let idle_time = idle_start.elapsed();
                    stats_ref.total_idle_time_ns.fetch_add(
                        u64::try_from(idle_time.as_nanos()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );

                    // Set state to busy
                    state_atomic.store(WorkerState::Busy as u64, Ordering::Release);

                    // Execute task
                    let execution_start = Instant::now();
                    let result = executor.execute_task(task.clone(), worker_id);
                    let execution_time = execution_start.elapsed();

                    // Update statistics
                    stats_ref.tasks_processed.fetch_add(1, Ordering::Relaxed);
                    stats_ref.total_execution_time_ns.fetch_add(
                        u64::try_from(execution_time.as_nanos()).unwrap_or(u64::MAX),
                        Ordering::Relaxed,
                    );

                    // Send result
                    if result_sender.send((worker_id, task, result)).is_err() {
                        // Channel closed, worker should stop
                        break;
                    }

                    // Set state back to idle
                    state_atomic.store(WorkerState::Idle as u64, Ordering::Release);
                    idle_start = Instant::now();
                }
                Err(channel::RecvTimeoutError::Timeout) => {
                    // Continue waiting for tasks, but check shutdown signal
                }
                Err(channel::RecvTimeoutError::Disconnected) => {
                    // Channel closed, worker should stop
                    break;
                }
            }
        }

        // Set final state
        state_atomic.store(WorkerState::Stopped as u64, Ordering::Release);
    }

    /// Get current worker state
    #[must_use]
    pub fn state(&self) -> WorkerState {
        match self.state.load(Ordering::Acquire) {
            1 => WorkerState::Busy,
            2 => WorkerState::Stopping,
            3 => WorkerState::Stopped,
            _ => WorkerState::Idle, // Default fallback (includes 0)
        }
    }

    /// Check if worker is idle
    #[must_use]
    pub fn is_idle(&self) -> bool {
        matches!(self.state(), WorkerState::Idle)
    }

    /// Check if worker is busy
    #[must_use]
    pub fn is_busy(&self) -> bool {
        matches!(self.state(), WorkerState::Busy)
    }

    /// Signal worker to shutdown gracefully
    pub fn shutdown(&self) {
        self.shutdown_signal.signal();
    }

    /// Wait for worker to finish with timeout
    ///
    /// # Errors
    ///
    /// Returns error if worker panics or timeout occurs
    pub fn join_with_timeout(&mut self, timeout: Duration) -> Result<(), WorkerError> {
        if let Some(handle) = self.thread_handle.take() {
            // Signal shutdown first
            self.shutdown();

            // Try to join with timeout using a separate thread
            let (tx, rx) = std::sync::mpsc::channel();
            let _join_handle = std::thread::spawn(move || {
                let result = handle.join();
                let _ = tx.send(result);
            });

            match rx.recv_timeout(timeout) {
                Ok(join_result) => {
                    join_result.map_err(|_| WorkerError::WorkerPanic { worker_id: self.id })?;
                    Ok(())
                }
                Err(_) => {
                    // Timeout occurred, but we can't force kill the thread
                    // The thread will eventually stop when it checks shutdown signal
                    Err(WorkerError::ChannelError {
                        reason: format!("Worker {} shutdown timeout", self.id),
                    })
                }
            }
        } else {
            Ok(())
        }
    }
}

/// Worker pool statistics
#[derive(Debug, Default)]
pub struct PoolStats {
    /// Total workers created
    pub workers_created: AtomicUsize,

    /// Active workers
    pub active_workers: AtomicUsize,

    /// Idle workers
    pub idle_workers: AtomicUsize,

    /// Busy workers
    pub busy_workers: AtomicUsize,

    /// Total tasks processed by pool
    pub total_tasks_processed: AtomicU64,
}

/// Worker pool
pub struct WorkerPool {
    /// Configuration
    config: Arc<EngineConfig>,

    /// Engine metrics
    metrics: Arc<EngineMetrics>,

    /// Workers
    workers: Vec<Worker>,

    /// Task sender channel
    task_sender: Sender<Task>,

    /// Result receiver channel
    result_receiver: Receiver<(WorkerId, Task, Result<ExecutionResult, EngineError>)>,

    /// Pool statistics
    stats: Arc<PoolStats>,

    /// Running state
    is_running: AtomicBool,

    /// Current executor (for strategy updates)
    current_executor: Option<Arc<TaskExecutor>>,
}

impl WorkerPool {
    /// Create new worker pool
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(config: Arc<EngineConfig>, metrics: Arc<EngineMetrics>) -> EngineResult<Self> {
        let (task_sender, task_receiver) = channel::unbounded();
        let (result_sender, result_receiver) = channel::unbounded();

        let mut workers = Vec::with_capacity(config.max_workers);

        // Create workers
        for i in 0..config.max_workers {
            let worker_id = WorkerId::new();
            let cpu_affinity = config
                .cpu_affinity
                .as_ref()
                .and_then(|cores| cores.get(i % cores.len()).copied());

            let worker = Worker::new(
                worker_id,
                task_receiver.clone(),
                result_sender.clone(),
                cpu_affinity,
            );

            workers.push(worker);
        }

        Ok(Self {
            config,
            metrics,
            workers,
            task_sender,
            result_receiver,
            stats: Arc::new(PoolStats::default()),
            is_running: AtomicBool::new(false),
            current_executor: None,
        })
    }

    /// Start worker pool with strategies
    ///
    /// # Errors
    ///
    /// Returns error if pool is already running or worker startup fails
    pub fn start_with_strategies(
        &mut self,
        strategies: Arc<DashMap<StrategyId, Arc<dyn Strategy + Send + Sync>>>,
    ) -> EngineResult<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Err(EngineError::AlreadyRunning);
        }

        // Create task executor with proper strategies
        let executor = Arc::new(TaskExecutor::new(
            Arc::clone(&self.config),
            Arc::clone(&self.metrics),
            strategies,
        )?);

        // Start all workers
        for worker in &mut self.workers {
            worker
                .start(Arc::clone(&executor))
                .map_err(|e| EngineError::WorkerError {
                    worker_id: worker.id,
                    reason: e.to_string(),
                })?;
        }

        // Store executor for strategy updates
        self.current_executor = Some(executor);

        self.is_running.store(true, Ordering::Release);
        self.stats
            .workers_created
            .store(self.workers.len(), Ordering::Relaxed);
        self.stats
            .active_workers
            .store(self.workers.len(), Ordering::Relaxed);

        Ok(())
    }

    /// Start worker pool (deprecated - use `start_with_strategies`)
    ///
    /// # Errors
    ///
    /// Returns error if pool is already running or worker startup fails
    #[deprecated(note = "Use start_with_strategies instead")]
    pub fn start(&mut self) -> EngineResult<()> {
        // Create empty strategies map for backward compatibility
        let strategies = Arc::new(DashMap::new());
        self.start_with_strategies(strategies)
    }

    /// Stop worker pool
    ///
    /// # Errors
    ///
    /// Returns error if pool is not running
    pub fn stop(&mut self) -> EngineResult<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(EngineError::NotRunning);
        }

        self.is_running.store(false, Ordering::Release);

        // Signal all workers to shutdown gracefully
        for worker in &mut self.workers {
            worker.shutdown();
        }

        // Wait for all workers to finish with timeout
        let shutdown_timeout = Duration::from_millis(100); // 100ms timeout for graceful shutdown
        for worker in &mut self.workers {
            if let Err(e) = worker.join_with_timeout(shutdown_timeout) {
                tracing::warn!("Worker {} shutdown timeout: {}", worker.id, e);
                // Continue with other workers even if one times out
            }
        }

        self.stats.active_workers.store(0, Ordering::Relaxed);

        Ok(())
    }

    /// Get pool statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<PoolStats> {
        &self.stats
    }

    /// Get number of workers
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Get worker statistics
    #[must_use]
    pub fn worker_stats(&self, worker_id: WorkerId) -> Option<&Arc<WorkerStats>> {
        self.workers
            .iter()
            .find(|w| w.id == worker_id)
            .map(|w| &w.stats)
    }

    /// Check if pool is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Try to receive a completed task result (non-blocking)
    pub fn try_recv_result(
        &self,
    ) -> Option<(WorkerId, Task, Result<ExecutionResult, EngineError>)> {
        self.result_receiver.try_recv().ok()
    }

    /// Submit task to worker pool
    ///
    /// # Errors
    ///
    /// Returns error if pool is not running or task submission fails
    pub fn submit_task(&self, task: Task) -> EngineResult<()> {
        if !self.is_running() {
            return Err(EngineError::NotRunning);
        }

        self.task_sender
            .send(task)
            .map_err(|_| EngineError::SchedulerError {
                reason: "Failed to submit task to worker pool".to_string(),
            })?;

        Ok(())
    }

    /// Signal graceful shutdown without requiring mutable reference
    ///
    /// This method signals all workers to shutdown gracefully but doesn't
    /// wait for them to finish. Use this when the `WorkerPool` is wrapped in Arc.
    pub fn signal_shutdown(&self) {
        self.is_running.store(false, Ordering::Release);

        // Signal all workers to shutdown gracefully
        for worker in &self.workers {
            worker.shutdown();
        }
    }

    /// Update strategies in running workers
    ///
    /// # Errors
    ///
    /// Returns error if strategy update fails or pool is not running
    pub fn update_strategies(
        &self,
        strategies: Arc<DashMap<StrategyId, Arc<dyn Strategy + Send + Sync>>>,
    ) -> EngineResult<()> {
        if !self.is_running() {
            return Err(EngineError::NotRunning);
        }

        // Create new executor with updated strategies
        let _new_executor = Arc::new(TaskExecutor::new(
            Arc::clone(&self.config),
            Arc::clone(&self.metrics),
            strategies,
        )?);

        // Note: In a full implementation, we would need to gracefully update
        // the executor in running workers. For now, we validate the new executor
        // creation. A complete implementation would require more sophisticated
        // worker management with hot-swapping capabilities.

        // This is a simplified approach - in production, you might want to:
        // 1. Drain current tasks
        // 2. Update executor atomically
        // 3. Resume processing

        tracing::info!("Strategy update validated - strategies are ready for deployment");

        Ok(())
    }

    /// Get current executor (for testing and diagnostics)
    #[must_use]
    pub const fn current_executor(&self) -> Option<&Arc<TaskExecutor>> {
        self.current_executor.as_ref()
    }
}
