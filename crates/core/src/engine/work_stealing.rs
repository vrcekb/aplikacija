//! Work-Stealing Scheduler - Ultra-Performance Multi-threaded Task Scheduling
//!
//! Implements work-stealing algorithm for optimal thread utilization and minimal contention.
//! Designed for <1ms latency requirements in MEV/DeFi applications.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_deque::{Injector, Stealer, Worker};
use thiserror::Error;

use super::executor::{Task, TaskPriority};
use super::{EngineConfig, EngineMetrics};
use crate::types::TaskId;

/// Work-stealing scheduler error types
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum WorkStealingError {
    /// Worker thread failed to start
    #[error("Failed to start worker thread {worker_id}: {reason}")]
    WorkerStartFailed {
        /// Worker identifier
        worker_id: usize,
        /// Failure reason
        reason: String,
    },

    /// Task injection failed
    #[error("Failed to inject task: {reason}")]
    TaskInjectionFailed {
        /// Failure reason
        reason: String,
    },

    /// Scheduler not initialized
    #[error("Work-stealing scheduler not initialized")]
    NotInitialized,

    /// Invalid worker count
    #[error("Invalid worker count: {count} (must be 1-{max})")]
    InvalidWorkerCount {
        /// Requested count
        count: usize,
        /// Maximum allowed
        max: usize,
    },
}

/// Work-stealing worker statistics
#[derive(Debug, Default)]
pub struct WorkerStats {
    /// Tasks executed by this worker
    pub tasks_executed: AtomicU64,
    /// Tasks stolen from other workers
    pub tasks_stolen: AtomicU64,
    /// Tasks given to other workers
    pub tasks_given: AtomicU64,
    /// Local queue hits
    pub local_hits: AtomicU64,
    /// Global queue hits
    pub global_hits: AtomicU64,
    /// Idle cycles (no work found)
    pub idle_cycles: AtomicU64,
}

/// Work-stealing worker
struct WorkStealingWorker {
    /// Worker ID
    id: usize,
    /// Local task queue
    worker: Worker<Task>,
    /// Statistics
    stats: Arc<WorkerStats>,
    /// Thread handle
    handle: Option<JoinHandle<()>>,
    /// Running state
    running: Arc<AtomicBool>,
}

impl WorkStealingWorker {
    /// Create new worker
    fn new(id: usize) -> Self {
        Self {
            id,
            worker: Worker::new_fifo(),
            stats: Arc::new(WorkerStats::default()),
            handle: None,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start worker thread
    fn start(
        &mut self,
        global_queue: Arc<Injector<Task>>,
        stealers: Vec<Stealer<Task>>,
        metrics: Arc<EngineMetrics>,
    ) -> Result<(), WorkStealingError> {
        if self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.running.store(true, Ordering::Release);
        let worker_stealer = self.worker.stealer();
        let stats = Arc::clone(&self.stats);
        let running = Arc::clone(&self.running);
        let worker_id = self.id;

        let handle = thread::Builder::new()
            .name(format!("tallyio-worker-{worker_id}"))
            .spawn(move || {
                Self::worker_loop(
                    worker_id,
                    &worker_stealer,
                    &global_queue,
                    &stealers,
                    &stats,
                    &metrics,
                    &running,
                );
            })
            .map_err(|e| WorkStealingError::WorkerStartFailed {
                worker_id,
                reason: e.to_string(),
            })?;

        self.handle = Some(handle);
        Ok(())
    }

    /// Stop worker thread
    fn stop(&mut self) -> Result<(), WorkStealingError> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.running.store(false, Ordering::Release);

        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| WorkStealingError::WorkerStartFailed {
                    worker_id: self.id,
                    reason: "Failed to join worker thread".to_string(),
                })?;
        }

        Ok(())
    }

    /// Worker main loop
    fn worker_loop(
        worker_id: usize,
        worker: &Stealer<Task>,
        global_queue: &Arc<Injector<Task>>,
        stealers: &[Stealer<Task>],
        stats: &Arc<WorkerStats>,
        metrics: &Arc<EngineMetrics>,
        running: &Arc<AtomicBool>,
    ) {
        while running.load(Ordering::Acquire) {
            let task_found = Self::find_and_execute_task(
                worker_id,
                worker,
                global_queue,
                stealers,
                stats,
                metrics,
            );

            if !task_found {
                stats.idle_cycles.fetch_add(1, Ordering::Relaxed);
                // Brief yield to prevent busy waiting
                thread::yield_now();
            }
        }
    }

    /// Find and execute a task
    fn find_and_execute_task(
        worker_id: usize,
        worker: &Stealer<Task>,
        global_queue: &Arc<Injector<Task>>,
        stealers: &[Stealer<Task>],
        stats: &Arc<WorkerStats>,
        metrics: &Arc<EngineMetrics>,
    ) -> bool {
        // 1. Try local queue first (fastest path)
        if let crossbeam_deque::Steal::Success(task) = worker.steal() {
            stats.local_hits.fetch_add(1, Ordering::Relaxed);
            Self::execute_task(&task, stats, metrics);
            return true;
        }

        // 2. Try global queue
        if let crossbeam_deque::Steal::Success(task) = global_queue.steal() {
            stats.global_hits.fetch_add(1, Ordering::Relaxed);
            Self::execute_task(&task, stats, metrics);
            return true;
        }

        // 3. Try stealing from other workers
        for (i, stealer) in stealers.iter().enumerate() {
            if i != worker_id {
                if let crossbeam_deque::Steal::Success(task) = stealer.steal() {
                    stats.tasks_stolen.fetch_add(1, Ordering::Relaxed);
                    Self::execute_task(&task, stats, metrics);
                    return true;
                }
            }
        }

        false
    }

    /// Execute a task
    fn execute_task(task: &Task, stats: &Arc<WorkerStats>, metrics: &Arc<EngineMetrics>) {
        let start = Instant::now();

        // TODO: Actual task execution logic
        // For now, simulate processing based on priority
        let processing_time = match task.priority {
            TaskPriority::Critical => Duration::from_nanos(100),
            TaskPriority::High => Duration::from_nanos(200),
            TaskPriority::Normal => Duration::from_nanos(500),
            TaskPriority::Low => Duration::from_micros(1),
        };

        thread::sleep(processing_time);

        let execution_time = start.elapsed();
        stats.tasks_executed.fetch_add(1, Ordering::Relaxed);

        // Update metrics
        metrics.tasks_processed.fetch_add(1, Ordering::Relaxed);
        metrics.total_execution_time_ns.fetch_add(
            u64::try_from(execution_time.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    /// Push task to local queue
    fn push_local(&self, task: Task) {
        self.worker.push(task);
    }

    /// Get worker statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<WorkerStats> {
        &self.stats
    }
}

/// Work-stealing scheduler
pub struct WorkStealingScheduler {
    /// Configuration
    #[allow(dead_code)]
    config: Arc<EngineConfig>,
    /// Engine metrics
    metrics: Arc<EngineMetrics>,
    /// Global task queue (injector)
    global_queue: Arc<Injector<Task>>,
    /// Worker threads
    workers: Vec<WorkStealingWorker>,
    /// Stealer handles for work stealing
    stealers: Vec<Stealer<Task>>,
    /// Running state
    is_running: AtomicBool,
    /// Total tasks submitted
    tasks_submitted: AtomicU64,
    /// Round-robin counter for load balancing
    round_robin_counter: AtomicUsize,
}

impl WorkStealingScheduler {
    /// Create new work-stealing scheduler
    ///
    /// # Errors
    ///
    /// Returns error if worker count is invalid
    pub fn new(
        config: Arc<EngineConfig>,
        metrics: Arc<EngineMetrics>,
        worker_count: Option<usize>,
    ) -> Result<Self, WorkStealingError> {
        let num_workers = worker_count.unwrap_or_else(num_cpus::get);
        let max_workers = num_cpus::get() * 2; // Allow up to 2x CPU cores

        if num_workers == 0 || num_workers > max_workers {
            return Err(WorkStealingError::InvalidWorkerCount {
                count: num_workers,
                max: max_workers,
            });
        }

        let mut workers = Vec::with_capacity(num_workers);
        let mut stealers = Vec::with_capacity(num_workers);

        // Create workers and collect stealers
        for i in 0..num_workers {
            let worker = WorkStealingWorker::new(i);
            stealers.push(worker.worker.stealer());
            workers.push(worker);
        }

        Ok(Self {
            config,
            metrics,
            global_queue: Arc::new(Injector::new()),
            workers,
            stealers,
            is_running: AtomicBool::new(false),
            tasks_submitted: AtomicU64::new(0),
            round_robin_counter: AtomicUsize::new(0),
        })
    }

    /// Start the scheduler
    ///
    /// # Errors
    ///
    /// Returns error if workers fail to start
    pub fn start(&mut self) -> Result<(), WorkStealingError> {
        if self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        // Start all workers
        for worker in &mut self.workers {
            worker.start(
                Arc::clone(&self.global_queue),
                self.stealers.clone(),
                Arc::clone(&self.metrics),
            )?;
        }

        self.is_running.store(true, Ordering::Release);
        Ok(())
    }

    /// Stop the scheduler
    ///
    /// # Errors
    ///
    /// Returns error if workers fail to stop
    pub fn stop(&mut self) -> Result<(), WorkStealingError> {
        if !self.is_running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.is_running.store(false, Ordering::Release);

        // Stop all workers
        for worker in &mut self.workers {
            worker.stop()?;
        }

        Ok(())
    }

    /// Submit task for execution
    ///
    /// # Errors
    ///
    /// Returns error if scheduler is not running
    pub fn submit_task(&self, task: Task) -> Result<TaskId, WorkStealingError> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(WorkStealingError::NotInitialized);
        }

        let task_id = task.id;

        // Use round-robin to distribute tasks to workers
        let worker_index =
            self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.workers.len();

        // Try to push to local worker queue first
        if let Some(worker) = self.workers.get(worker_index) {
            worker.push_local(task);
        } else {
            // Fallback to global queue
            self.global_queue.push(task);
        }

        self.tasks_submitted.fetch_add(1, Ordering::Relaxed);
        Ok(task_id)
    }

    /// Get scheduler statistics
    #[must_use]
    pub fn stats(&self) -> Vec<&Arc<WorkerStats>> {
        self.workers.iter().map(WorkStealingWorker::stats).collect()
    }

    /// Get total tasks submitted
    #[must_use]
    pub fn total_tasks_submitted(&self) -> u64 {
        self.tasks_submitted.load(Ordering::Relaxed)
    }

    /// Check if scheduler is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get worker count
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for WorkStealingScheduler {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
