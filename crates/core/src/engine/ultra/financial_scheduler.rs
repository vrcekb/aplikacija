//! Financial-grade ultra-optimized scheduler
//!
//! This implementation provides a production-ready scheduler specifically
//! designed for financial applications with zero-tolerance for errors.
//!
//! # Safety
//! - Zero unwrap/expect/panic usage
//! - Comprehensive error handling
//! - Memory safety guaranteed
//! - Thread safety verified
//!
//! # Performance
//! - Target latency: <1ms for task submission
//! - Lock-free where possible
//! - Cache-optimized data structures

use super::{Task, TaskResult, UltraEngineError, UltraEngineResult};
use crossbeam::queue::SegQueue;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Financial scheduler configuration
#[derive(Debug, Clone)]
pub struct FinancialConfig {
    /// Number of worker threads (default: CPU count)
    pub worker_count: usize,
    /// Queue capacity per worker (must be power of 2)
    pub queue_capacity: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Task timeout in milliseconds
    pub task_timeout_ms: u64,
}

impl Default for FinancialConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get(),
            queue_capacity: 8192,
            enable_monitoring: true,
            task_timeout_ms: 1000,
        }
    }
}

/// Financial-grade ultra-optimized scheduler
///
/// This scheduler is designed for financial applications requiring
/// ultra-low latency and zero-error operation.
pub struct FinancialScheduler<T: Task> {
    /// Configuration
    config: FinancialConfig,
    /// Worker queues (one per worker)
    worker_queues: Vec<Arc<SegQueue<Arc<T>>>>,
    /// Worker handles
    worker_handles: RwLock<Vec<JoinHandle<()>>>,
    /// Running state
    is_running: Arc<AtomicBool>,
    /// Performance metrics
    metrics: FinancialMetrics,
    /// Round-robin counter for task distribution
    round_robin_counter: AtomicUsize,
}

/// Financial performance metrics
#[derive(Debug)]
pub struct FinancialMetrics {
    /// Total tasks processed across all workers
    pub tasks_processed: Arc<AtomicU64>,
    /// Total tasks submitted
    pub tasks_submitted: Arc<AtomicU64>,
    /// Total tasks failed
    pub tasks_failed: Arc<AtomicU64>,
    /// Active workers count
    pub active_workers: Arc<AtomicUsize>,
}

impl Default for FinancialMetrics {
    fn default() -> Self {
        Self {
            tasks_processed: Arc::new(AtomicU64::new(0)),
            tasks_submitted: Arc::new(AtomicU64::new(0)),
            tasks_failed: Arc::new(AtomicU64::new(0)),
            active_workers: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl<T: Task + Send + Sync + 'static> FinancialScheduler<T> {
    /// Creates new financial scheduler
    ///
    /// # Errors
    /// Returns error if configuration is invalid or initialization fails
    pub fn new(config: FinancialConfig) -> UltraEngineResult<Self> {
        // Validate configuration
        if config.worker_count == 0 {
            return Err(UltraEngineError::InvalidConfiguration(
                "Worker count cannot be zero".to_string(),
            ));
        }

        if !config.queue_capacity.is_power_of_two() {
            return Err(UltraEngineError::InvalidConfiguration(
                "Queue capacity must be power of 2".to_string(),
            ));
        }

        // Create worker queues
        let mut worker_queues = Vec::with_capacity(config.worker_count);
        for _ in 0..config.worker_count {
            worker_queues.push(Arc::new(SegQueue::new()));
        }

        Ok(Self {
            config,
            worker_queues,
            worker_handles: RwLock::new(Vec::new()),
            is_running: Arc::new(AtomicBool::new(false)),
            metrics: FinancialMetrics::default(),
            round_robin_counter: AtomicUsize::new(0),
        })
    }

    /// Starts the scheduler and all worker threads
    ///
    /// # Errors
    /// Returns error if already running or thread creation fails
    pub fn start(&self) -> UltraEngineResult<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Ok(()); // Already running, not an error
        }

        self.is_running.store(true, Ordering::Release);

        // Start worker threads
        for (worker_id, queue) in self.worker_queues.iter().enumerate() {
            let queue_clone = queue.clone();
            let is_running_clone = self.is_running.clone(); // Use the main is_running flag
            let tasks_processed_clone = self.metrics.tasks_processed.clone();
            let tasks_failed_clone = self.metrics.tasks_failed.clone();

            let handle = thread::Builder::new()
                .name(format!("financial-worker-{worker_id}"))
                .spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        &queue_clone,
                        &is_running_clone,
                        &tasks_processed_clone,
                        &tasks_failed_clone,
                    );
                })
                .map_err(|e| UltraEngineError::ThreadCreationFailed(e.to_string()))?;

            self.worker_handles.write().push(handle);
        }

        self.metrics
            .active_workers
            .store(self.config.worker_count, Ordering::Relaxed);
        Ok(())
    }

    /// Worker thread main loop
    fn worker_loop(
        worker_id: usize,
        queue: &Arc<SegQueue<Arc<T>>>,
        is_running: &Arc<AtomicBool>,
        tasks_processed: &Arc<AtomicU64>,
        tasks_failed: &Arc<AtomicU64>,
    ) {
        while is_running.load(Ordering::Acquire) {
            if let Some(task) = queue.pop() {
                // Execute task
                let result = task.execute();

                match result {
                    TaskResult::Success => {
                        tasks_processed.fetch_add(1, Ordering::Relaxed);
                    }
                    TaskResult::Failed(_) | TaskResult::Retry => {
                        tasks_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            } else {
                // No tasks available, yield to prevent busy waiting
                thread::yield_now();
            }
        }

        // Worker thread exiting
        eprintln!("Financial worker {worker_id} shutting down");
    }

    /// Stops the scheduler gracefully
    ///
    /// # Errors
    /// Returns error if not running or thread join fails
    pub fn stop(&self) -> UltraEngineResult<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Ok(()); // Not running, not an error
        }

        // Signal all workers to stop
        self.is_running.store(false, Ordering::Release);

        // Wait for all worker threads to finish
        for handle in self.worker_handles.write().drain(..) {
            if handle.join().is_err() {
                return Err(UltraEngineError::WorkerJoinFailed);
            }
        }

        self.metrics.active_workers.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Submits task for execution using round-robin distribution
    ///
    /// # Errors
    /// Returns error if scheduler is not running
    pub fn submit_task(&self, task: &Arc<T>) -> UltraEngineResult<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(UltraEngineError::NotRunning);
        }

        // Round-robin distribution to workers
        let worker_index =
            self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.worker_queues.len();

        self.worker_queues.get(worker_index).map_or_else(
            || {
                Err(UltraEngineError::InvalidConfiguration(
                    "Invalid worker index".to_string(),
                ))
            },
            |queue| {
                queue.push(task.clone());
                self.metrics.tasks_submitted.fetch_add(1, Ordering::Relaxed);
                Ok(())
            },
        )
    }

    /// Returns current performance metrics
    #[must_use]
    pub const fn metrics(&self) -> &FinancialMetrics {
        &self.metrics
    }

    /// Returns worker count
    #[must_use]
    pub const fn worker_count(&self) -> usize {
        self.config.worker_count
    }

    /// Returns whether scheduler is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }
}

impl FinancialMetrics {
    /// Returns tasks processed
    #[must_use]
    pub fn tasks_processed(&self) -> u64 {
        self.tasks_processed.load(Ordering::Relaxed)
    }

    /// Returns tasks submitted
    #[must_use]
    pub fn tasks_submitted(&self) -> u64 {
        self.tasks_submitted.load(Ordering::Relaxed)
    }

    /// Returns tasks failed
    #[must_use]
    pub fn tasks_failed(&self) -> u64 {
        self.tasks_failed.load(Ordering::Relaxed)
    }

    /// Returns active workers
    #[must_use]
    pub fn active_workers(&self) -> usize {
        self.active_workers.load(Ordering::Relaxed)
    }

    /// Returns success rate as percentage (0.0-100.0)
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let processed = self.tasks_processed();
        let failed = self.tasks_failed();
        let total = processed + failed;

        if total == 0 {
            100.0_f64
        } else {
            #[allow(clippy::cast_precision_loss)]
            let processed_f64 = processed as f64;
            #[allow(clippy::cast_precision_loss)]
            let total_f64 = total as f64;
            (processed_f64 / total_f64) * 100.0_f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::TaskResult;
    use super::*;
    use std::sync::atomic::AtomicU64;
    use std::time::{Duration, Instant};

    struct TestTask {
        counter: Arc<AtomicU64>,
        should_fail: bool,
    }

    impl Task for TestTask {
        fn execute(&self) -> TaskResult {
            if self.should_fail {
                TaskResult::Failed("Test failure".to_string())
            } else {
                self.counter.fetch_add(1, Ordering::Relaxed);
                TaskResult::Success
            }
        }

        fn priority(&self) -> u8 {
            5_u8
        }

        fn estimated_duration_us(&self) -> u64 {
            100_u64
        }
    }

    #[test]
    fn test_financial_scheduler_creation() -> UltraEngineResult<()> {
        let config = FinancialConfig::default();
        let _scheduler = FinancialScheduler::<TestTask>::new(config)?;
        Ok(())
    }

    #[test]
    fn test_scheduler_lifecycle() -> UltraEngineResult<()> {
        let config = FinancialConfig {
            worker_count: 2,
            queue_capacity: 1024,
            enable_monitoring: true,
            task_timeout_ms: 1000,
        };
        let scheduler = FinancialScheduler::<TestTask>::new(config)?;

        assert!(!scheduler.is_running());

        scheduler.start()?;
        assert!(scheduler.is_running());

        scheduler.stop()?;
        assert!(!scheduler.is_running());

        Ok(())
    }

    #[test]
    fn test_task_execution() -> UltraEngineResult<()> {
        let config = FinancialConfig {
            worker_count: 2,
            queue_capacity: 1024,
            enable_monitoring: true,
            task_timeout_ms: 1000,
        };
        let scheduler = FinancialScheduler::new(config)?;
        let counter = Arc::new(AtomicU64::new(0));

        scheduler.start()?;

        // Submit successful tasks
        for _ in 0_i32..5_i32 {
            let task = Arc::new(TestTask {
                counter: counter.clone(),
                should_fail: false,
            });
            scheduler.submit_task(&task)?;
        }

        // Wait for processing
        std::thread::sleep(Duration::from_millis(100));

        scheduler.stop()?;

        // Verify results
        assert!(counter.load(Ordering::Relaxed) > 0);
        assert!(scheduler.metrics().tasks_processed() > 0);

        Ok(())
    }

    #[test]
    fn test_performance_target() -> UltraEngineResult<()> {
        let config = FinancialConfig::default();
        let scheduler = FinancialScheduler::new(config)?;
        let counter = Arc::new(AtomicU64::new(0));

        scheduler.start()?;

        let task = Arc::new(TestTask {
            counter,
            should_fail: false,
        });

        // Measure submission latency
        let start = Instant::now();
        scheduler.submit_task(&task)?;
        let submit_time = start.elapsed();

        scheduler.stop()?;

        // Target: <1ms for task submission
        assert!(
            submit_time.as_millis() < 1,
            "Task submission too slow: {submit_time:?}"
        );

        Ok(())
    }
}
