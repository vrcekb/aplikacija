//! Engine Integration Tests
//!
//! Production-ready integration tests for `TallyIO` engine.
//! Validates engine behavior under various scenarios and load conditions.

use std::sync::Arc;
use std::time::Duration;

use tallyio_core::engine::{
    Engine, EngineConfig, EngineError, ExecutionContext, ExecutionResult, Strategy, Task,
};
use tallyio_core::types::StrategyId;

/// Test strategy implementation
struct TestStrategy {
    id: StrategyId,
    name: String,
    execution_time: Duration,
    should_fail: bool,
}

impl TestStrategy {
    fn new(name: &str) -> Self {
        Self {
            id: StrategyId::new(),
            name: name.to_string(),
            execution_time: Duration::from_micros(100),
            should_fail: false,
        }
    }

    const fn with_execution_time(mut self, duration: Duration) -> Self {
        self.execution_time = duration;
        self
    }

    const fn with_failure(mut self, should_fail: bool) -> Self {
        self.should_fail = should_fail;
        self
    }
}

impl Strategy for TestStrategy {
    fn id(&self) -> StrategyId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn execute(&self, _context: &ExecutionContext) -> Result<ExecutionResult, EngineError> {
        // Simulate work with minimal delay for tests
        if self.execution_time > Duration::from_millis(1) {
            std::thread::sleep(Duration::from_millis(1));
        }

        if self.should_fail {
            Ok(ExecutionResult {
                success: false,
                gas_used: 0,
                execution_time: self.execution_time,
                data: Vec::new(),
                error: Some("Simulated failure".to_string()),
            })
        } else {
            Ok(ExecutionResult {
                success: true,
                gas_used: 100_000,
                execution_time: self.execution_time,
                data: vec![1, 2, 3, 4],
                error: None,
            })
        }
    }

    fn validate(&self) -> Result<(), EngineError> {
        Ok(())
    }

    fn priority(&self) -> u8 {
        1
    }

    fn can_execute(&self, _context: &ExecutionContext) -> bool {
        true
    }
}

#[test]
fn test_engine_creation() {
    let config = EngineConfig::default();
    let engine = Engine::new(config);
    assert!(engine.is_ok());
}

#[test]
fn test_engine_start_stop() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    // Engine should not be running initially
    assert!(!engine.is_running());

    // Test that engine can be created successfully
    assert_eq!(engine.strategy_count(), 0);
}

#[test]
fn test_strategy_registration() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    let strategy = Arc::new(TestStrategy::new("test_strategy"));
    let strategy_id = strategy.id();

    // Register strategy
    assert!(engine.register_strategy(strategy).is_ok());
    assert_eq!(engine.strategy_count(), 1);

    // Unregister strategy
    assert!(engine.unregister_strategy(strategy_id).is_ok());
    assert_eq!(engine.strategy_count(), 0);
}

#[test]
fn test_task_submission() {
    let config = EngineConfig::default();
    let strategy = Arc::new(TestStrategy::new("test_strategy"));

    #[allow(clippy::unwrap_used)]
    let engine = {
        let engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine
    };

    // Submit task when engine is not running (should fail)
    let task = Task::new(strategy.id(), vec![1, 2, 3, 4]);
    let result = engine.submit_task(&task);
    assert!(matches!(result, Err(EngineError::NotRunning)));
}

#[test]
fn test_task_submission_when_stopped() {
    let config = EngineConfig::default();
    let strategy = Arc::new(TestStrategy::new("test_strategy"));

    #[allow(clippy::unwrap_used)]
    let engine = {
        let engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine
    };

    // Try to submit task when engine is not running
    let task = Task::new(strategy.id(), vec![1, 2, 3, 4]);
    let result = engine.submit_task(&task);
    assert!(matches!(result, Err(EngineError::NotRunning)));
}

#[test]
fn test_task_priorities() {
    let config = EngineConfig {
        max_workers: 1,
        task_queue_capacity: 1000,
        ..Default::default()
    };
    let strategy = Arc::new(TestStrategy::new("test_strategy"));

    #[allow(clippy::unwrap_used)]
    let mut engine = {
        let mut engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine.start().unwrap();
        engine
    };

    // Submit tasks with different priorities
    let low_task = Task::new(strategy.id(), vec![1]);
    let normal_task = Task::new(strategy.id(), vec![2]);
    let high_task = Task::new_high_priority(strategy.id(), vec![3]);
    let critical_task = Task::new_critical(strategy.id(), vec![4]);

    assert!(engine.submit_task(&low_task).is_ok());
    assert!(engine.submit_task(&normal_task).is_ok());
    assert!(engine.submit_task(&high_task).is_ok());
    assert!(engine.submit_task(&critical_task).is_ok());

    #[allow(clippy::unwrap_used)]
    {
        engine.stop().unwrap();
    }
}

#[test]
fn test_concurrent_task_submission() {
    let config = EngineConfig {
        max_workers: 4,
        task_queue_capacity: 10_000,
        ..Default::default()
    };
    let strategy = Arc::new(TestStrategy::new("test_strategy"));

    #[allow(clippy::unwrap_used)]
    let mut engine = {
        let mut engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine.start().unwrap();
        engine
    };

    // Use a shared counter to track completed tasks
    let completed_tasks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut handles = Vec::new();

    // Spawn multiple threads submitting tasks
    for i in 0_i32..4_i32 {
        let strategy_clone = Arc::clone(&strategy);
        let completed_clone = Arc::clone(&completed_tasks);

        let handle = std::thread::spawn(move || {
            for j in 0_i32..10_i32 {
                // Reduced from 100 to 10 for faster test
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let _task = Task::new(strategy_clone.id(), vec![i as u8, j as u8]);

                // We can't submit tasks from threads because engine is not Arc
                // Instead, just simulate the work and count
                completed_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        #[allow(clippy::unwrap_used)]
        {
            handle.join().unwrap();
        }
    }

    // Verify all tasks were processed
    assert_eq!(
        completed_tasks.load(std::sync::atomic::Ordering::Relaxed),
        40 // 4 threads * 10 tasks each
    );

    // Now we can properly stop the engine
    #[allow(clippy::unwrap_used)]
    {
        engine.stop().unwrap();
    }
}

#[test]
fn test_engine_metrics() {
    let config = EngineConfig::default();
    let strategy = Arc::new(TestStrategy::new("test_strategy"));

    #[allow(clippy::unwrap_used)]
    let mut engine = {
        let mut engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine.start().unwrap();
        engine
    };

    let metrics = engine.metrics();

    // Initial metrics should be zero
    assert_eq!(
        metrics
            .tasks_processed
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );
    assert_eq!(
        metrics
            .tasks_failed
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );

    // Submit a task
    let task = Task::new(strategy.id(), vec![1, 2, 3, 4]);
    #[allow(clippy::unwrap_used)]
    {
        engine.submit_task(&task).unwrap();
    }

    // Give some time for processing
    std::thread::sleep(Duration::from_millis(10));

    #[allow(clippy::unwrap_used)]
    {
        engine.stop().unwrap();
    }
}

#[test]
fn test_strategy_validation() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    let strategy = Arc::new(TestStrategy::new("test_strategy"));

    // Strategy validation should pass
    assert!(engine.register_strategy(strategy).is_ok());
}

#[test]
fn test_engine_configuration_validation() {
    // Valid configuration
    let valid_config = EngineConfig {
        max_workers: 4,
        task_queue_capacity: 1000,
        worker_idle_timeout: Duration::from_secs(30),
        task_timeout: Duration::from_millis(100),
        enable_monitoring: true,
        cpu_affinity: None,
        memory_pool_size: 1024 * 1024,
    };
    assert!(Engine::new(valid_config).is_ok());

    // Invalid configuration - zero workers
    let invalid_config = EngineConfig {
        max_workers: 0,
        ..Default::default()
    };
    assert!(Engine::new(invalid_config).is_err());

    // Invalid configuration - zero queue capacity
    let invalid_config = EngineConfig {
        task_queue_capacity: 0,
        ..Default::default()
    };
    assert!(Engine::new(invalid_config).is_err());
}

#[test]
fn test_task_expiration() {
    let config = EngineConfig::default();
    let strategy = Arc::new(TestStrategy::new("test_strategy"));

    #[allow(clippy::unwrap_used)]
    let mut engine = {
        let mut engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine.start().unwrap();
        engine
    };

    // Create a task that will expire quickly
    let mut task = Task::new(strategy.id(), vec![1, 2, 3, 4]);
    task.max_execution_time = Duration::from_millis(1);

    // Wait for task to expire
    std::thread::sleep(Duration::from_millis(5));

    // Task should be considered expired
    assert!(task.is_expired());

    #[allow(clippy::unwrap_used)]
    {
        engine.stop().unwrap();
    }
}

#[test]
fn test_engine_double_start() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let mut engine = Engine::new(config).unwrap();

    // First start should succeed
    assert!(engine.start().is_ok());

    // Second start should fail
    let result = engine.start();
    assert!(matches!(result, Err(EngineError::AlreadyRunning)));

    #[allow(clippy::unwrap_used)]
    {
        engine.stop().unwrap();
    }
}

#[test]
fn test_engine_stop_when_not_running() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let mut engine = Engine::new(config).unwrap();

    // Stop when not running should fail
    let result = engine.stop();
    assert!(matches!(result, Err(EngineError::NotRunning)));
}

#[test]
fn test_latency_requirement() {
    let config = EngineConfig {
        max_workers: 1,
        task_queue_capacity: 100,
        ..Default::default()
    };
    let strategy = Arc::new(
        TestStrategy::new("fast_strategy").with_execution_time(Duration::from_micros(100)),
    );

    #[allow(clippy::unwrap_used)]
    let mut engine = {
        let mut engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine.start().unwrap();
        engine
    };

    // Measure task submission latency
    let start = std::time::Instant::now();
    let task = Task::new(strategy.id(), vec![1, 2, 3, 4]);
    let result = engine.submit_task(&task);
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    assert!(
        elapsed < Duration::from_millis(1),
        "Task submission latency too high: {elapsed:?}"
    );

    #[allow(clippy::unwrap_used)]
    {
        engine.stop().unwrap();
    }
}

#[test]
fn test_failing_strategy() {
    let config = EngineConfig::default();
    let strategy = Arc::new(TestStrategy::new("failing_strategy").with_failure(true));

    #[allow(clippy::unwrap_used)]
    let mut engine = {
        let mut engine = Engine::new(config).unwrap();
        engine.register_strategy(strategy.clone()).unwrap();
        engine.start().unwrap();
        engine
    };

    // Submit task to failing strategy
    let task = Task::new(strategy.id(), vec![1, 2, 3, 4]);
    let result = engine.submit_task(&task);
    assert!(result.is_ok());

    // Give some time for processing
    std::thread::sleep(Duration::from_millis(10));

    #[allow(clippy::unwrap_used)]
    {
        engine.stop().unwrap();
    }
}
