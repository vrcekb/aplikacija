//! Basic engine tests that don't require start/stop
//! These tests focus on engine creation, configuration, and basic functionality

use std::sync::Arc;
use std::time::Duration;

use tallyio_core::engine::{
    Engine, EngineConfig, EngineError, ExecutionContext, ExecutionResult, Strategy, Task,
};
use tallyio_core::types::StrategyId;

/// Test strategy implementation for basic tests
#[derive(Debug)]
struct BasicTestStrategy {
    id: StrategyId,
    name: String,
}

impl BasicTestStrategy {
    fn new(name: &str) -> Self {
        Self {
            id: StrategyId::new(),
            name: name.to_string(),
        }
    }
}

impl Strategy for BasicTestStrategy {
    fn id(&self) -> StrategyId {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn validate(&self) -> Result<(), EngineError> {
        Ok(())
    }

    fn execute(&self, _context: &ExecutionContext) -> Result<ExecutionResult, EngineError> {
        Ok(ExecutionResult {
            success: true,
            gas_used: 100_000,
            execution_time: Duration::from_micros(100),
            data: vec![1, 2, 3, 4],
            error: None,
        })
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
fn test_engine_initial_state() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    // Engine should not be running initially
    assert!(!engine.is_running());

    // Should have no strategies
    assert_eq!(engine.strategy_count(), 0);
}

#[test]
fn test_strategy_registration() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    let strategy = Arc::new(BasicTestStrategy::new("test_strategy"));
    let strategy_id = strategy.id();

    // Register strategy
    assert!(engine.register_strategy(strategy).is_ok());
    assert_eq!(engine.strategy_count(), 1);

    // Unregister strategy
    assert!(engine.unregister_strategy(strategy_id).is_ok());
    assert_eq!(engine.strategy_count(), 0);
}

#[test]
fn test_task_submission_when_stopped() {
    let config = EngineConfig::default();
    let strategy = Arc::new(BasicTestStrategy::new("test_strategy"));

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
fn test_strategy_validation() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    let strategy = Arc::new(BasicTestStrategy::new("test_strategy"));

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
fn test_task_creation() {
    let strategy_id = StrategyId::new();

    // Test normal priority task
    let task = Task::new(strategy_id, vec![1, 2, 3, 4]);
    assert_eq!(task.strategy_id, strategy_id);
    assert_eq!(task.data, vec![1, 2, 3, 4]);

    // Test high priority task
    let high_task = Task::new_high_priority(strategy_id, vec![5, 6, 7, 8]);
    assert_eq!(high_task.strategy_id, strategy_id);
    assert_eq!(high_task.data, vec![5, 6, 7, 8]);

    // Test critical priority task
    let critical_task = Task::new_critical(strategy_id, vec![9, 10, 11, 12]);
    assert_eq!(critical_task.strategy_id, strategy_id);
    assert_eq!(critical_task.data, vec![9, 10, 11, 12]);
}

#[test]
fn test_task_expiration() {
    let strategy_id = StrategyId::new();

    // Create a task that will expire quickly
    let mut task = Task::new(strategy_id, vec![1, 2, 3, 4]);
    task.max_execution_time = Duration::from_millis(1);

    // Task should not be expired initially
    assert!(!task.is_expired());

    // Wait for task to expire
    std::thread::sleep(Duration::from_millis(5));

    // Task should be considered expired
    assert!(task.is_expired());
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
fn test_multiple_strategy_registration() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    let strategy1 = Arc::new(BasicTestStrategy::new("strategy1"));
    let strategy2 = Arc::new(BasicTestStrategy::new("strategy2"));
    let strategy3 = Arc::new(BasicTestStrategy::new("strategy3"));

    // Register multiple strategies
    assert!(engine.register_strategy(strategy1).is_ok());
    assert_eq!(engine.strategy_count(), 1);

    assert!(engine.register_strategy(strategy2).is_ok());
    assert_eq!(engine.strategy_count(), 2);

    assert!(engine.register_strategy(strategy3).is_ok());
    assert_eq!(engine.strategy_count(), 3);
}

#[test]
fn test_duplicate_strategy_registration() {
    let config = EngineConfig::default();
    #[allow(clippy::unwrap_used)]
    let engine = Engine::new(config).unwrap();

    let strategy1 = Arc::new(BasicTestStrategy::new("same_name"));
    let strategy2 = Arc::new(BasicTestStrategy::new("same_name"));

    // First registration should succeed
    assert!(engine.register_strategy(strategy1).is_ok());
    assert_eq!(engine.strategy_count(), 1);

    // Second registration with same name but different ID should succeed
    // (Engine currently allows multiple strategies with same name but different IDs)
    let result = engine.register_strategy(strategy2);
    assert!(result.is_ok());
    assert_eq!(engine.strategy_count(), 2);
}
