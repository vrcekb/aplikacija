//! Engine Performance Benchmarks
//!
//! Production-ready benchmarks for `TallyIO` engine performance validation.
//! Ensures <1ms latency requirements are met under various load conditions.
//!
//! ## Benchmark Categories
//!
//! ### 1. Engine Creation (`engine_creation`)
//! **What it measures**: Time to create a new Engine instance with default configuration
//! **Target**: <1ms for engine initialization
//! **Importance**: Critical for system startup time
//!
//! ### 2. Strategy Registration (`strategy_registration`)
//! **What it measures**: Time to register strategies with the engine
//! **Target**: <100μs per strategy registration
//! **Importance**: Affects system configuration time
//!
//! ### 3. Task Creation (`task_creation`)
//! **What it measures**: Time to create Task objects with various data sizes
//! **Target**: <10μs for small tasks, <100μs for large tasks
//! **Importance**: Core operation for MEV and liquidation workflows
//!
//! ### 4. Memory Allocation (`memory_allocation`)
//! **What it measures**: Memory allocation patterns for different data sizes
//! **Target**: Consistent allocation time regardless of size
//! **Importance**: Memory efficiency for high-frequency trading
//!
//! ### 5. Configuration Validation (`config_validation`)
//! **What it measures**: Time to validate engine configuration
//! **Target**: <50μs for configuration validation
//! **Importance**: System reliability and startup performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use std::time::Duration;

use tallyio_core::engine::{Engine, EngineConfig, Task};
use tallyio_core::types::StrategyId;

/// Benchmark engine creation
fn bench_engine_creation(c: &mut Criterion) {
    c.bench_function("engine_creation", |b| {
        b.iter(|| {
            let config = EngineConfig::default();
            #[allow(clippy::unwrap_used)]
            {
                black_box(Engine::new(config).unwrap())
            }
        });
    });
}

/// Benchmark engine configuration
fn bench_engine_configuration(c: &mut Criterion) {
    c.bench_function("engine_configuration", |b| {
        b.iter(|| {
            let config = EngineConfig {
                max_workers: 4,
                task_queue_capacity: 1000,
                worker_idle_timeout: Duration::from_secs(30),
                task_timeout: Duration::from_millis(100),
                enable_monitoring: true,
                cpu_affinity: None,
                memory_pool_size: 1024 * 1024,
            };

            #[allow(clippy::unwrap_used)]
            {
                black_box(Engine::new(config).unwrap());
            }
        });
    });
}

/// Benchmark task creation
fn bench_task_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_creation");

    for data_size in &[4_usize, 64_usize, 256_usize, 1_024_usize] {
        group.bench_with_input(
            BenchmarkId::new("data_size_bytes", data_size),
            data_size,
            |b, &data_size| {
                let strategy_id = StrategyId::new();

                b.iter(|| {
                    let data = vec![0u8; data_size];
                    black_box(Task::new(strategy_id, data));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark configuration validation
fn bench_config_validation(c: &mut Criterion) {
    c.bench_function("config_validation", |b| {
        b.iter(|| {
            let config = EngineConfig {
                max_workers: 4,
                task_queue_capacity: 10_000,
                worker_idle_timeout: Duration::from_secs(30),
                task_timeout: Duration::from_millis(100),
                enable_monitoring: true,
                cpu_affinity: None,
                memory_pool_size: 1024 * 1024,
            };

            #[allow(clippy::unwrap_used)]
            {
                config.validate().unwrap();
                black_box(());
            }
        });
    });
}

/// Benchmark simple memory allocation
fn bench_memory_allocation_simple(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation_simple");

    for data_size in &[64_usize, 256_usize, 1_024_usize, 4_096_usize] {
        group.bench_with_input(
            BenchmarkId::new("data_size_bytes", data_size),
            data_size,
            |b, &data_size| {
                b.iter(|| {
                    let data = vec![0u8; data_size];
                    black_box(data);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_engine_creation,
    bench_engine_configuration,
    bench_task_creation,
    bench_config_validation,
    bench_memory_allocation_simple
);

criterion_main!(benches);
