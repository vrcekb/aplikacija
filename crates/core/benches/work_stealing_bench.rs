//! Work-Stealing Scheduler Benchmarks
//!
//! Performance benchmarks for work-stealing scheduler implementation.
//! Measures thread scaling, task throughput, and latency characteristics.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::time::Duration;

use tallyio_core::engine::{
    EngineConfig, EngineMetrics, Task, TaskPriority, WorkStealingError, WorkStealingScheduler,
};
use tallyio_core::types::StrategyId;

/// Create test task
fn create_test_task(priority: TaskPriority, data_size: usize) -> Task {
    let strategy_id = StrategyId::new();
    let data = vec![0u8; data_size];

    match priority {
        TaskPriority::Critical => Task::new_critical(strategy_id, data),
        TaskPriority::High => Task::new_high_priority(strategy_id, data),
        _ => Task::new(strategy_id, data),
    }
}

/// Create test scheduler
fn create_test_scheduler(worker_count: usize) -> Result<WorkStealingScheduler, WorkStealingError> {
    let config = Arc::new(EngineConfig::default());
    let metrics = Arc::new(EngineMetrics::default());

    WorkStealingScheduler::new(config, metrics, Some(worker_count))
}

/// Benchmark work-stealing scheduler creation
fn bench_scheduler_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_scheduler_creation");

    for worker_count in &[1, 2, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let config = Arc::new(EngineConfig::default());
                    let metrics = Arc::new(EngineMetrics::default());

                    black_box(WorkStealingScheduler::new(
                        config,
                        metrics,
                        Some(worker_count),
                    ))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark task submission
fn bench_task_submission(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_task_submission");

    for worker_count in &[1, 2, 4, 8] {
        let Ok(mut scheduler) = create_test_scheduler(*worker_count) else {
            return; // Skip benchmark if scheduler creation fails
        };
        if scheduler.start().is_err() {
            return; // Skip benchmark if start fails
        }

        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, _| {
                b.iter(|| {
                    let task = create_test_task(TaskPriority::Normal, 64);
                    black_box(scheduler.submit_task(task))
                });
            },
        );

        let _ = scheduler.stop(); // Ignore stop errors in benchmarks
    }

    group.finish();
}

/// Benchmark task submission with different priorities
fn bench_task_priorities(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_task_priorities");

    let Ok(mut scheduler) = create_test_scheduler(4) else {
        return; // Skip benchmark if scheduler creation fails
    };
    if scheduler.start().is_err() {
        return; // Skip benchmark if start fails
    }

    for priority in &[
        TaskPriority::Low,
        TaskPriority::Normal,
        TaskPriority::High,
        TaskPriority::Critical,
    ] {
        group.bench_with_input(
            BenchmarkId::new("priority", format!("{priority:?}")),
            priority,
            |b, &priority| {
                b.iter(|| {
                    let task = create_test_task(priority, 64);
                    black_box(scheduler.submit_task(task))
                });
            },
        );
    }

    let _ = scheduler.stop(); // Ignore stop errors in benchmarks
    group.finish();
}

/// Benchmark task submission with different data sizes
fn bench_task_data_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_task_data_sizes");

    let Ok(mut scheduler) = create_test_scheduler(4) else {
        return; // Skip benchmark if scheduler creation fails
    };
    if scheduler.start().is_err() {
        return; // Skip benchmark if start fails
    }

    for data_size in &[64, 256, 1024, 4096, 16384] {
        group.throughput(Throughput::Bytes(*data_size as u64));
        group.bench_with_input(
            BenchmarkId::new("bytes", data_size),
            data_size,
            |b, &data_size| {
                b.iter(|| {
                    let task = create_test_task(TaskPriority::Normal, data_size);
                    black_box(scheduler.submit_task(task))
                });
            },
        );
    }

    let _ = scheduler.stop(); // Ignore stop errors in benchmarks
    group.finish();
}

/// Benchmark concurrent task submission
fn bench_concurrent_submission(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_concurrent_submission");

    for worker_count in &[2, 4, 8] {
        let Ok(mut scheduler) = create_test_scheduler(*worker_count) else {
            return; // Skip benchmark if scheduler creation fails
        };
        if scheduler.start().is_err() {
            return; // Skip benchmark if start fails
        }

        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, _| {
                b.iter(|| {
                    // Submit multiple tasks concurrently
                    let tasks: Vec<_> = (0_i32..100_i32)
                        .map(|_| create_test_task(TaskPriority::Normal, 64))
                        .collect();

                    for task in tasks {
                        let _ = black_box(scheduler.submit_task(task));
                    }
                });
            },
        );

        let _ = scheduler.stop(); // Ignore stop errors in benchmarks
    }

    group.finish();
}

/// Benchmark scheduler throughput
fn bench_scheduler_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_throughput");

    for worker_count in &[1, 2, 4, 8] {
        let Ok(mut scheduler) = create_test_scheduler(*worker_count) else {
            return; // Skip benchmark if scheduler creation fails
        };
        if scheduler.start().is_err() {
            return; // Skip benchmark if start fails
        }

        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, _| {
                b.iter(|| {
                    for _ in 0_i32..1_000_i32 {
                        let task = create_test_task(TaskPriority::Normal, 64);
                        let _ = black_box(scheduler.submit_task(task));
                    }
                });
            },
        );

        let _ = scheduler.stop(); // Ignore stop errors in benchmarks
    }

    group.finish();
}

/// Benchmark scheduler startup/shutdown
fn bench_scheduler_lifecycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_lifecycle");

    for worker_count in &[1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    if let Ok(mut scheduler) = create_test_scheduler(worker_count) {
                        let _ = scheduler.start();
                        black_box(());
                        let _ = scheduler.stop();
                        black_box(());
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark work stealing efficiency
fn bench_work_stealing_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_efficiency");

    // Test with uneven load distribution
    for worker_count in &[2, 4, 8] {
        let Ok(mut scheduler) = create_test_scheduler(*worker_count) else {
            return; // Skip benchmark if scheduler creation fails
        };
        if scheduler.start().is_err() {
            return; // Skip benchmark if start fails
        }

        group.bench_with_input(
            BenchmarkId::new("workers", worker_count),
            worker_count,
            |b, _| {
                b.iter(|| {
                    // Create burst of tasks to test work stealing
                    for _ in 0_i32..50_i32 {
                        let task = create_test_task(TaskPriority::High, 128);
                        let _ = black_box(scheduler.submit_task(task));
                    }

                    // Brief pause to allow work stealing
                    std::thread::sleep(Duration::from_micros(100));
                });
            },
        );

        let _ = scheduler.stop(); // Ignore stop errors in benchmarks
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_memory");

    let Ok(mut scheduler) = create_test_scheduler(4) else {
        return; // Skip benchmark if scheduler creation fails
    };
    if scheduler.start().is_err() {
        return; // Skip benchmark if start fails
    }

    // Test different memory access patterns
    for pattern in &["sequential", "random", "burst"] {
        group.bench_with_input(
            BenchmarkId::new("pattern", pattern),
            pattern,
            |b, &pattern| {
                b.iter(|| {
                    match pattern {
                        "sequential" => {
                            for i in 0_i32..100_i32 {
                                let task = create_test_task(
                                    TaskPriority::Normal,
                                    64 + usize::try_from(i).unwrap_or(0),
                                );
                                let _ = black_box(scheduler.submit_task(task));
                            }
                        }
                        "random" => {
                            for _ in 0_i32..100_i32 {
                                let size = 64 + (std::ptr::addr_of!(scheduler) as usize % 1000);
                                let task = create_test_task(TaskPriority::Normal, size);
                                let _ = black_box(scheduler.submit_task(task));
                            }
                        }
                        "burst" => {
                            // Submit tasks in bursts
                            for _ in 0_i32..10_i32 {
                                for _ in 0_i32..10_i32 {
                                    let task = create_test_task(TaskPriority::High, 64);
                                    let _ = black_box(scheduler.submit_task(task));
                                }
                                std::thread::sleep(Duration::from_micros(10));
                            }
                        }
                        _ => {}
                    }
                });
            },
        );
    }

    let _ = scheduler.stop(); // Ignore stop errors in benchmarks
    group.finish();
}

criterion_group!(
    work_stealing_benches,
    bench_scheduler_creation,
    bench_task_submission,
    bench_task_priorities,
    bench_task_data_sizes,
    bench_concurrent_submission,
    bench_scheduler_throughput,
    bench_scheduler_lifecycle,
    bench_work_stealing_efficiency,
    bench_memory_patterns
);

criterion_main!(work_stealing_benches);
