//! Storage performance benchmarks
//!
//! Benchmarks for different storage operations to ensure performance requirements.

#![allow(clippy::unwrap_used)] // Benchmarks are allowed to use unwrap for simplicity
#![allow(clippy::expect_used)] // Benchmarks are allowed to use expect for simplicity
#![allow(clippy::panic)] // Benchmarks are allowed to panic
#![allow(clippy::unreadable_literal)] // Benchmark data can have unreadable literals
#![allow(clippy::default_numeric_fallback)] // Benchmark data can have default numeric types
#![allow(clippy::uninlined_format_args)] // Benchmark formatting can be verbose
#![allow(clippy::unused_async)] // Benchmark helper functions can be async
#![allow(clippy::unit_arg)] // Benchmarks can pass unit values
#![allow(clippy::explicit_iter_loop)] // Benchmarks can use explicit iteration

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use tokio::runtime::Runtime;

use tallyio_data_storage::{
    config::HotStorageConfig, storage::UltraHotStorage, types::Opportunity, DataStorage,
    DataStorageConfig,
};

/// Benchmark hot storage operations
fn bench_hot_storage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let storage = rt.block_on(async {
        let config = create_bench_config().await;
        DataStorage::new(config).await.unwrap()
    });

    let mut group = c.benchmark_group("hot_storage");
    group.measurement_time(Duration::from_secs(10));

    // Benchmark opportunity storage
    group.bench_function("store_opportunity", |b| {
        b.iter(|| {
            let opportunity = create_test_opportunity();
            black_box(storage.store_opportunity_fast(&opportunity).unwrap());
        });
    });

    // Benchmark opportunity retrieval
    let opportunity = create_test_opportunity();
    storage.store_opportunity_fast(&opportunity).unwrap();

    group.bench_function("get_opportunity", |b| {
        b.iter(|| {
            let result = black_box(storage.get_opportunity_fast(&opportunity.id).unwrap());
            assert!(result.is_some());
        });
    });

    group.finish();
}

/// Benchmark concurrent operations
fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let storage = rt.block_on(async {
        let config = create_bench_config().await;
        DataStorage::new(config).await.unwrap()
    });

    let mut group = c.benchmark_group("concurrent_operations");
    group.measurement_time(Duration::from_secs(15));

    for concurrency in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_store", concurrency),
            concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::new();

                        for _ in 0..concurrency {
                            let storage_clone = storage.clone();
                            let handle = tokio::task::spawn_blocking(move || {
                                let opportunity = create_test_opportunity();
                                storage_clone.store_opportunity_fast(&opportunity).unwrap();
                            });
                            handles.push(handle);
                        }

                        for handle in handles {
                            handle.await.unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark throughput
fn bench_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let storage = rt.block_on(async {
        let config = create_bench_config().await;
        DataStorage::new(config).await.unwrap()
    });

    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(20));

    for batch_size in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_store", batch_size),
            batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    rt.block_on(async {
                        for _ in 0..batch_size {
                            let opportunity = create_test_opportunity();
                            storage.store_opportunity_fast(&opportunity).unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

/// Benchmark latency requirements
fn bench_latency_requirements(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let storage = rt.block_on(async {
        let config = create_bench_config().await;
        DataStorage::new(config).await.unwrap()
    });

    let mut group = c.benchmark_group("latency_requirements");
    group.measurement_time(Duration::from_secs(10));

    // Set strict latency requirements
    group.significance_level(0.1).confidence_level(0.95);

    group.bench_function("sub_millisecond_store", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let opportunity = create_test_opportunity();
            storage.store_opportunity_fast(&opportunity).unwrap();
            let duration = start.elapsed();

            // Assert <1ms requirement
            assert!(
                duration.as_millis() < 1,
                "Store operation too slow: {duration:?}"
            );
        });
    });

    group.bench_function("sub_millisecond_get", |b| {
        // Pre-store an opportunity
        let opportunity = create_test_opportunity();
        storage.store_opportunity_fast(&opportunity).unwrap();

        b.iter(|| {
            let start = std::time::Instant::now();
            let result = storage.get_opportunity_fast(&opportunity.id).unwrap();
            let duration = start.elapsed();

            assert!(result.is_some());
            // Assert <1ms requirement
            assert!(
                duration.as_millis() < 1,
                "Get operation too slow: {duration:?}"
            );
        });
    });

    group.finish();
}

/// Benchmark memory usage
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let storage = rt.block_on(async {
        let config = create_bench_config().await;
        DataStorage::new(config).await.unwrap()
    });

    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(15));

    group.bench_function("memory_efficient_operations", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Perform operations that should have minimal memory allocations
                for _ in 0..1000 {
                    let opportunity = create_test_opportunity();
                    storage.store_opportunity_fast(&opportunity).unwrap();
                    let _result = storage.get_opportunity_fast(&opportunity.id).unwrap();
                }
            });
        });
    });

    group.finish();
}

// Helper functions

async fn create_bench_config() -> DataStorageConfig {
    let mut config = DataStorageConfig::default();

    // Use temporary directories for benchmarking
    let temp_dir = tempfile::tempdir().unwrap();
    // Use in-memory storage for ultra-low latency benchmarking
    config.hot_storage.use_memory_storage = true;
    config.hot_storage.database_path = None;
    config.cold_storage.storage_path = temp_dir.path().join("cold");
    config.cold_storage.enable_encryption = false; // Disable for benchmarking

    // Optimize for performance
    config.hot_storage.cache_size_bytes = 256 * 1024 * 1024; // 256MB cache
    config.cache.redis_url = None; // Use memory cache only
    config.cache.enable_memory_cache = true;
    config.cache.memory_cache_size_bytes = 128 * 1024 * 1024; // 128MB

    config
}

fn create_test_opportunity() -> Opportunity {
    Opportunity::new(
        "arbitrage".to_string(),
        1,
        "1.5".to_string(),
        "0.1".to_string(),
        "1.4".to_string(),
        0.95,
    )
}

/// Benchmark ultra-optimized storage
fn bench_ultra_storage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let ultra_storage = rt.block_on(async {
        let config = create_ultra_bench_config().await;
        UltraHotStorage::new(&config).unwrap()
    });

    let mut group = c.benchmark_group("ultra_storage");
    group.measurement_time(Duration::from_secs(10));

    // Benchmark ultra-fast opportunity storage
    group.bench_function("ultra_store_opportunity", |b| {
        b.iter(|| {
            let opportunity = create_test_opportunity();
            black_box(ultra_storage.store_opportunity(&opportunity).unwrap());
        });
    });

    // Benchmark ultra-fast opportunity retrieval
    let opportunity = create_test_opportunity();
    ultra_storage.store_opportunity(&opportunity).unwrap();

    group.bench_function("ultra_get_opportunity", |b| {
        b.iter(|| {
            let result = black_box(ultra_storage.get_opportunity(&opportunity.id).unwrap());
            assert!(result.is_some());
        });
    });

    group.finish();
}

/// Benchmark ultra-latency requirements (target: <100μs)
fn bench_ultra_latency_requirements(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let ultra_storage = rt.block_on(async {
        let config = create_ultra_bench_config().await;
        UltraHotStorage::new(&config).unwrap()
    });

    let mut group = c.benchmark_group("ultra_latency_requirements");
    group.measurement_time(Duration::from_secs(10));

    // Set ultra-strict latency requirements
    group.significance_level(0.1).confidence_level(0.95);

    group.bench_function("sub_100us_store", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let opportunity = create_test_opportunity();
            ultra_storage.store_opportunity(&opportunity).unwrap();
            let duration = start.elapsed();

            // Assert <100μs requirement
            assert!(
                duration.as_micros() < 100,
                "Ultra store operation too slow: {duration:?}"
            );
        });
    });

    group.bench_function("sub_50us_get", |b| {
        // Pre-store an opportunity
        let opportunity = create_test_opportunity();
        ultra_storage.store_opportunity(&opportunity).unwrap();

        b.iter(|| {
            let start = std::time::Instant::now();
            let result = ultra_storage.get_opportunity(&opportunity.id).unwrap();
            let duration = start.elapsed();

            assert!(result.is_some());
            // Assert <50μs requirement for reads
            assert!(
                duration.as_micros() < 50,
                "Ultra get operation too slow: {duration:?}"
            );
        });
    });

    group.finish();
}

async fn create_ultra_bench_config() -> HotStorageConfig {
    HotStorageConfig {
        use_memory_storage: true,
        database_path: None,
        cache_size_bytes: 512 * 1024 * 1024, // 512MB cache
        ..Default::default()
    }
}

criterion_group!(
    benches,
    bench_hot_storage,
    bench_concurrent_operations,
    bench_throughput,
    bench_latency_requirements,
    bench_memory_usage,
    bench_ultra_storage,
    bench_ultra_latency_requirements
);
criterion_main!(benches);
