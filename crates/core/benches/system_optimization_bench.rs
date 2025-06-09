//! System Optimization Benchmarks for `TallyIO` Phase 2
//!
//! Comprehensive benchmarks for NUMA-aware allocation, CPU affinity,
//! and jemalloc tuning optimizations in financial trading applications.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Instant;

use tallyio_core::optimization::{
    init_jemalloc_for_trading, init_numa, numa_alloc, numa_dealloc, CpuAffinityManager,
    CriticalThreadType, JemallocConfig, JemallocTuner,
};

/// Benchmark NUMA-aware memory allocation
fn bench_numa_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("numa_allocation");

    // Initialize NUMA allocator
    if init_numa().is_err() {
        eprintln!("NUMA not available - skipping NUMA benchmarks");
        return;
    }

    for size in [1024_usize, 4096_usize, 16384_usize, 65536_usize] {
        group.bench_with_input(
            BenchmarkId::new("numa_alloc_dealloc", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    if let Ok(ptr) = numa_alloc(size) {
                        black_box(ptr);
                        numa_dealloc(ptr, size, 0);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("numa_batch_allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut ptrs = Vec::with_capacity(100_usize);

                    // Allocate batch
                    for _ in 0_i32..100_i32 {
                        if let Ok(ptr) = numa_alloc(size) {
                            ptrs.push(ptr);
                        }
                    }

                    // Deallocate batch
                    for ptr in ptrs {
                        numa_dealloc(ptr, size, 0);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CPU affinity management
fn bench_cpu_affinity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_affinity");

    // Create CPU affinity manager
    let cpu_count = u32::try_from(num_cpus::get()).unwrap_or(1);
    let cpu_cores: Vec<u32> = (0_u32..cpu_count).collect();
    let Ok(affinity_manager) = CpuAffinityManager::new(cpu_cores) else {
        eprintln!("CPU affinity not available - skipping affinity benchmarks");
        return;
    };

    group.bench_function("pin_critical_thread", |b| {
        b.iter(|| {
            let start = Instant::now();

            if affinity_manager
                .pin_critical_thread(CriticalThreadType::MevScanner)
                .is_ok()
            {
                black_box(start.elapsed());
            }
        });
    });

    group.bench_function("pin_worker_thread", |b| {
        b.iter(|| {
            let start = Instant::now();

            if affinity_manager.pin_worker_thread(0).is_ok() {
                black_box(start.elapsed());
            }
        });
    });

    group.bench_function("numa_aware_pinning", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Pin to NUMA node 0
            if let Ok(core) = affinity_manager.pin_to_numa_node(0, 0) {
                black_box(core);
                black_box(start.elapsed());
            }
        });
    });

    group.bench_function("topology_queries", |b| {
        b.iter(|| {
            let start = Instant::now();

            let _topology = affinity_manager.get_topology_info();
            let _cores_numa0 = affinity_manager.get_cores_on_numa_node(0);
            let _usage = affinity_manager.get_core_usage(0);

            black_box(start.elapsed());
        });
    });

    group.finish();
}

/// Benchmark jemalloc tuning
fn bench_jemalloc_tuning(c: &mut Criterion) {
    let mut group = c.benchmark_group("jemalloc_tuning");

    group.bench_function("jemalloc_init", |b| {
        b.iter(|| {
            let start = Instant::now();

            if let Ok(tuner) = init_jemalloc_for_trading() {
                black_box(tuner);
                black_box(start.elapsed());
            }
        });
    });

    group.bench_function("config_application", |b| {
        b.iter(|| {
            let config = JemallocConfig::financial_trading();

            if let Ok(tuner) = JemallocTuner::new(config) {
                let start = Instant::now();

                if tuner.apply_config().is_ok() {
                    black_box(start.elapsed());
                }
            }
        });
    });

    group.bench_function("stats_collection", |b| {
        b.iter(|| {
            let config = JemallocConfig::ultra_low_latency();

            if let Ok(tuner) = JemallocTuner::new(config) {
                let start = Instant::now();

                if let Ok(stats) = tuner.get_stats() {
                    black_box(stats);
                    black_box(start.elapsed());
                }
            }
        });
    });

    group.bench_function("auto_tuning", |b| {
        b.iter(|| {
            let config = JemallocConfig::high_throughput();

            if let Ok(mut tuner) = JemallocTuner::new(config) {
                // Record some latency samples
                tuner.record_allocation_latency(500);
                tuner.record_allocation_latency(1000);
                tuner.record_allocation_latency(1500);

                let start = Instant::now();

                if let Ok(changed) = tuner.auto_tune() {
                    black_box(changed);
                    black_box(start.elapsed());
                }
            }
        });
    });

    group.finish();
}

/// Benchmark integrated system optimization
fn bench_integrated_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("integrated_optimization");

    group.bench_function("full_system_init", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Initialize all optimizations
            let numa_init = init_numa().is_ok();

            let cpu_count = u32::try_from(num_cpus::get()).unwrap_or(1);
            let cpu_cores: Vec<u32> = (0_u32..cpu_count).collect();
            let affinity_init = CpuAffinityManager::new(cpu_cores).is_ok();

            let jemalloc_init = init_jemalloc_for_trading().is_ok();

            black_box((numa_init, affinity_init, jemalloc_init));
            black_box(start.elapsed());
        });
    });

    group.bench_function("optimized_allocation_workflow", |b| {
        // Pre-initialize systems
        let _ = init_numa();
        let cpu_count = u32::try_from(num_cpus::get()).unwrap_or(1);
        let cpu_cores: Vec<u32> = (0_u32..cpu_count).collect();
        let affinity_manager = CpuAffinityManager::new(cpu_cores).ok();
        let _jemalloc_tuner = init_jemalloc_for_trading().ok();

        b.iter(|| {
            let start = Instant::now();

            // Pin to optimal core
            if let Some(ref manager) = affinity_manager {
                let _ = manager.pin_to_numa_node(0, 0);
            }

            // NUMA-aware allocation
            if let Ok(ptr) = numa_alloc(4096_usize) {
                black_box(ptr);
                numa_dealloc(ptr, 4096_usize, 0);
            }

            black_box(start.elapsed());
        });
    });

    group.bench_function("latency_under_optimization", |b| {
        // Pre-initialize all optimizations
        let _ = init_numa();
        let cpu_count = u32::try_from(num_cpus::get()).unwrap_or(1);
        let cpu_cores: Vec<u32> = (0_u32..cpu_count).collect();
        let affinity_manager = CpuAffinityManager::new(cpu_cores).ok();
        let mut jemalloc_tuner = init_jemalloc_for_trading().ok();

        b.iter(|| {
            let start = Instant::now();

            // Critical path simulation
            if let Some(ref manager) = affinity_manager {
                let _ = manager.pin_critical_thread(CriticalThreadType::MevScanner);
            }

            // Fast allocation
            if let Ok(ptr) = numa_alloc(1024_usize) {
                black_box(ptr);
                numa_dealloc(ptr, 1024_usize, 0);
            }

            // Record latency for tuning
            let elapsed = start.elapsed();
            if let Some(ref mut tuner) = jemalloc_tuner {
                let latency_ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
                tuner.record_allocation_latency(latency_ns);
            }

            // Validate <1ms requirement
            if elapsed.as_nanos() > 1_000_000_u128 {
                eprintln!("Latency violation: {}ns > 1ms", elapsed.as_nanos());
            }

            black_box(elapsed);
        });
    });

    group.finish();
}

/// Benchmark memory scaling with optimizations
fn bench_memory_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    // Pre-initialize optimizations
    let _ = init_numa();

    for allocation_count in [10_usize, 100_usize, 1000_usize, 10000_usize] {
        group.bench_with_input(
            BenchmarkId::new("numa_scaling", allocation_count),
            &allocation_count,
            |b, &count| {
                b.iter(|| {
                    let start = Instant::now();
                    let mut ptrs = Vec::with_capacity(count);

                    // Allocate many blocks
                    for _ in 0..count {
                        if let Ok(ptr) = numa_alloc(1024_usize) {
                            ptrs.push(ptr);
                        }
                    }

                    // Deallocate all
                    for ptr in ptrs {
                        numa_dealloc(ptr, 1024_usize, 0);
                    }

                    black_box(start.elapsed());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    system_optimization_benches,
    bench_numa_allocation,
    bench_cpu_affinity,
    bench_jemalloc_tuning,
    bench_integrated_optimization,
    bench_memory_scaling
);

criterion_main!(system_optimization_benches);
