//! Load Balancer Performance Benchmarks
//! 
//! Ultra-low latency load balancer benchmarks for `TallyIO` financial application.
//! Target: <1ms latency for critical real-time operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use tallyio_network::prelude::*;

/// Setup load balancer for benchmarks
fn setup_load_balancer(endpoint_count: usize) -> NetworkResult<LoadBalancer> {
    let endpoints = (0_i32..endpoint_count.try_into().unwrap_or(i32::MAX))
        .map(|i| Endpoint {
            url: format!("https://api{i}.example.com"),
            socket_addr: None,
            priority: 100,
            weight: 1,
            health_check: None,
        })
        .collect();

    LoadBalancer::new(
        LoadBalancingStrategy::RoundRobin,
        endpoints,
        HealthCheckConfig::default()
    )
}

/// Benchmark endpoint selection latency
fn bench_endpoint_selection(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    let endpoint_counts = vec![5, 10, 20, 50];
    
    for count in endpoint_counts {
        c.bench_with_input(
            BenchmarkId::new("load_balancer_selection", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let Ok(load_balancer) = setup_load_balancer(count) else { return };
                        let start = std::time::Instant::now();

                        let result = load_balancer.select_endpoint().await;
                        let elapsed = start.elapsed();

                        let _ = black_box((result, elapsed));
                    });
                });
            },
        );
    }
}

/// Benchmark round-robin distribution
fn bench_round_robin_distribution(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    c.bench_function("load_balancer_round_robin", |b| {
        b.iter(|| {
            rt.block_on(async {
                let Ok(load_balancer) = setup_load_balancer(10) else { return };
                let mut selections = Vec::with_capacity(100);

                let start = std::time::Instant::now();

                for _ in 0_i32..100_i32 {
                    if let Ok(endpoint) = load_balancer.select_endpoint().await {
                        selections.push(endpoint.url);
                    }
                }

                let elapsed = start.elapsed();

                // Verify round-robin distribution
                let unique_count = selections.iter().collect::<std::collections::HashSet<_>>().len();

                let _ = black_box((selections, unique_count, elapsed));
            });
        });
    });
}

/// Benchmark weighted round-robin
fn bench_weighted_round_robin(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    c.bench_function("load_balancer_weighted", |b| {
        b.iter(|| {
            rt.block_on(async {
                let endpoints = vec![
                    Endpoint {
                        url: "https://api1.example.com".to_string(),
                        socket_addr: None,
                        priority: 100,
                        weight: 10,
                        health_check: None,
                    },
                    Endpoint {
                        url: "https://api2.example.com".to_string(),
                        socket_addr: None,
                        priority: 100,
                        weight: 1,
                        health_check: None,
                    },
                ];

                let Ok(load_balancer) = LoadBalancer::new(
                    LoadBalancingStrategy::WeightedRoundRobin,
                    endpoints,
                    HealthCheckConfig::default()
                ) else { return };

                let mut high_weight_count = 0_i32;
                for _ in 0_i32..1_000_i32 {
                    if let Ok(endpoint) = load_balancer.select_endpoint().await {
                        if endpoint.url == "https://api1.example.com" {
                            high_weight_count += 1_i32;
                        }
                    }
                }

                let _ = black_box(high_weight_count);
            });
        });
    });
}

/// Benchmark health check performance
fn bench_health_check(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    c.bench_function("load_balancer_health_check", |b| {
        b.iter(|| {
            rt.block_on(async {
                let Ok(load_balancer) = setup_load_balancer(20) else { return };

                let start = std::time::Instant::now();

                // Check if load balancer is healthy
                let is_healthy = load_balancer.is_healthy();

                let elapsed = start.elapsed();

                let _ = black_box((is_healthy, elapsed));
            });
        });
    });
}

/// Benchmark concurrent endpoint selection
fn bench_concurrent_selection(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    let concurrency_levels = vec![10, 50, 100];
    
    for concurrency in concurrency_levels {
        c.bench_with_input(
            BenchmarkId::new("load_balancer_concurrent", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async {
                        let Ok(load_balancer) = setup_load_balancer(5) else { return };
                        let mut handles = Vec::with_capacity(concurrency);

                        let start = std::time::Instant::now();

                        for _ in 0_i32..concurrency.try_into().unwrap_or(i32::MAX) {
                            let load_balancer = load_balancer.clone();
                            let handle: tokio::task::JoinHandle<NetworkResult<Endpoint>> = tokio::spawn(async move {
                                load_balancer.select_endpoint().await
                            });
                            handles.push(handle);
                        }

                        let results = futures::future::join_all(handles).await;
                        let elapsed = start.elapsed();

                        // Count successful selections
                        let successful_count = results
                            .iter()
                            .filter(|r| r.is_ok() && r.as_ref().is_ok_and(std::result::Result::is_ok))
                            .count();

                        let _ = black_box((successful_count, elapsed));
                    });
                });
            },
        );
    }
}

/// Benchmark least connections strategy
fn bench_least_connections(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    c.bench_function("load_balancer_least_connections", |b| {
        b.iter(|| {
            rt.block_on(async {
                let endpoints = (0_i32..5_i32)
                    .map(|i| Endpoint {
                        url: format!("https://api{i}.example.com"),
                        socket_addr: None,
                        priority: 100,
                        weight: 1,
                        health_check: None,
                    })
                    .collect();

                let Ok(load_balancer) = LoadBalancer::new(
                    LoadBalancingStrategy::LeastConnections,
                    endpoints,
                    HealthCheckConfig::default()
                ) else { return };

                let start = std::time::Instant::now();

                // Simulate multiple selections
                let mut selections = Vec::with_capacity(100);
                for _ in 0_i32..100_i32 {
                    if let Ok(endpoint) = load_balancer.select_endpoint().await {
                        selections.push(endpoint.url);
                    }
                }

                let elapsed = start.elapsed();

                let _ = black_box((selections, elapsed));
            });
        });
    });
}

criterion_group!(
    benches,
    bench_endpoint_selection,
    bench_round_robin_distribution,
    bench_weighted_round_robin,
    bench_health_check,
    bench_concurrent_selection,
    bench_least_connections
);

criterion_main!(benches);
