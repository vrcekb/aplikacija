//! Load Balancer Performance Benchmarks
//! 
//! Ultra-low latency load balancer benchmarks for TallyIO financial application.
//! Target: <1ms latency for critical load balancing decisions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use tokio::runtime::Runtime;
use tallyio_network::prelude::*;

/// Benchmark configuration for load balancer operations
struct BenchConfig {
    pub endpoint_count: usize,
    pub requests_per_second: usize,
    pub health_check_interval: Duration,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            endpoint_count: 10,
            requests_per_second: 10_000,
            health_check_interval: Duration::from_millis(100),
        }
    }
}

/// Setup load balancer for benchmarks
fn setup_load_balancer(endpoint_count: usize) -> LoadBalancer {
    let endpoints = (0..endpoint_count)
        .map(|i| Endpoint {
            url: format!("https://api{}.example.com", i),
            socket_addr: None,
            priority: 100,
            weight: 1,
            health_check: None,
        })
        .collect();

    LoadBalancer::new(LoadBalancingStrategy::RoundRobin, endpoints)
}

/// Benchmark endpoint selection latency
fn bench_endpoint_selection(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    
    let endpoint_counts = vec![5, 10, 50, 100];
    
    for count in endpoint_counts {
        c.bench_with_input(
            BenchmarkId::new("load_balancer_selection", count),
            &count,
            |b, &count| {
                let load_balancer = setup_load_balancer(count);
                
                b.iter(|| {
                    rt.block_on(async {
                        let start = std::time::Instant::now();
                        
                        let endpoint = load_balancer.select_endpoint().await;
                        
                        let elapsed = start.elapsed();
                        
                        black_box((endpoint, elapsed))
                    })
                });
            },
        );
    }
}

/// Benchmark round-robin strategy
fn bench_round_robin_strategy(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let load_balancer = setup_load_balancer(10);
    
    c.bench_function("load_balancer_round_robin", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = std::time::Instant::now();
                
                // Select multiple endpoints to test round-robin
                let mut selections = Vec::with_capacity(100);
                for _ in 0..100 {
                    if let Ok(endpoint) = load_balancer.select_endpoint().await {
                        selections.push(endpoint.url);
                    }
                }
                
                let elapsed = start.elapsed();
                
                // Verify round-robin distribution
                let unique_count = selections.iter().collect::<std::collections::HashSet<_>>().len();

                black_box((selections, unique_count, elapsed))
            })
        });
    });
}

/// Benchmark weighted load balancing
fn bench_weighted_strategy(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    
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

                let load_balancer = LoadBalancer::new(LoadBalancingStrategy::WeightedRoundRobin, endpoints);
                
                let start = std::time::Instant::now();
                
                // Test weighted selection
                let mut high_weight_count = 0;
                for _ in 0..1000 {
                    if let Ok(endpoint) = load_balancer.select_endpoint().await {
                        if endpoint.url == "https://api1.example.com" {
                            high_weight_count += 1;
                        }
                    }
                }
                
                let elapsed = start.elapsed();
                
                black_box((high_weight_count, elapsed))
            })
        });
    });
}

/// Benchmark health check performance
fn bench_health_checks(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let load_balancer = setup_load_balancer(20);
    
    c.bench_function("load_balancer_health_checks", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = std::time::Instant::now();
                
                // Check if load balancer is healthy
                let _ = load_balancer.is_healthy();
                
                let elapsed = start.elapsed();
                
                black_box(elapsed)
            })
        });
    });
}

/// Benchmark failover performance
fn bench_failover_performance(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    
    c.bench_function("load_balancer_failover", |b| {
        b.iter(|| {
            rt.block_on(async {
                let load_balancer = setup_load_balancer(5);
                
                let start = std::time::Instant::now();
                
                // Select endpoint (failover logic is internal)
                let endpoint = load_balancer.select_endpoint().await;
                
                let elapsed = start.elapsed();
                
                black_box((endpoint, elapsed))
            })
        });
    });
}

/// Benchmark concurrent load balancing
fn bench_concurrent_load_balancing(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let load_balancer = setup_load_balancer(10);
    
    let concurrency_levels = vec![10, 100, 1000];
    
    for concurrency in concurrency_levels {
        c.bench_with_input(
            BenchmarkId::new("load_balancer_concurrent", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::with_capacity(concurrency);
                        
                        let start = std::time::Instant::now();
                        
                        for _ in 0..concurrency {
                            let load_balancer = load_balancer.clone();
                            let handle: tokio::task::JoinHandle<NetworkResult<Endpoint>> = tokio::spawn(async move {
                                load_balancer.select_endpoint().await
                            });
                            handles.push(handle);
                        }
                        
                        let results = futures::future::join_all(handles).await;
                        let elapsed = start.elapsed();
                        
                        // Count successful selections
                        let successful_selections = results.iter()
                            .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
                            .count();
                        
                        black_box((results, successful_selections, elapsed))
                    })
                });
            },
        );
    }
}

/// Benchmark adaptive load balancing
fn bench_adaptive_strategy(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    
    c.bench_function("load_balancer_adaptive", |b| {
        b.iter(|| {
            rt.block_on(async {
                let endpoints = (0..5)
                    .map(|i| Endpoint {
                        url: format!("https://api{}.example.com", i),
                        socket_addr: None,
                        priority: 100,
                        weight: 1,
                        health_check: None,
                    })
                    .collect();

                let load_balancer = LoadBalancer::new(LoadBalancingStrategy::LeastConnections, endpoints);
                
                let start = std::time::Instant::now();
                
                // Test adaptive selection
                let endpoint = load_balancer.select_endpoint().await;
                
                let elapsed = start.elapsed();
                
                black_box((endpoint, elapsed))
            })
        });
    });
}

criterion_group!(
    benches,
    bench_endpoint_selection,
    bench_round_robin_strategy,
    bench_weighted_strategy,
    bench_health_checks,
    bench_failover_performance,
    bench_concurrent_load_balancing,
    bench_adaptive_strategy
);

criterion_main!(benches);
