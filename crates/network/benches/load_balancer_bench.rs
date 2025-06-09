//! Load Balancer Performance Benchmarks
//! 
//! Ultra-low latency load balancer benchmarks for TallyIO financial application.
//! Target: <1ms latency for critical load balancing decisions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use tokio::runtime::Runtime;
use tallyio_network::{
    LoadBalancer, NetworkConfig, NetworkError,
    types::{Endpoint, LoadBalancingStrategy, HealthStatus},
};

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
fn setup_load_balancer(endpoint_count: usize) -> Result<LoadBalancer, NetworkError> {
    let config = NetworkConfig {
        max_connections: 1000,
        connection_timeout: Duration::from_millis(100),
        request_timeout: Duration::from_millis(1),
        load_balancer_strategy: LoadBalancingStrategy::RoundRobin,
        health_check_interval: Duration::from_millis(100),
        ..Default::default()
    };
    
    let endpoints = (0..endpoint_count)
        .map(|i| Endpoint {
            id: format!("endpoint_{}", i),
            url: format!("https://api{}.example.com", i),
            weight: 1,
            health_status: HealthStatus::Healthy,
            response_time: Duration::from_millis(10),
            error_rate: 0.0,
        })
        .collect();
    
    LoadBalancer::new(config, endpoints)
}

/// Benchmark endpoint selection latency
fn bench_endpoint_selection(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let endpoint_counts = vec![5, 10, 50, 100];
    
    for count in endpoint_counts {
        c.bench_with_input(
            BenchmarkId::new("load_balancer_selection", count),
            &count,
            |b, &count| {
                let load_balancer = setup_load_balancer(count).unwrap();
                
                b.to_async(&rt).iter(|| async {
                    let start = std::time::Instant::now();
                    
                    let endpoint = load_balancer.select_endpoint().await;
                    
                    let elapsed = start.elapsed();
                    
                    // Assert <1ms selection latency for financial application
                    assert!(elapsed.as_nanos() < 1_000_000, "Selection latency violation: {:?}", elapsed);
                    
                    black_box((endpoint, elapsed))
                });
            },
        );
    }
}

/// Benchmark round-robin strategy
fn bench_round_robin_strategy(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let load_balancer = setup_load_balancer(10).unwrap();
    
    c.bench_function("load_balancer_round_robin", |b| {
        b.to_async(&rt).iter(|| async {
            let start = std::time::Instant::now();
            
            // Select multiple endpoints to test round-robin
            let mut selections = Vec::with_capacity(100);
            for _ in 0..100 {
                let endpoint = load_balancer.select_endpoint().await.unwrap();
                selections.push(endpoint.id);
            }
            
            let elapsed = start.elapsed();
            
            // Verify round-robin distribution
            let unique_selections: std::collections::HashSet<_> = selections.iter().collect();
            assert!(unique_selections.len() >= 5, "Poor round-robin distribution");
            
            black_box((selections, elapsed))
        });
    });
}

/// Benchmark weighted load balancing
fn bench_weighted_strategy(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("load_balancer_weighted", |b| {
        b.to_async(&rt).iter(|| async {
            let config = NetworkConfig {
                load_balancer_strategy: LoadBalancingStrategy::Weighted,
                ..Default::default()
            };
            
            let endpoints = vec![
                Endpoint {
                    id: "high_weight".to_string(),
                    url: "https://api1.example.com".to_string(),
                    weight: 10,
                    health_status: HealthStatus::Healthy,
                    response_time: Duration::from_millis(5),
                    error_rate: 0.0,
                },
                Endpoint {
                    id: "low_weight".to_string(),
                    url: "https://api2.example.com".to_string(),
                    weight: 1,
                    health_status: HealthStatus::Healthy,
                    response_time: Duration::from_millis(20),
                    error_rate: 0.0,
                },
            ];
            
            let load_balancer = LoadBalancer::new(config, endpoints).unwrap();
            
            let start = std::time::Instant::now();
            
            // Test weighted selection
            let mut high_weight_count = 0;
            for _ in 0..1000 {
                let endpoint = load_balancer.select_endpoint().await.unwrap();
                if endpoint.id == "high_weight" {
                    high_weight_count += 1;
                }
            }
            
            let elapsed = start.elapsed();
            
            // Verify weighted distribution (should be ~90% high weight)
            assert!(high_weight_count > 800, "Poor weighted distribution: {}", high_weight_count);
            
            black_box((high_weight_count, elapsed))
        });
    });
}

/// Benchmark health check performance
fn bench_health_checks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let load_balancer = setup_load_balancer(20).unwrap();
    
    c.bench_function("load_balancer_health_checks", |b| {
        b.to_async(&rt).iter(|| async {
            let start = std::time::Instant::now();
            
            // Trigger health checks for all endpoints
            load_balancer.perform_health_checks().await.unwrap();
            
            let elapsed = start.elapsed();
            
            // Assert fast health checks for financial application
            assert!(elapsed.as_millis() < 10, "Health check latency violation: {:?}", elapsed);
            
            black_box(elapsed)
        });
    });
}

/// Benchmark failover performance
fn bench_failover_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("load_balancer_failover", |b| {
        b.to_async(&rt).iter(|| async {
            let load_balancer = setup_load_balancer(5).unwrap();
            
            // Mark first endpoint as unhealthy
            load_balancer.mark_endpoint_unhealthy("endpoint_0").await.unwrap();
            
            let start = std::time::Instant::now();
            
            // Select endpoint (should skip unhealthy one)
            let endpoint = load_balancer.select_endpoint().await.unwrap();
            
            let elapsed = start.elapsed();
            
            // Verify failover worked
            assert_ne!(endpoint.id, "endpoint_0", "Failover failed");
            
            // Assert fast failover for financial application
            assert!(elapsed.as_nanos() < 1_000_000, "Failover latency violation: {:?}", elapsed);
            
            black_box((endpoint, elapsed))
        });
    });
}

/// Benchmark concurrent load balancing
fn bench_concurrent_load_balancing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let load_balancer = setup_load_balancer(10).unwrap();
    
    let concurrency_levels = vec![10, 100, 1000];
    
    for concurrency in concurrency_levels {
        c.bench_with_input(
            BenchmarkId::new("load_balancer_concurrent", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let mut handles = Vec::with_capacity(concurrency);
                    
                    let start = std::time::Instant::now();
                    
                    for _ in 0..concurrency {
                        let load_balancer = load_balancer.clone();
                        let handle = tokio::spawn(async move {
                            load_balancer.select_endpoint().await
                        });
                        handles.push(handle);
                    }
                    
                    let results = futures::future::join_all(handles).await;
                    let elapsed = start.elapsed();
                    
                    // Verify all selections succeeded
                    let successful_selections = results.iter()
                        .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
                        .count();
                    
                    assert_eq!(successful_selections, concurrency, "Some selections failed");
                    
                    // Assert reasonable concurrent performance
                    let avg_latency = elapsed.as_nanos() / concurrency as u128;
                    assert!(avg_latency < 1_000_000, "Concurrent latency violation: {} ns", avg_latency);
                    
                    black_box((results, elapsed))
                });
            },
        );
    }
}

/// Benchmark adaptive load balancing
fn bench_adaptive_strategy(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("load_balancer_adaptive", |b| {
        b.to_async(&rt).iter(|| async {
            let config = NetworkConfig {
                load_balancer_strategy: LoadBalancingStrategy::LeastConnections,
                ..Default::default()
            };
            
            let endpoints = (0..5)
                .map(|i| Endpoint {
                    id: format!("endpoint_{}", i),
                    url: format!("https://api{}.example.com", i),
                    weight: 1,
                    health_status: HealthStatus::Healthy,
                    response_time: Duration::from_millis(i * 5 + 5),
                    error_rate: i as f64 * 0.01,
                })
                .collect();
            
            let load_balancer = LoadBalancer::new(config, endpoints).unwrap();
            
            let start = std::time::Instant::now();
            
            // Test adaptive selection based on response times
            let endpoint = load_balancer.select_endpoint().await.unwrap();
            
            let elapsed = start.elapsed();
            
            // Should select endpoint with best response time
            assert_eq!(endpoint.id, "endpoint_0", "Adaptive selection failed");
            
            // Assert fast adaptive selection
            assert!(elapsed.as_nanos() < 1_000_000, "Adaptive latency violation: {:?}", elapsed);
            
            black_box((endpoint, elapsed))
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
