//! HTTP Client Performance Benchmarks
//! 
//! Ultra-low latency HTTP benchmarks for TallyIO financial application.
//! Target: <1ms latency for critical operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use tokio::runtime::Runtime;
use tallyio_network::{
    HttpClient, NetworkConfig, NetworkError,
    types::{HttpRequest, HttpResponse, RequestMethod},
};

/// Benchmark configuration for HTTP operations
struct BenchConfig {
    pub concurrent_requests: usize,
    pub request_size: usize,
    pub timeout_ms: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            concurrent_requests: 100,
            request_size: 1024,
            timeout_ms: 1,
        }
    }
}

/// Setup HTTP client for benchmarks
fn setup_http_client() -> Result<HttpClient, NetworkError> {
    let config = NetworkConfig {
        max_connections: 1000,
        connection_timeout: Duration::from_millis(100),
        request_timeout: Duration::from_millis(1),
        keep_alive: true,
        http2_only: true,
        ..Default::default()
    };
    
    HttpClient::new(config)
}

/// Benchmark single HTTP GET request
fn bench_single_get_request(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let client = rt.block_on(async { setup_http_client().unwrap() });
    
    c.bench_function("http_single_get", |b| {
        b.to_async(&rt).iter(|| async {
            let request = HttpRequest {
                method: RequestMethod::Get,
                url: "https://httpbin.org/get".to_string(),
                headers: Default::default(),
                body: None,
                timeout: Duration::from_millis(1),
            };
            
            let result = client.send_request(black_box(request)).await;
            black_box(result)
        });
    });
}

/// Benchmark concurrent HTTP requests
fn bench_concurrent_requests(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let client = rt.block_on(async { setup_http_client().unwrap() });
    
    let configs = vec![
        BenchConfig { concurrent_requests: 10, ..Default::default() },
        BenchConfig { concurrent_requests: 50, ..Default::default() },
        BenchConfig { concurrent_requests: 100, ..Default::default() },
    ];
    
    for config in configs {
        c.bench_with_input(
            BenchmarkId::new("http_concurrent", config.concurrent_requests),
            &config,
            |b, config| {
                b.to_async(&rt).iter(|| async {
                    let mut handles = Vec::with_capacity(config.concurrent_requests);
                    
                    for _ in 0..config.concurrent_requests {
                        let client = client.clone();
                        let handle = tokio::spawn(async move {
                            let request = HttpRequest {
                                method: RequestMethod::Get,
                                url: "https://httpbin.org/get".to_string(),
                                headers: Default::default(),
                                body: None,
                                timeout: Duration::from_millis(config.timeout_ms),
                            };
                            
                            client.send_request(request).await
                        });
                        handles.push(handle);
                    }
                    
                    let results = futures::future::join_all(handles).await;
                    black_box(results)
                });
            },
        );
    }
}

/// Benchmark HTTP POST with payload
fn bench_post_with_payload(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let client = rt.block_on(async { setup_http_client().unwrap() });
    
    let payload_sizes = vec![256, 1024, 4096];
    
    for size in payload_sizes {
        c.bench_with_input(
            BenchmarkId::new("http_post_payload", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let payload = vec![0u8; size];
                    let request = HttpRequest {
                        method: RequestMethod::Post,
                        url: "https://httpbin.org/post".to_string(),
                        headers: Default::default(),
                        body: Some(payload),
                        timeout: Duration::from_millis(1),
                    };
                    
                    let result = client.send_request(black_box(request)).await;
                    black_box(result)
                });
            },
        );
    }
}

/// Benchmark connection pooling efficiency
fn bench_connection_pooling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("http_connection_pooling", |b| {
        b.to_async(&rt).iter(|| async {
            let client = setup_http_client().unwrap();
            
            // Make multiple requests to same host to test pooling
            let mut handles = Vec::with_capacity(10);
            
            for _ in 0..10 {
                let client = client.clone();
                let handle = tokio::spawn(async move {
                    let request = HttpRequest {
                        method: RequestMethod::Get,
                        url: "https://httpbin.org/get".to_string(),
                        headers: Default::default(),
                        body: None,
                        timeout: Duration::from_millis(1),
                    };
                    
                    client.send_request(request).await
                });
                handles.push(handle);
            }
            
            let results = futures::future::join_all(handles).await;
            black_box(results)
        });
    });
}

/// Benchmark HTTP/2 multiplexing
fn bench_http2_multiplexing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let client = rt.block_on(async { setup_http_client().unwrap() });
    
    c.bench_function("http2_multiplexing", |b| {
        b.to_async(&rt).iter(|| async {
            // Send multiple requests simultaneously over same connection
            let requests = (0..20).map(|i| {
                let client = client.clone();
                tokio::spawn(async move {
                    let request = HttpRequest {
                        method: RequestMethod::Get,
                        url: format!("https://httpbin.org/delay/{}", i % 3),
                        headers: Default::default(),
                        body: None,
                        timeout: Duration::from_millis(1),
                    };
                    
                    client.send_request(request).await
                })
            }).collect::<Vec<_>>();
            
            let results = futures::future::join_all(requests).await;
            black_box(results)
        });
    });
}

/// Benchmark latency measurement
fn bench_latency_measurement(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let client = rt.block_on(async { setup_http_client().unwrap() });
    
    c.bench_function("http_latency_critical", |b| {
        b.to_async(&rt).iter(|| async {
            let start = std::time::Instant::now();
            
            let request = HttpRequest {
                method: RequestMethod::Get,
                url: "https://httpbin.org/get".to_string(),
                headers: Default::default(),
                body: None,
                timeout: Duration::from_millis(1),
            };
            
            let result = client.send_request(black_box(request)).await;
            let elapsed = start.elapsed();
            
            // Assert <1ms latency requirement for financial application
            assert!(elapsed.as_millis() < 1, "Latency violation: {:?}", elapsed);
            
            black_box((result, elapsed))
        });
    });
}

criterion_group!(
    benches,
    bench_single_get_request,
    bench_concurrent_requests,
    bench_post_with_payload,
    bench_connection_pooling,
    bench_http2_multiplexing,
    bench_latency_measurement
);

criterion_main!(benches);
