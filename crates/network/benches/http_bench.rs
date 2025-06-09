//! HTTP Client Performance Benchmarks
//!
//! Ultra-low latency HTTP benchmarks for TallyIO financial application.
//! Target: <1ms latency for critical operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::{sync::Arc, time::Duration};
use tokio::runtime::Runtime;
use tallyio_network::prelude::*;

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
fn setup_http_client() -> NetworkResult<Arc<HttpClient>> {
    let config = HttpConfig {
        max_connections_per_host: 1000,
        connection_timeout_s: 1,
        request_timeout_s: 1,
        enable_http2: true,
        enable_http3: false,
        ..HttpConfig::default()
    };

    HttpClient::new(config).map(Arc::new)
}

/// Benchmark single HTTP GET request
fn bench_single_get_request(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async {
        setup_http_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap()
    });

    c.bench_function("http_single_get", |b| {
        b.iter(|| {
            rt.block_on(async {
                let result = HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                    .timeout(Duration::from_millis(1))
                    .send(client.as_ref()).await;
                black_box(result)
            })
        });
    });
}

/// Benchmark concurrent HTTP requests
fn bench_concurrent_requests(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async {
        setup_http_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap()
    });

    let configs = vec![
        BenchConfig { concurrent_requests: 10, ..BenchConfig::default() },
        BenchConfig { concurrent_requests: 50, ..BenchConfig::default() },
        BenchConfig { concurrent_requests: 100, ..BenchConfig::default() },
    ];

    for config in configs {
        c.bench_with_input(
            BenchmarkId::new("http_concurrent", config.concurrent_requests),
            &config,
            |b, config| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::with_capacity(config.concurrent_requests);

                        for _ in 0..config.concurrent_requests {
                            let client = Arc::clone(&client);
                            let handle = tokio::spawn(async move {
                                HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                                    .timeout(Duration::from_millis(config.timeout_ms))
                                    .send(client.as_ref()).await
                            });
                            handles.push(handle);
                        }

                        let results = futures::future::join_all(handles).await;
                        black_box(results)
                    })
                });
            },
        );
    }
}

/// Benchmark HTTP POST with payload
fn bench_post_with_payload(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async {
        setup_http_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap()
    });

    let payload_sizes = vec![256, 1024, 4096];

    for size in payload_sizes {
        c.bench_with_input(
            BenchmarkId::new("http_post_payload", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let payload = vec![0u8; size];
                        let result = HttpRequestBuilder::new(HttpMethod::Post, "https://httpbin.org/post")
                            .body(payload)
                            .timeout(Duration::from_millis(1))
                            .send(client.as_ref()).await;
                        black_box(result)
                    })
                });
            },
        );
    }
}

/// Benchmark connection pooling efficiency
fn bench_connection_pooling(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();

    c.bench_function("http_connection_pooling", |b| {
        b.iter(|| {
            rt.block_on(async {
                let client = setup_http_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap();

                // Make multiple requests to same host to test pooling
                let mut handles = Vec::with_capacity(10);

                for _ in 0..10 {
                    let client = Arc::clone(&client);
                    let handle = tokio::spawn(async move {
                        HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                            .timeout(Duration::from_millis(1))
                            .send(client.as_ref()).await
                    });
                    handles.push(handle);
                }

                let results = futures::future::join_all(handles).await;
                black_box(results)
            })
        });
    });
}

/// Benchmark HTTP/2 multiplexing
fn bench_http2_multiplexing(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async {
        setup_http_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap()
    });

    c.bench_function("http2_multiplexing", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Send multiple requests simultaneously over same connection
                let requests = (0..20).map(|i| {
                    let client = Arc::clone(&client);
                    tokio::spawn(async move {
                        HttpRequestBuilder::new(
                            HttpMethod::Get,
                            &format!("https://httpbin.org/delay/{}", i % 3)
                        )
                        .timeout(Duration::from_millis(1))
                        .send(client.as_ref()).await
                    })
                }).collect::<Vec<_>>();

                let results = futures::future::join_all(requests).await;
                black_box(results)
            })
        });
    });
}

/// Benchmark latency measurement
fn bench_latency_measurement(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async {
        setup_http_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap()
    });

    c.bench_function("http_latency_critical", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = std::time::Instant::now();

                let result = HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                    .timeout(Duration::from_millis(1))
                    .send(client.as_ref()).await;
                let elapsed = start.elapsed();

                // Note: <1ms latency requirement for financial application
                // In real benchmarks, this would be measured properly

                black_box((result, elapsed))
            })
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
