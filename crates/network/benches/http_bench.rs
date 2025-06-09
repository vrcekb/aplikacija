//! HTTP Performance Benchmarks
//! 
//! Ultra-low latency HTTP benchmarks for `TallyIO` financial application.
//! Target: <1ms latency for critical real-time operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::{sync::Arc, time::Duration};
use tokio::runtime::Runtime;
use tallyio_network::prelude::*;

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

/// Benchmark single HTTP request latency
fn bench_request_latency(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_http_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(HttpClient::new(HttpConfig::default()).unwrap_or_else(|_| {
                    // This should never happen with default config
                    eprintln!("Failed to create default HTTP client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    c.bench_function("http_request_latency", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = std::time::Instant::now();
                
                let result = HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                    .timeout(Duration::from_millis(1))
                    .send(client.as_ref()).await;
                let elapsed = start.elapsed();
                
                black_box((result, elapsed))
            })
        });
    });
}

/// Benchmark concurrent HTTP requests
fn bench_concurrent_requests(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_http_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(HttpClient::new(HttpConfig::default()).unwrap_or_else(|_| {
                    eprintln!("Failed to create default HTTP client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    let concurrent_counts = vec![10, 50, 100];
    
    for count in concurrent_counts {
        c.bench_with_input(
            BenchmarkId::new("http_concurrent_requests", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut handles = Vec::with_capacity(count);
                        let start = std::time::Instant::now();
                        
                        for _ in 0_i32..count.try_into().unwrap_or(i32::MAX) {
                            let client = Arc::clone(&client);
                            let handle = tokio::spawn(async move {
                                HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                                    .timeout(Duration::from_millis(1))
                                    .send(client.as_ref()).await
                            });
                            handles.push(handle);
                        }
                        
                        let results = futures::future::join_all(handles).await;
                        let elapsed = start.elapsed();
                        
                        black_box((results, elapsed))
                    })
                });
            },
        );
    }
}

/// Benchmark POST request with different payload sizes
fn bench_post_payload_sizes(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_http_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(HttpClient::new(HttpConfig::default()).unwrap_or_else(|_| {
                    eprintln!("Failed to create default HTTP client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    let payload_sizes = vec![64, 256, 1024, 4096];
    
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
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    c.bench_function("http_connection_pooling", |b| {
        b.iter(|| {
            rt.block_on(async {
                let client = match setup_http_client() {
                    Ok(client) => client,
                    Err(e) => {
                        eprintln!("Failed to setup client: {e}");
                        return black_box((Vec::new(), std::time::Duration::ZERO));
                    }
                };
                let mut handles = Vec::with_capacity(10);
                
                let start = std::time::Instant::now();
                
                for _ in 0_i32..10_i32 {
                    let client = Arc::clone(&client);
                    let handle = tokio::spawn(async move {
                        HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                            .timeout(Duration::from_millis(1))
                            .send(client.as_ref()).await
                    });
                    handles.push(handle);
                }
                
                let results = futures::future::join_all(handles).await;
                let elapsed = start.elapsed();
                
                black_box((results, elapsed))
            })
        });
    });
}

/// Benchmark HTTP/2 multiplexing
fn bench_http2_multiplexing(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_http_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(HttpClient::new(HttpConfig::default()).unwrap_or_else(|_| {
                    eprintln!("Failed to create default HTTP client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    c.bench_function("http2_multiplexing", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = std::time::Instant::now();
                
                let requests = (0_i32..20_i32).map(|i| {
                    let client = Arc::clone(&client);
                    tokio::spawn(async move {
                        HttpRequestBuilder::new(
                            HttpMethod::Get, 
                            format!("https://httpbin.org/delay/{}", i % 3_i32)
                        )
                        .timeout(Duration::from_millis(1))
                        .send(client.as_ref()).await
                    })
                }).collect::<Vec<_>>();
                
                let results = futures::future::join_all(requests).await;
                let elapsed = start.elapsed();
                
                black_box((results, elapsed))
            })
        });
    });
}

/// Benchmark latency measurement precision
fn bench_latency_measurement(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_http_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(HttpClient::new(HttpConfig::default()).unwrap_or_else(|_| {
                    eprintln!("Failed to create default HTTP client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    c.bench_function("http_latency_measurement", |b| {
        b.iter(|| {
            rt.block_on(async {
                let start = std::time::Instant::now();
                
                let result = HttpRequestBuilder::new(HttpMethod::Get, "https://httpbin.org/get")
                    .timeout(Duration::from_millis(1))
                    .send(client.as_ref()).await;
                let elapsed = start.elapsed();
                
                black_box((result, elapsed))
            })
        });
    });
}

criterion_group!(
    benches,
    bench_request_latency,
    bench_concurrent_requests,
    bench_post_payload_sizes,
    bench_connection_pooling,
    bench_http2_multiplexing,
    bench_latency_measurement
);

criterion_main!(benches);
