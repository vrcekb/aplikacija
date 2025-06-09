//! WebSocket Performance Benchmarks
//! 
//! Ultra-low latency WebSocket benchmarks for `TallyIO` financial application.
//! Target: <1ms latency for critical real-time operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::sync::Arc;
use tokio::runtime::Runtime;
use tallyio_network::prelude::*;

/// Setup WebSocket client for benchmarks
fn setup_websocket_client() -> NetworkResult<Arc<WebSocketClient>> {
    let config = WebSocketConfig {
        ping_interval_s: 30,
        pong_timeout_s: 10,
        max_message_size: 16 * 1024 * 1024,
        enable_compression: false,
        ..WebSocketConfig::default()
    };
    
    WebSocketClient::new(config).map(Arc::new)
}

/// Benchmark single WebSocket connection establishment
fn bench_connection_establishment(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    c.bench_function("websocket_connect", |b| {
        b.iter(|| {
            rt.block_on(async {
                let client = match setup_websocket_client() {
                    Ok(client) => client,
                    Err(e) => {
                        eprintln!("Failed to setup client: {e}");
                        return black_box((Err(NetworkError::internal("Setup failed")), std::time::Duration::ZERO));
                    }
                };
                let start = std::time::Instant::now();
                
                let result = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .connect(client.as_ref()).await;
                let elapsed = start.elapsed();
                
                black_box((result, elapsed))
            })
        });
    });
}

/// Benchmark message sending latency
fn bench_message_send_latency(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_websocket_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(WebSocketClient::new(WebSocketConfig::default()).unwrap_or_else(|_| {
                    eprintln!("Failed to create default WebSocket client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    let message_sizes = vec![64, 256, 1024, 4096];
    
    for size in message_sizes {
        c.bench_with_input(
            BenchmarkId::new("websocket_send_latency", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let conn_id = match WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                            .connect(client.as_ref()).await {
                            Ok(id) => id,
                            Err(e) => {
                                eprintln!("Failed to connect: {e}");
                                return black_box((Err(NetworkError::internal("Connection failed")), std::time::Duration::ZERO));
                            }
                        };
                        
                        let payload = vec![0u8; size];
                        let message = WebSocketMessage::Binary(payload);
                        
                        let start = std::time::Instant::now();
                        let result = client.send(conn_id, message).await;
                        let elapsed = start.elapsed();
                        
                        black_box((result, elapsed))
                    })
                });
            },
        );
    }
}

/// Benchmark concurrent connections
fn bench_concurrent_connections(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    let connection_counts = vec![10, 50, 100];
    
    for count in connection_counts {
        c.bench_with_input(
            BenchmarkId::new("websocket_concurrent_connections", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let client = match setup_websocket_client() {
                            Ok(client) => client,
                            Err(e) => {
                                eprintln!("Failed to setup client: {e}");
                                return black_box((Vec::new(), std::time::Duration::ZERO));
                            }
                        };
                        let mut handles = Vec::with_capacity(count);
                        
                        let start = std::time::Instant::now();
                        
                        for _ in 0_i32..count.try_into().unwrap_or(i32::MAX) {
                            let client = Arc::clone(&client);
                            let handle = tokio::spawn(async move {
                                WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                                    .connect(client.as_ref()).await
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

/// Benchmark message throughput
fn bench_message_throughput(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_websocket_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(WebSocketClient::new(WebSocketConfig::default()).unwrap_or_else(|_| {
                    eprintln!("Failed to create default WebSocket client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    c.bench_function("websocket_throughput", |b| {
        b.iter(|| {
            rt.block_on(async {
                let conn_id = match WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .connect(client.as_ref()).await {
                    Ok(id) => id,
                    Err(e) => {
                        eprintln!("Failed to connect: {e}");
                        return black_box((Vec::new(), 0_u64));
                    }
                };
                
                let message_count = 1000_usize;
                let payload = vec![0u8; 256];
                let mut handles = Vec::with_capacity(message_count);
                
                let start = std::time::Instant::now();
                
                for _ in 0_i32..1_000_i32 {
                    let client = Arc::clone(&client);
                    let payload = payload.clone();
                    let handle = tokio::spawn(async move {
                        let message = WebSocketMessage::Binary(payload);
                        client.send(conn_id, message).await
                    });
                    handles.push(handle);
                }
                
                let results = futures::future::join_all(handles).await;
                let elapsed = start.elapsed();
                
                // Safe calculation avoiding precision loss
                let messages_per_second = if elapsed.as_secs_f64() > 0.0_f64 {
                    let max_safe_usize = 2_usize.pow(52); // f64 mantissa precision limit
                    if message_count > max_safe_usize {
                        u64::MAX // Avoid precision loss for very large values
                    } else {
                        #[allow(clippy::cast_precision_loss)] // Checked above
                        let result = (message_count as f64 / elapsed.as_secs_f64()).round();
                        // Safe conversion with bounds checking
                        if result >= f64::from(u32::MAX) {
                            u64::from(u32::MAX)
                        } else if result < 0.0_f64 {
                            0_u64
                        } else {
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Bounds checked
                            { result as u64 }
                        }
                    }
                } else {
                    u64::MAX // Infinite throughput for zero time
                };
                
                black_box((results, messages_per_second))
            })
        });
    });
}

/// Benchmark ping-pong latency
fn bench_ping_pong_latency(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    let client = rt.block_on(async { 
        match setup_websocket_client() {
            Ok(client) => client,
            Err(e) => {
                eprintln!("Failed to setup client: {e}");
                Arc::new(WebSocketClient::new(WebSocketConfig::default()).unwrap_or_else(|_| {
                    eprintln!("Failed to create default WebSocket client");
                    std::process::exit(1);
                }))
            }
        }
    });
    
    c.bench_function("websocket_ping_pong", |b| {
        b.iter(|| {
            rt.block_on(async {
                let conn_id = match WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .connect(client.as_ref()).await {
                    Ok(id) => id,
                    Err(e) => {
                        eprintln!("Failed to connect: {e}");
                        return black_box((Err(NetworkError::internal("Connection failed")), std::time::Duration::ZERO));
                    }
                };
                
                let message = WebSocketMessage::Ping(b"ping".to_vec());
                
                let start = std::time::Instant::now();
                
                // Send ping and measure latency
                let result = client.send(conn_id, message).await;
                let elapsed = start.elapsed();
                
                black_box((result, elapsed))
            })
        });
    });
}

/// Benchmark connection recovery
fn bench_connection_recovery(c: &mut Criterion) {
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Failed to create runtime: {e}");
            return;
        }
    };
    
    c.bench_function("websocket_recovery", |b| {
        b.iter(|| {
            rt.block_on(async {
                let client = match setup_websocket_client() {
                    Ok(client) => client,
                    Err(e) => {
                        eprintln!("Failed to setup client: {e}");
                        return black_box((Err(NetworkError::internal("Setup failed")), std::time::Duration::ZERO));
                    }
                };
                let conn_id = match WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .connect(client.as_ref()).await {
                    Ok(id) => id,
                    Err(e) => {
                        eprintln!("Failed to connect: {e}");
                        return black_box((Err(NetworkError::internal("Connection failed")), std::time::Duration::ZERO));
                    }
                };
                
                // Simulate connection drop
                let _ = client.close(conn_id).await;
                
                let start = std::time::Instant::now();
                
                // Reconnect
                let new_conn_id = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .connect(client.as_ref()).await;
                
                let elapsed = start.elapsed();
                
                black_box((new_conn_id, elapsed))
            })
        });
    });
}

criterion_group!(
    benches,
    bench_connection_establishment,
    bench_message_send_latency,
    bench_concurrent_connections,
    bench_message_throughput,
    bench_ping_pong_latency,
    bench_connection_recovery
);

criterion_main!(benches);
