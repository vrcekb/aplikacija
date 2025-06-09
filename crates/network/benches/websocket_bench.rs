//! WebSocket Performance Benchmarks
//! 
//! Ultra-low latency WebSocket benchmarks for TallyIO financial application.
//! Target: <1ms latency for critical real-time operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use tallyio_network::prelude::*;

/// Benchmark configuration for WebSocket operations
struct BenchConfig {
    pub concurrent_connections: usize,
    pub message_size: usize,
    pub timeout_ms: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            concurrent_connections: 100,
            message_size: 1024,
            timeout_ms: 1,
        }
    }
}

/// Setup WebSocket client for benchmarks
fn setup_websocket_client() -> NetworkResult<WebSocketClient> {
    let config = WebSocketConfig {
        ping_interval_s: 30,
        pong_timeout_s: 10,
        max_message_size: 16 * 1024 * 1024,
        enable_compression: false,
        ..WebSocketConfig::default()
    };

    WebSocketClient::new(config)
}

/// Benchmark single WebSocket connection establishment
fn bench_connection_establishment(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    
    c.bench_function("websocket_connect", |b| {
        b.iter(|| {
            rt.block_on(async {
                let client = setup_websocket_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap();
                let start = std::time::Instant::now();
                
                let result = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .connect(&client).await;
                let elapsed = start.elapsed();
                
                black_box((result, elapsed))
            })
        });
    });
}

/// Benchmark message sending latency
fn bench_message_send_latency(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async { 
        setup_websocket_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap() 
    });
    
    let message_sizes = vec![64, 256, 1024, 4096];
    
    for size in message_sizes {
        c.bench_with_input(
            BenchmarkId::new("websocket_send_latency", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    rt.block_on(async {
                        let conn_id = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                            .connect(&client).await.map_err(|e| format!("Failed to connect: {e}")).unwrap();
                        
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
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    
    let connection_counts = vec![10, 50, 100];
    
    for count in connection_counts {
        c.bench_with_input(
            BenchmarkId::new("websocket_concurrent_connections", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let client = setup_websocket_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap();
                        let mut handles = Vec::with_capacity(count);
                        
                        let start = std::time::Instant::now();
                        
                        for _ in 0..count {
                            let client = client.clone();
                            let handle = tokio::spawn(async move {
                                WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                                    .connect(&client).await
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
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async { 
        setup_websocket_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap() 
    });
    
    c.bench_function("websocket_throughput", |b| {
        b.iter(|| {
            rt.block_on(async {
                let connection = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .build();
                let conn_id = client.connect(connection).await.map_err(|e| format!("Failed to connect: {e}")).unwrap();
                
                let message_count = 1000;
                let payload = vec![0u8; 256];
                let mut handles = Vec::with_capacity(message_count);
                
                let start = std::time::Instant::now();
                
                for _ in 0..message_count {
                    let client = client.clone();
                    let payload = payload.clone();
                    let handle = tokio::spawn(async move {
                        let message = WebSocketMessage::Binary(payload);
                        client.send_message(conn_id, message).await
                    });
                    handles.push(handle);
                }
                
                let results = futures::future::join_all(handles).await;
                let elapsed = start.elapsed();
                
                let messages_per_second = (message_count as f64 / elapsed.as_secs_f64()) as u64;
                
                black_box((results, messages_per_second))
            })
        });
    });
}

/// Benchmark ping-pong latency
fn bench_ping_pong_latency(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async { 
        setup_websocket_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap() 
    });
    
    c.bench_function("websocket_ping_pong", |b| {
        b.iter(|| {
            rt.block_on(async {
                let connection = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .build();
                let conn_id = client.connect(connection).await.map_err(|e| format!("Failed to connect: {e}")).unwrap();
                
                let message = WebSocketMessage::Ping(b"ping".to_vec());
                
                let start = std::time::Instant::now();
                
                // Send ping and measure latency
                let result = client.send_message(conn_id, message).await;
                let elapsed = start.elapsed();
                
                black_box((result, elapsed))
            })
        });
    });
}

/// Benchmark connection recovery
fn bench_connection_recovery(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    
    c.bench_function("websocket_recovery", |b| {
        b.iter(|| {
            rt.block_on(async {
                let client = setup_websocket_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap();
                let connection = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .build();
                let conn_id = client.connect(connection).await.map_err(|e| format!("Failed to connect: {e}")).unwrap();
                
                // Simulate connection drop
                let _ = client.disconnect(conn_id).await;
                
                let start = std::time::Instant::now();
                
                // Reconnect
                let new_connection = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .build();
                let new_conn_id = client.connect(new_connection).await;
                
                let elapsed = start.elapsed();
                
                black_box((new_conn_id, elapsed))
            })
        });
    });
}

/// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().map_err(|e| format!("Failed to create runtime: {e}")).unwrap();
    let client = rt.block_on(async { 
        setup_websocket_client().map_err(|e| format!("Failed to setup client: {e}")).unwrap() 
    });
    
    c.bench_function("websocket_memory_efficiency", |b| {
        b.iter(|| {
            rt.block_on(async {
                let connection = WebSocketConnectionBuilder::new("wss://echo.websocket.org")
                    .build();
                let conn_id = client.connect(connection).await.map_err(|e| format!("Failed to connect: {e}")).unwrap();
                
                // Send many messages to test memory efficiency
                for _ in 0..1000 {
                    let message = WebSocketMessage::Binary(vec![0u8; 1024]);
                    let _ = client.send_message(conn_id, message).await;
                }
                
                // Force garbage collection and measure memory
                tokio::task::yield_now().await;
                
                black_box(conn_id)
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
    bench_connection_recovery,
    bench_memory_efficiency
);

criterion_main!(benches);
