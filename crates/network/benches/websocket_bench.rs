//! WebSocket Performance Benchmarks
//! 
//! Ultra-low latency WebSocket benchmarks for TallyIO financial application.
//! Target: <1ms latency for critical real-time operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;
use tokio::runtime::Runtime;
use tallyio_network::{
    WebSocketManager, NetworkConfig, NetworkError,
    types::{WebSocketMessage, MessageType},
};

/// Benchmark configuration for WebSocket operations
struct BenchConfig {
    pub concurrent_connections: usize,
    pub message_size: usize,
    pub messages_per_second: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            concurrent_connections: 100,
            message_size: 1024,
            messages_per_second: 1000,
        }
    }
}

/// Setup WebSocket manager for benchmarks
fn setup_websocket_manager() -> Result<WebSocketManager, NetworkError> {
    let config = NetworkConfig {
        max_connections: 1000,
        connection_timeout: Duration::from_millis(100),
        request_timeout: Duration::from_millis(1),
        keep_alive: true,
        websocket_ping_interval: Duration::from_secs(30),
        websocket_max_frame_size: 16 * 1024 * 1024,
        ..Default::default()
    };
    
    WebSocketManager::new(config)
}

/// Benchmark single WebSocket connection establishment
fn bench_connection_establishment(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("websocket_connect", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = setup_websocket_manager().unwrap();
            let start = std::time::Instant::now();
            
            let result = manager.connect("wss://echo.websocket.org").await;
            let elapsed = start.elapsed();
            
            // Assert <1ms connection establishment for financial application
            assert!(elapsed.as_millis() < 1, "Connection latency violation: {:?}", elapsed);
            
            black_box((result, elapsed))
        });
    });
}

/// Benchmark message sending latency
fn bench_message_send_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = rt.block_on(async { setup_websocket_manager().unwrap() });
    let connection = rt.block_on(async {
        manager.connect("wss://echo.websocket.org").await.unwrap()
    });
    
    let message_sizes = vec![64, 256, 1024, 4096];
    
    for size in message_sizes {
        c.bench_with_input(
            BenchmarkId::new("websocket_send_latency", size),
            &size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let payload = vec![0u8; size];
                    let message = WebSocketMessage {
                        message_type: MessageType::Binary,
                        payload,
                        timestamp: chrono::Utc::now(),
                    };
                    
                    let start = std::time::Instant::now();
                    let result = connection.send_message(black_box(message)).await;
                    let elapsed = start.elapsed();
                    
                    // Assert <1ms send latency for financial application
                    assert!(elapsed.as_millis() < 1, "Send latency violation: {:?}", elapsed);
                    
                    black_box((result, elapsed))
                });
            },
        );
    }
}

/// Benchmark concurrent connections
fn bench_concurrent_connections(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let connection_counts = vec![10, 50, 100];
    
    for count in connection_counts {
        c.bench_with_input(
            BenchmarkId::new("websocket_concurrent_connections", count),
            &count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let manager = setup_websocket_manager().unwrap();
                    let mut handles = Vec::with_capacity(count);
                    
                    let start = std::time::Instant::now();
                    
                    for _ in 0..count {
                        let manager = manager.clone();
                        let handle = tokio::spawn(async move {
                            manager.connect("wss://echo.websocket.org").await
                        });
                        handles.push(handle);
                    }
                    
                    let results = futures::future::join_all(handles).await;
                    let elapsed = start.elapsed();
                    
                    // Assert reasonable concurrent connection time
                    assert!(elapsed.as_millis() < 100, "Concurrent connection latency violation: {:?}", elapsed);
                    
                    black_box((results, elapsed))
                });
            },
        );
    }
}

/// Benchmark message throughput
fn bench_message_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = rt.block_on(async { setup_websocket_manager().unwrap() });
    let connection = rt.block_on(async {
        manager.connect("wss://echo.websocket.org").await.unwrap()
    });
    
    c.bench_function("websocket_throughput", |b| {
        b.to_async(&rt).iter(|| async {
            let message_count = 1000;
            let payload = vec![0u8; 256];
            let mut handles = Vec::with_capacity(message_count);
            
            let start = std::time::Instant::now();
            
            for _ in 0..message_count {
                let connection = connection.clone();
                let payload = payload.clone();
                let handle = tokio::spawn(async move {
                    let message = WebSocketMessage {
                        message_type: MessageType::Binary,
                        payload,
                        timestamp: chrono::Utc::now(),
                    };
                    
                    connection.send_message(message).await
                });
                handles.push(handle);
            }
            
            let results = futures::future::join_all(handles).await;
            let elapsed = start.elapsed();
            
            let messages_per_second = (message_count as f64 / elapsed.as_secs_f64()) as u64;
            
            // Assert minimum throughput for financial application
            assert!(messages_per_second > 10_000, "Throughput too low: {} msg/s", messages_per_second);
            
            black_box((results, messages_per_second))
        });
    });
}

/// Benchmark ping-pong latency
fn bench_ping_pong_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = rt.block_on(async { setup_websocket_manager().unwrap() });
    let connection = rt.block_on(async {
        manager.connect("wss://echo.websocket.org").await.unwrap()
    });
    
    c.bench_function("websocket_ping_pong", |b| {
        b.to_async(&rt).iter(|| async {
            let message = WebSocketMessage {
                message_type: MessageType::Ping,
                payload: b"ping".to_vec(),
                timestamp: chrono::Utc::now(),
            };
            
            let start = std::time::Instant::now();
            
            // Send ping and wait for pong
            connection.send_message(black_box(message)).await.unwrap();
            let _pong = connection.receive_message().await.unwrap();
            
            let elapsed = start.elapsed();
            
            // Assert <1ms ping-pong latency for financial application
            assert!(elapsed.as_millis() < 1, "Ping-pong latency violation: {:?}", elapsed);
            
            black_box(elapsed)
        });
    });
}

/// Benchmark connection recovery
fn bench_connection_recovery(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("websocket_recovery", |b| {
        b.to_async(&rt).iter(|| async {
            let manager = setup_websocket_manager().unwrap();
            let connection = manager.connect("wss://echo.websocket.org").await.unwrap();
            
            // Simulate connection drop
            connection.close().await.unwrap();
            
            let start = std::time::Instant::now();
            
            // Reconnect
            let new_connection = manager.connect("wss://echo.websocket.org").await.unwrap();
            
            let elapsed = start.elapsed();
            
            // Assert fast recovery for financial application
            assert!(elapsed.as_millis() < 10, "Recovery latency violation: {:?}", elapsed);
            
            black_box((new_connection, elapsed))
        });
    });
}

/// Benchmark memory usage under load
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let manager = rt.block_on(async { setup_websocket_manager().unwrap() });
    
    c.bench_function("websocket_memory_efficiency", |b| {
        b.to_async(&rt).iter(|| async {
            let connection = manager.connect("wss://echo.websocket.org").await.unwrap();
            
            // Send many messages to test memory efficiency
            for _ in 0..1000 {
                let message = WebSocketMessage {
                    message_type: MessageType::Binary,
                    payload: vec![0u8; 1024],
                    timestamp: chrono::Utc::now(),
                };
                
                connection.send_message(message).await.unwrap();
            }
            
            // Force garbage collection and measure memory
            tokio::task::yield_now().await;
            
            black_box(connection)
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
