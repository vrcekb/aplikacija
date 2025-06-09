# TallyIO Network Module

🚀 **Ultra-performant network layer with <1ms latency for financial trading**

## 🎯 **OVERVIEW**

High-performance network module providing WebSocket, HTTP, and P2P networking capabilities for TallyIO financial trading platform. Designed for ultra-low latency (<1ms) and zero-panic guarantees.

## 🏗️ **ARCHITECTURE**

```rust
pub struct NetworkManager {
    http_client: Arc<HttpClient>,
    ws_manager: Arc<WebSocketManager>,
    p2p_network: Option<Arc<P2PNetwork>>,
    load_balancer: Arc<LoadBalancer>,
}
```

### **Key Components:**

- **HttpClient**: HTTP/2 and HTTP/3 support with connection pooling
- **WebSocketManager**: Automatic reconnection with message buffering
- **LoadBalancer**: Multiple strategies with health checking
- **P2PNetwork**: Decentralized networking (future expansion)

## ⚡ **PERFORMANCE FEATURES**

- **<1ms latency** for critical paths
- **Zero allocations** in hot paths
- **Lock-free** data structures
- **Connection pooling** with reuse
- **Batch processing** for throughput
- **SIMD optimizations** where applicable

## 🔒 **SECURITY FEATURES**

- **TLS 1.3** mandatory for production
- **Certificate pinning** for critical endpoints
- **Rate limiting** and DDoS protection
- **Request signing** for authentication
- **Circuit breaker** patterns for resilience

## 🚀 **USAGE**

```rust
use tallyio_network::prelude::*;

// Create network manager
let config = NetworkConfig::default();
let manager = NetworkManager::new(config).await?;

// HTTP request with retry
let response = manager
    .http_client()
    .get("https://api.example.com/data")
    .retry_policy(RetryPolicy::exponential())
    .send()
    .await?;

// WebSocket connection with auto-reconnect
let ws_conn = manager
    .ws_manager()
    .connect("wss://stream.example.com")
    .with_handlers(WsHandlers {
        on_message: |msg| async move { /* handle */ },
        on_error: |err| async move { /* handle */ },
        on_close: || async move { /* handle */ },
    })
    .await?;
```

## 📊 **METRICS**

- Connection pool utilization
- Request/response latencies
- WebSocket message rates
- Error rates and circuit breaker states
- Load balancer health checks

## 🧪 **TESTING**

```bash
# Run all tests
cargo test

# Run benchmarks
cargo bench

# Run with coverage
cargo llvm-cov --html
```

## 🔧 **CONFIGURATION**

```rust
#[derive(Validate)]
pub struct NetworkConfig {
    #[garde(range(min = 1, max = 1000))]
    pub max_connections: u32,
    
    #[garde(range(min = 1, max = 60))]
    pub connection_timeout_s: u64,
    
    #[garde(dive)]
    pub retry_policy: RetryConfig,
    
    #[garde(dive)]
    pub load_balancer: LoadBalancerConfig,
}
```

---

**Production-ready networking for financial applications where every millisecond counts.**
