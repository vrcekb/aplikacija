# TallyIO Core - Ultra-Performance Engine

⚡ **Production-ready core engine with <1ms latency guarantee**

## 🎯 **PURPOSE**

The core crate provides the foundational ultra-high performance engine for TallyIO's financial trading platform. Every component is optimized for sub-millisecond latency with zero-panic guarantees.

## 🏗️ **ARCHITECTURE**

```
crates/core/
├── src/
│   ├── engine/           # Main execution engine
│   │   ├── executor.rs   # Strategy execution
│   │   ├── scheduler.rs  # Task scheduling  
│   │   └── worker.rs     # Worker threads
│   ├── state/            # State management
│   │   ├── global.rs     # Global state
│   │   ├── local.rs      # Thread-local state
│   │   └── sync.rs       # Synchronization
│   ├── mempool/          # Mempool monitoring
│   │   ├── watcher.rs    # Mempool watcher
│   │   ├── analyzer.rs   # Transaction analyzer
│   │   └── filter.rs     # Transaction filter
│   ├── optimization/     # Performance optimizations
│   │   ├── cpu_affinity.rs  # CPU pinning
│   │   ├── memory_pool.rs   # Memory pooling
│   │   ├── lock_free.rs     # Lock-free structures
│   │   └── simd.rs          # SIMD optimizations
│   ├── types/            # Core types
│   │   ├── opportunity.rs   # Opportunity types
│   │   ├── transaction.rs   # Transaction types
│   │   └── result.rs        # Result types
│   ├── error.rs          # Error definitions
│   ├── config.rs         # Core configuration
│   └── prelude.rs        # Common imports
├── benches/              # Performance benchmarks
└── tests/                # Unit tests
```

## ⚡ **PERFORMANCE GUARANTEES**

- **Critical Path Latency:** <1ms (MANDATORY)
- **Memory Allocation:** Zero in hot paths
- **Panic Policy:** Zero panics (unwrap/expect FORBIDDEN)
- **Concurrency:** Lock-free data structures
- **CPU Efficiency:** SIMD optimizations where applicable

## 🔧 **FEATURES**

### **Engine Module**
- Ultra-fast strategy execution
- Lock-free task scheduling
- Worker thread management
- Circuit breaker patterns

### **State Management**
- Global state coordination
- Thread-local optimizations
- Lock-free synchronization
- Memory-mapped persistence

### **Mempool Monitoring**
- Real-time transaction watching
- Sub-millisecond analysis
- Intelligent filtering
- Opportunity detection

### **Optimization Layer**
- CPU affinity management
- Memory pool allocation
- SIMD vectorization
- Cache-line optimization

## 🚨 **USAGE EXAMPLE**

```rust
use tallyio_core::prelude::*;

#[tokio::main]
async fn main() -> CoreResult<()> {
    // Initialize core engine
    let config = CoreConfig::production()?;
    let engine = Engine::new(config)?;
    
    // Start monitoring
    engine.start().await?;
    
    // Process opportunities
    while let Some(opportunity) = engine.next_opportunity().await? {
        let result = engine.execute_strategy(&opportunity).await?;
        println!("Executed: {:?}", result);
    }
    
    Ok(())
}
```

## 📊 **BENCHMARKS**

```bash
# Run performance benchmarks
cargo bench

# Expected results:
# engine_execution    time: [245.2 μs 251.7 μs 259.1 μs]
# mempool_analysis     time: [89.3 μs 92.1 μs 95.8 μs]
# state_sync          time: [12.4 μs 13.1 μs 13.9 μs]
```

## 🔒 **SAFETY GUARANTEES**

- **Memory Safety:** No unsafe code without documentation
- **Thread Safety:** All public APIs are Send + Sync
- **Error Handling:** All operations return Result<T, E>
- **Input Validation:** All external inputs validated
- **Resource Management:** RAII patterns throughout

## 🧪 **TESTING**

```bash
# Run all tests
cargo test

# Run with coverage
cargo test --features coverage

# Run property-based tests
cargo test --features proptest
```

## 📈 **MONITORING**

The core engine provides comprehensive metrics:

- Execution latency histograms
- Memory usage tracking
- CPU utilization monitoring
- Error rate tracking
- Throughput measurements

---

**TallyIO Core** - Where nanoseconds matter in financial technology.
