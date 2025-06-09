# TallyIO Data Storage

Ultra-performant data storage layer for TallyIO MEV/DeFi platform with <1ms latency guarantee.

## 🎯 Features

- **Multi-tier Storage**: Hot (redb), Warm (PostgreSQL/TimescaleDB), Cold (encrypted archive)
- **Ultra-low Latency**: <1ms for hot path operations
- **High Throughput**: >10,000 operations/second
- **Zero Allocations**: Hot paths optimized for zero heap allocations
- **Production Ready**: Comprehensive error handling and monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Data Storage Module                        │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│   Hot    │  Cache   │Pipeline  │ Stream   │    Indexer      │
│ Storage  │  Layer   │ Engine   │Processor │   (Blockchain)  │
│ (redb)   │ (Redis)  │          │          │                 │
├──────────┼──────────┼──────────┼──────────┼─────────────────┤
│         Warm Storage (PostgreSQL + TimescaleDB)            │
├─────────────────────────────────────────────────────────────┤
│              Cold Storage (Encrypted Archive)              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```rust
use tallyio_data_storage::{DataStorage, DataStorageConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = DataStorageConfig::default();
    let storage = DataStorage::new(config).await?;
    
    // Hot path - <1ms operations
    let opportunity = storage.get_opportunity_fast(&opportunity_id).await?;
    
    // Warm path - analytical queries
    let opportunities = storage.query_opportunities(filter).await?;
    
    Ok(())
}
```

## 📊 Performance Targets

- **Hot Storage**: <1ms (redb + memory cache)
- **Warm Storage**: <10ms (PostgreSQL)
- **Cold Storage**: <100ms (encrypted archive)
- **Throughput**: >10,000 ops/sec
- **Memory**: Zero allocations in hot paths

## 🔧 Configuration

```toml
[data_storage]
hot_storage_path = "./data/hot"
warm_storage_url = "postgresql://user:pass@localhost/tallyio"
cache_url = "redis://localhost:6379"
max_connections = 100
connection_timeout_ms = 5000
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run performance benchmarks
cargo bench

# Run with coverage
cargo test --features test-coverage
```

## 📈 Monitoring

The module provides comprehensive metrics:

- Storage operation latencies
- Cache hit/miss ratios
- Connection pool statistics
- Data pipeline throughput
- Error rates and types

## 🔒 Security

- Encrypted storage for sensitive data
- Secure connection handling
- Input validation and sanitization
- Audit logging for all operations
