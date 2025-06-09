# Ultra-Optimized Engine Module

This module contains ultra-optimized components designed for <1ms latency requirements in MEV trading and financial applications.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Ultra Engine                             │
├─────────────────┬───────────────────┬───────────────────────┤
│  NUMA Scheduler │  Work Stealing    │   Load Balancer       │
│  (<100μs)       │  Deques (<50μs)   │   (<10μs)             │
├─────────────────┼───────────────────┼───────────────────────┤
│  Worker Threads │  Task Execution   │   CPU Affinity        │
│  (NUMA-aware)   │  (Parallel)       │   (Core Binding)      │
└─────────────────┴───────────────────┴───────────────────────┘
```

## Components

### 1. NUMA Scheduler (`numa_scheduler.rs`)
- **Purpose**: NUMA-aware task scheduling for optimal memory locality
- **Latency Target**: <100μs for task submission
- **Features**:
  - Automatic NUMA topology detection
  - Load balancing across NUMA nodes
  - Work stealing between workers
  - Real-time metrics collection

### 2. Work Stealing Deques (`work_stealing.rs`)
- **Purpose**: Lock-free work stealing for parallel task execution
- **Latency Target**: <50μs for task push/pop operations
- **Features**:
  - Lock-free local and steal queues
  - NUMA-aware steal target selection
  - CPU affinity binding
  - Performance statistics

### 3. Load Balancer (`load_balancer.rs`)
- **Purpose**: Atomic load balancing across NUMA nodes
- **Latency Target**: <10μs for load updates
- **Features**:
  - Atomic load tracking
  - NUMA-aware load distribution
  - Real-time load metrics
  - Automatic rebalancing

### 4. Worker Threads (`worker.rs`)
- **Purpose**: High-performance worker thread implementation
- **Features**:
  - CPU affinity binding
  - NUMA-aware memory allocation
  - Task execution with metrics
  - Graceful shutdown handling

## Performance Targets

| Component | Operation | Target Latency | Measured |
|-----------|-----------|----------------|----------|
| NUMA Scheduler | Task Submit | <100μs | TBD |
| Work Stealing | Push/Pop | <50μs | TBD |
| Load Balancer | Load Update | <10μs | TBD |
| Worker Thread | Task Execute | <1ms | TBD |

## Usage Example

```rust
use tallyio_core::engine::ultra::{
    NumaScheduler, SchedulerConfig, Task, TaskResult
};

// Define a task
struct MyTask {
    data: Vec<u8>,
}

impl Task for MyTask {
    fn execute(&self) -> TaskResult {
        // Process data with <1ms latency
        TaskResult::Success
    }
    
    fn priority(&self) -> u8 { 5 }
    fn estimated_duration_us(&self) -> u64 { 500 }
}

// Create scheduler
let config = SchedulerConfig {
    workers_per_numa_node: 4,
    queue_capacity: 8192,
    enable_work_stealing: true,
    enable_cpu_affinity: true,
};

let scheduler = NumaScheduler::new(config)?;
scheduler.start()?;

// Submit tasks
let task = Arc::new(MyTask { data: vec![1, 2, 3] });
scheduler.submit_task(task)?;

// Get statistics
let stats = scheduler.stats();
println!("Tasks processed: {}", stats.metrics.tasks_processed);
```

## Configuration

### Environment Variables
- `TALLYIO_NUMA_ENABLED`: Enable NUMA awareness (default: true)
- `TALLYIO_CPU_AFFINITY`: Enable CPU affinity (default: true)
- `TALLYIO_WORKERS_PER_NODE`: Workers per NUMA node (default: 4)

### Scheduler Configuration
```rust
pub struct SchedulerConfig {
    /// Number of worker threads per NUMA node
    pub workers_per_numa_node: usize,
    
    /// Capacity of each work queue
    pub queue_capacity: usize,
    
    /// Enable work stealing between workers
    pub enable_work_stealing: bool,
    
    /// Enable CPU affinity binding
    pub enable_cpu_affinity: bool,
    
    /// Load balancing threshold (0.0-1.0)
    pub load_balance_threshold: f64,
}
```

## Benchmarks

Run benchmarks to validate performance targets:

```bash
cargo bench --bench ultra_optimized_bench
```

Expected results:
- SPSC Queue: <100ns per operation
- Memory Pool: <1000ns per operation
- End-to-end workflow: <100μs

## Testing

Integration tests validate component interaction:

```bash
cargo test --test ultra_integration_tests
```

## NUMA Topology Detection

The scheduler automatically detects NUMA topology:

```rust
let topology = NumaTopology::detect()?;
println!("NUMA nodes: {}", topology.node_count());
println!("CPUs per node: {}", topology.cpus_per_node());
```

## Monitoring and Metrics

Real-time metrics are available:

```rust
let stats = scheduler.stats();
println!("Queue utilization: {:.2}%", 
    stats.total_queue_size() as f64 / stats.total_capacity() as f64 * 100.0);
println!("Steal ratio: {:.2}", stats.average_steal_ratio());
println!("Throughput: {:.0} tasks/sec", stats.throughput());
```

## Error Handling

All operations return `UltraEngineResult<T>`:

```rust
match scheduler.submit_task(task) {
    Ok(()) => println!("Task submitted successfully"),
    Err(UltraEngineError::QueueFull) => {
        // Handle queue full condition
    },
    Err(e) => eprintln!("Error: {:?}", e),
}
```

## Safety Considerations

- All unsafe code is documented and justified
- Memory safety guaranteed through Rust's type system
- Lock-free algorithms prevent deadlocks
- NUMA-aware allocation prevents false sharing

## Future Optimizations

1. **SIMD Instructions**: Vectorized operations for batch processing
2. **Custom Allocators**: NUMA-aware memory allocation
3. **Hardware Prefetching**: Cache optimization hints
4. **Real-time Scheduling**: Priority-based task scheduling
5. **GPU Acceleration**: Offload compute-intensive tasks
