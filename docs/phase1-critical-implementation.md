# FAZA 1: Kritične Optimizacije - Implementacijski Vodič

## 🚨 Circuit Breaker Enhancement

### Datoteka: `crates/core/src/engine/circuit_breaker.rs`

```rust
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crate::lockfree::ring_buffer::LockFreeRingBuffer;

const CIRCUIT_CLOSED: u8 = 0;
const CIRCUIT_OPEN: u8 = 1;
const CIRCUIT_HALF_OPEN: u8 = 2;

pub struct LatencyCircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure_time: AtomicU64,
    last_success_time: AtomicU64,
    
    // Latency monitoring
    latency_threshold: Duration,
    latency_violations: AtomicU64,
    latency_window: LockFreeRingBuffer<u64>, // Store latencies in nanoseconds
    
    // Configuration
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
    latency_violation_threshold: u32,
}

impl LatencyCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: AtomicU8::new(CIRCUIT_CLOSED),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure_time: AtomicU64::new(0),
            last_success_time: AtomicU64::new(0),
            
            latency_threshold: config.latency_threshold,
            latency_violations: AtomicU64::new(0),
            latency_window: LockFreeRingBuffer::new(1000), // Last 1000 operations
            
            failure_threshold: config.failure_threshold,
            success_threshold: config.success_threshold,
            timeout: config.timeout,
            latency_violation_threshold: config.latency_violation_threshold,
        }
    }
    
    #[inline(always)]
    pub fn execute_with_latency_check<F, R, E>(&self, operation: F) -> Result<R, CircuitBreakerError>
    where 
        F: FnOnce() -> Result<R, E>,
        E: std::error::Error + Send + Sync + 'static,
    {
        // Check circuit state
        if !self.can_execute()? {
            return Err(CircuitBreakerError::CircuitOpen);
        }
        
        let start = Instant::now();
        let result = operation();
        let latency = start.elapsed();
        
        // Record latency
        self.record_latency(latency);
        
        match result {
            Ok(value) => {
                self.on_success(latency);
                Ok(value)
            }
            Err(error) => {
                self.on_failure(latency);
                Err(CircuitBreakerError::OperationFailed(Box::new(error)))
            }
        }
    }
    
    #[inline(always)]
    fn can_execute(&self) -> Result<bool, CircuitBreakerError> {
        match self.state.load(Ordering::Acquire) {
            CIRCUIT_CLOSED => Ok(true),
            CIRCUIT_OPEN => {
                if self.should_attempt_reset() {
                    self.transition_to_half_open();
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            CIRCUIT_HALF_OPEN => Ok(true),
            _ => unreachable!(),
        }
    }
    
    #[inline(always)]
    fn record_latency(&self, latency: Duration) {
        let latency_ns = latency.as_nanos() as u64;
        let _ = self.latency_window.push(latency_ns);
        
        if latency > self.latency_threshold {
            self.latency_violations.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    fn on_success(&self, latency: Duration) {
        self.last_success_time.store(
            Instant::now().elapsed().as_nanos() as u64,
            Ordering::Relaxed
        );
        
        let current_state = self.state.load(Ordering::Acquire);
        
        match current_state {
            CIRCUIT_HALF_OPEN => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if success_count >= self.success_threshold as u64 {
                    self.transition_to_closed();
                }
            }
            CIRCUIT_CLOSED => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            _ => {}
        }
    }
    
    fn on_failure(&self, latency: Duration) {
        self.last_failure_time.store(
            Instant::now().elapsed().as_nanos() as u64,
            Ordering::Relaxed
        );
        
        let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        let latency_violations = self.latency_violations.load(Ordering::Relaxed);
        
        // Open circuit if too many failures or latency violations
        if failure_count >= self.failure_threshold as u64 || 
           latency_violations >= self.latency_violation_threshold as u64 {
            self.transition_to_open();
        }
    }
    
    fn should_attempt_reset(&self) -> bool {
        let now = Instant::now().elapsed().as_nanos() as u64;
        let last_failure = self.last_failure_time.load(Ordering::Relaxed);
        
        now - last_failure >= self.timeout.as_nanos() as u64
    }
    
    fn transition_to_closed(&self) {
        self.state.store(CIRCUIT_CLOSED, Ordering::Release);
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.latency_violations.store(0, Ordering::Relaxed);
    }
    
    fn transition_to_open(&self) {
        self.state.store(CIRCUIT_OPEN, Ordering::Release);
        self.success_count.store(0, Ordering::Relaxed);
    }
    
    fn transition_to_half_open(&self) {
        self.state.store(CIRCUIT_HALF_OPEN, Ordering::Release);
        self.success_count.store(0, Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> CircuitBreakerStats {
        let latencies: Vec<u64> = self.latency_window.iter().collect();
        let avg_latency = if latencies.is_empty() {
            0
        } else {
            latencies.iter().sum::<u64>() / latencies.len() as u64
        };
        
        CircuitBreakerStats {
            state: match self.state.load(Ordering::Acquire) {
                CIRCUIT_CLOSED => "CLOSED",
                CIRCUIT_OPEN => "OPEN", 
                CIRCUIT_HALF_OPEN => "HALF_OPEN",
                _ => "UNKNOWN",
            },
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            latency_violations: self.latency_violations.load(Ordering::Relaxed),
            avg_latency_ns: avg_latency,
        }
    }
}

#[derive(Debug)]
pub struct CircuitBreakerConfig {
    pub latency_threshold: Duration,
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
    pub latency_violation_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            latency_threshold: Duration::from_millis(1), // 1ms threshold
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
            latency_violation_threshold: 10,
        }
    }
}

#[derive(Debug)]
pub struct CircuitBreakerStats {
    pub state: &'static str,
    pub failure_count: u64,
    pub success_count: u64,
    pub latency_violations: u64,
    pub avg_latency_ns: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError {
    #[error("Circuit breaker is open")]
    CircuitOpen,
    #[error("Operation failed: {0}")]
    OperationFailed(Box<dyn std::error::Error + Send + Sync>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_latency_circuit_breaker() {
        let config = CircuitBreakerConfig {
            latency_threshold: Duration::from_millis(1),
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            latency_violation_threshold: 2,
        };
        
        let breaker = LatencyCircuitBreaker::new(config);
        
        // Test successful operation
        let result = breaker.execute_with_latency_check(|| {
            Ok::<_, std::io::Error>("success")
        });
        assert!(result.is_ok());
        
        // Test latency violation
        let result = breaker.execute_with_latency_check(|| {
            thread::sleep(Duration::from_millis(2)); // Exceed threshold
            Ok::<_, std::io::Error>("slow")
        });
        assert!(result.is_ok()); // Should still succeed but record violation
        
        // Another latency violation should open circuit
        let result = breaker.execute_with_latency_check(|| {
            thread::sleep(Duration::from_millis(2));
            Ok::<_, std::io::Error>("slow")
        });
        
        // Next operation should fail due to open circuit
        let result = breaker.execute_with_latency_check(|| {
            Ok::<_, std::io::Error>("should_fail")
        });
        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen)));
    }
}
```

## 🔄 Batch Processing Implementation

### Datoteka: `crates/core/src/engine/batch_processor.rs`

```rust
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::timeout;
use crate::lockfree::ring_buffer::LockFreeRingBuffer;

pub struct BatchProcessor<T, R> {
    batch_size: usize,
    batch_timeout: Duration,
    max_latency: Duration,
    
    // Internal state
    buffer: Arc<LockFreeRingBuffer<BatchItem<T, R>>>,
    processor: Arc<dyn Fn(Vec<T>) -> Result<Vec<R>, BatchError> + Send + Sync>,
    
    // Channels for coordination
    flush_tx: mpsc::UnboundedSender<FlushCommand>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

struct BatchItem<T, R> {
    item: T,
    response_tx: oneshot::Sender<Result<R, BatchError>>,
    timestamp: Instant,
}

enum FlushCommand {
    Force,
    Timeout,
}

impl<T, R> BatchProcessor<T, R> 
where 
    T: Send + 'static,
    R: Send + 'static,
{
    pub fn new<F>(
        batch_size: usize,
        batch_timeout: Duration,
        max_latency: Duration,
        processor: F,
    ) -> Self 
    where 
        F: Fn(Vec<T>) -> Result<Vec<R>, BatchError> + Send + Sync + 'static,
    {
        let buffer = Arc::new(LockFreeRingBuffer::new(batch_size * 4));
        let (flush_tx, flush_rx) = mpsc::unbounded_channel();
        
        let processor = Arc::new(processor);
        
        // Spawn background processor
        let buffer_clone = buffer.clone();
        let processor_clone = processor.clone();
        tokio::spawn(Self::background_processor(
            buffer_clone,
            processor_clone,
            batch_size,
            batch_timeout,
            max_latency,
            flush_rx,
        ));
        
        Self {
            batch_size,
            batch_timeout,
            max_latency,
            buffer,
            processor,
            flush_tx,
            shutdown_tx: None,
        }
    }
    
    pub async fn process(&self, item: T) -> Result<R, BatchError> {
        let (response_tx, response_rx) = oneshot::channel();
        
        let batch_item = BatchItem {
            item,
            response_tx,
            timestamp: Instant::now(),
        };
        
        // Add to buffer
        self.buffer.push(batch_item)
            .map_err(|_| BatchError::BufferFull)?;
        
        // Check if we should force flush
        if self.buffer.len() >= self.batch_size {
            let _ = self.flush_tx.send(FlushCommand::Force);
        }
        
        // Wait for response with timeout
        timeout(self.max_latency, response_rx)
            .await
            .map_err(|_| BatchError::Timeout)?
            .map_err(|_| BatchError::ChannelClosed)?
    }
    
    async fn background_processor(
        buffer: Arc<LockFreeRingBuffer<BatchItem<T, R>>>,
        processor: Arc<dyn Fn(Vec<T>) -> Result<Vec<R>, BatchError> + Send + Sync>,
        batch_size: usize,
        batch_timeout: Duration,
        max_latency: Duration,
        mut flush_rx: mpsc::UnboundedReceiver<FlushCommand>,
    ) {
        let mut timeout_interval = tokio::time::interval(batch_timeout);
        
        loop {
            tokio::select! {
                _ = timeout_interval.tick() => {
                    Self::process_batch(&buffer, &processor, batch_size, max_latency).await;
                }
                
                command = flush_rx.recv() => {
                    match command {
                        Some(FlushCommand::Force) | Some(FlushCommand::Timeout) => {
                            Self::process_batch(&buffer, &processor, batch_size, max_latency).await;
                        }
                        None => break, // Channel closed, shutdown
                    }
                }
            }
        }
    }
    
    async fn process_batch(
        buffer: &LockFreeRingBuffer<BatchItem<T, R>>,
        processor: &Arc<dyn Fn(Vec<T>) -> Result<Vec<R>, BatchError> + Send + Sync>,
        batch_size: usize,
        max_latency: Duration,
    ) {
        if buffer.is_empty() {
            return;
        }
        
        let start_time = Instant::now();
        let mut batch_items = Vec::with_capacity(batch_size);
        
        // Drain items from buffer
        while batch_items.len() < batch_size {
            if let Some(item) = buffer.pop() {
                // Check if item is too old
                if start_time.duration_since(item.timestamp) > max_latency {
                    let _ = item.response_tx.send(Err(BatchError::Timeout));
                    continue;
                }
                batch_items.push(item);
            } else {
                break;
            }
        }
        
        if batch_items.is_empty() {
            return;
        }
        
        // Extract items for processing
        let items: Vec<T> = batch_items.iter()
            .map(|bi| unsafe { std::ptr::read(&bi.item) })
            .collect();
        
        // Process batch
        let process_start = Instant::now();
        let results = processor(items);
        let process_duration = process_start.elapsed();
        
        // Send responses
        match results {
            Ok(results) => {
                for (batch_item, result) in batch_items.into_iter().zip(results.into_iter()) {
                    let _ = batch_item.response_tx.send(Ok(result));
                }
            }
            Err(error) => {
                for batch_item in batch_items {
                    let _ = batch_item.response_tx.send(Err(error.clone()));
                }
            }
        }
        
        // Log performance metrics
        if process_duration > Duration::from_micros(800) {
            tracing::warn!(
                "Batch processing exceeded latency budget: {:?} (items: {})",
                process_duration,
                items.len()
            );
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum BatchError {
    #[error("Buffer is full")]
    BufferFull,
    #[error("Operation timed out")]
    Timeout,
    #[error("Channel closed")]
    ChannelClosed,
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),
}
```

## ⚡ CPU Affinity Implementation

### Datoteka: `crates/core/src/optimization/cpu_affinity.rs`

```rust
use std::collections::HashMap;
use core_affinity::{CoreId, get_core_ids};

#[derive(Debug, Clone, Copy)]
pub enum CriticalThreadType {
    MevScanner,
    StateManager,
    MempoolProcessor,
    NetworkHandler,
    MetricsCollector,
}

pub struct CpuAffinityManager {
    critical_cores: HashMap<CriticalThreadType, usize>,
    worker_cores: Vec<usize>,
    total_cores: usize,
    isolation_enabled: bool,
}

impl CpuAffinityManager {
    pub fn new() -> Result<Self, AffinityError> {
        let core_ids = get_core_ids().ok_or(AffinityError::CoreDetectionFailed)?;
        let total_cores = core_ids.len();
        
        if total_cores < 4 {
            return Err(AffinityError::InsufficientCores);
        }
        
        // Reserve first 4 cores for critical threads
        let mut critical_cores = HashMap::new();
        critical_cores.insert(CriticalThreadType::MevScanner, 0);
        critical_cores.insert(CriticalThreadType::StateManager, 1);
        critical_cores.insert(CriticalThreadType::MempoolProcessor, 2);
        critical_cores.insert(CriticalThreadType::NetworkHandler, 3);
        
        // Remaining cores for workers
        let worker_cores: Vec<usize> = (4..total_cores.saturating_sub(2)).collect();
        
        Ok(Self {
            critical_cores,
            worker_cores,
            total_cores,
            isolation_enabled: true,
        })
    }
    
    pub fn pin_critical_thread(&self, thread_type: CriticalThreadType) -> Result<(), AffinityError> {
        let core_id = self.critical_cores.get(&thread_type)
            .ok_or(AffinityError::ThreadTypeNotFound)?;
        
        let core = CoreId { id: *core_id };
        core_affinity::set_for_current(core);
        
        // Set high priority for critical threads
        self.set_high_priority()?;
        
        tracing::info!("Pinned {:?} to core {}", thread_type, core_id);
        Ok(())
    }
    
    pub fn pin_worker_thread(&self, worker_id: usize) -> Result<(), AffinityError> {
        if self.worker_cores.is_empty() {
            return Err(AffinityError::NoWorkerCoresAvailable);
        }
        
        let core_id = self.worker_cores[worker_id % self.worker_cores.len()];
        let core = CoreId { id: core_id };
        core_affinity::set_for_current(core);
        
        tracing::debug!("Pinned worker {} to core {}", worker_id, core_id);
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn set_high_priority(&self) -> Result<(), AffinityError> {
        use libc::{sched_setscheduler, SCHED_FIFO, sched_param};
        
        let param = sched_param { sched_priority: 50 }; // High but not max priority
        
        unsafe {
            if sched_setscheduler(0, SCHED_FIFO, &param) != 0 {
                return Err(AffinityError::PrioritySetFailed);
            }
        }
        
        Ok(())
    }
    
    #[cfg(not(target_os = "linux"))]
    fn set_high_priority(&self) -> Result<(), AffinityError> {
        // Platform-specific implementation for Windows/macOS
        tracing::warn!("High priority setting not implemented for this platform");
        Ok(())
    }
    
    pub fn get_topology_info(&self) -> TopologyInfo {
        TopologyInfo {
            total_cores: self.total_cores,
            critical_cores: self.critical_cores.clone(),
            worker_cores: self.worker_cores.clone(),
            isolation_enabled: self.isolation_enabled,
        }
    }
}

#[derive(Debug)]
pub struct TopologyInfo {
    pub total_cores: usize,
    pub critical_cores: HashMap<CriticalThreadType, usize>,
    pub worker_cores: Vec<usize>,
    pub isolation_enabled: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum AffinityError {
    #[error("Failed to detect CPU cores")]
    CoreDetectionFailed,
    #[error("Insufficient CPU cores for optimal performance")]
    InsufficientCores,
    #[error("Thread type not found in configuration")]
    ThreadTypeNotFound,
    #[error("No worker cores available")]
    NoWorkerCoresAvailable,
    #[error("Failed to set thread priority")]
    PrioritySetFailed,
}
```

## 📋 Integracija v Engine

### Datoteka: `crates/core/src/engine/mod.rs`

```rust
// Dodaj v TallyEngine struct
pub struct TallyEngine {
    // ... existing fields
    circuit_breaker: Arc<LatencyCircuitBreaker>,
    batch_processor: Arc<BatchProcessor<Transaction, ProcessResult>>,
    cpu_affinity: Arc<CpuAffinityManager>,
}

impl TallyEngine {
    pub fn new(config: EngineConfig) -> Result<Self, EngineError> {
        // Initialize CPU affinity first
        let cpu_affinity = Arc::new(CpuAffinityManager::new()?);
        
        // Pin main engine thread
        cpu_affinity.pin_critical_thread(CriticalThreadType::StateManager)?;
        
        // Initialize circuit breaker
        let circuit_breaker = Arc::new(LatencyCircuitBreaker::new(
            config.circuit_breaker_config
        ));
        
        // Initialize batch processor
        let batch_processor = Arc::new(BatchProcessor::new(
            config.batch_size,
            config.batch_timeout,
            config.max_latency,
            |transactions| Self::process_transaction_batch(transactions),
        ));
        
        Ok(Self {
            // ... existing initialization
            circuit_breaker,
            batch_processor,
            cpu_affinity,
        })
    }
    
    pub async fn process_transaction_with_optimizations(
        &self,
        transaction: Transaction,
    ) -> Result<ProcessResult, EngineError> {
        // Use circuit breaker and batch processor
        self.circuit_breaker.execute_with_latency_check(|| {
            self.batch_processor.process(transaction)
        }).await
    }
}
```
