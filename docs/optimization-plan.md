# TallyIO Performance Optimization Implementation Plan

## 🚨 FAZA 1: KRITIČNE OPTIMIZACIJE (Teden 1-2)

### 1.1 Circuit Breaker Enhancement
**Lokacija:** `crates/core/src/engine/circuit_breaker.rs`
**Prioriteta:** KRITIČNA
**Časovni okvir:** 3 dni

#### Implementacija:
```rust
// Enhanced circuit breaker with latency monitoring
pub struct LatencyCircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU64,
    last_failure_time: AtomicU64,
    latency_threshold: Duration,
    latency_window: RingBuffer<Duration>,
    success_threshold: u32,
}

impl LatencyCircuitBreaker {
    pub fn execute_with_latency_check<F, R>(&self, operation: F) -> Result<R, CircuitBreakerError>
    where F: FnOnce() -> Result<R, Box<dyn Error>>
    {
        let start = Instant::now();
        let result = self.execute(operation)?;
        let latency = start.elapsed();
        
        if latency > self.latency_threshold {
            self.record_latency_violation(latency);
        }
        
        Ok(result)
    }
}
```

#### Akcijski koraki:
1. **Dan 1:** Razširitev obstoječega circuit breaker-ja z latency monitoring
2. **Dan 2:** Implementacija adaptive threshold-ov
3. **Dan 3:** Integracija v kritične poti (mempool, state operations)

#### Testiranje:
- Unit testi za različne latency scenarije
- Load testing z simuliranimi spike-i
- Benchmark validacija <1ms zahteve

---

### 1.2 Batch Processing Implementation
**Lokacija:** `crates/core/src/engine/batch_processor.rs`
**Prioriteta:** KRITIČNA
**Časovni okvir:** 4 dni

#### Implementacija:
```rust
pub struct BatchProcessor<T> {
    batch_size: usize,
    batch_timeout: Duration,
    buffer: LockFreeRingBuffer<T>,
    processor: Arc<dyn Fn(Vec<T>) -> Result<Vec<ProcessResult>, ProcessError>>,
}

impl<T> BatchProcessor<T> {
    pub async fn process_item(&self, item: T) -> Result<ProcessResult, ProcessError> {
        // Add to batch buffer
        self.buffer.push(item)?;
        
        // Check if batch is ready
        if self.buffer.len() >= self.batch_size || self.should_flush_timeout() {
            self.flush_batch().await
        } else {
            self.wait_for_batch_completion().await
        }
    }
    
    async fn flush_batch(&self) -> Result<ProcessResult, ProcessError> {
        let batch = self.buffer.drain_batch(self.batch_size);
        let start = Instant::now();
        
        let results = (self.processor)(batch)?;
        
        // Ensure batch processing stays under latency budget
        let elapsed = start.elapsed();
        if elapsed > Duration::from_micros(800) { // 800µs budget
            warn!("Batch processing exceeded latency budget: {:?}", elapsed);
        }
        
        Ok(results)
    }
}
```

#### Akcijski koraki:
1. **Dan 1-2:** Implementacija core batch processor strukture
2. **Dan 3:** Integracija v mempool operations
3. **Dan 4:** Integracija v state management operations

#### Optimizacija parametrov:
- **Batch size:** 16-64 items (optimalno za cache line)
- **Timeout:** 100-200µs (agresivno za low latency)
- **Buffer size:** 1024 items (power of 2)

---

### 1.3 CPU Affinity Implementation
**Lokacija:** `crates/core/src/optimization/cpu_affinity.rs`
**Prioriteta:** KRITIČNA
**Časovni okvir:** 3 dni

#### Implementacija:
```rust
pub struct CpuAffinityManager {
    critical_cores: Vec<usize>,
    worker_cores: Vec<usize>,
    isolation_enabled: bool,
}

impl CpuAffinityManager {
    pub fn pin_critical_thread(&self, thread_type: CriticalThreadType) -> Result<(), AffinityError> {
        let core_id = match thread_type {
            CriticalThreadType::MevScanner => self.critical_cores[0],
            CriticalThreadType::StateManager => self.critical_cores[1],
            CriticalThreadType::MempoolProcessor => self.critical_cores[2],
        };
        
        core_affinity::set_for_current(core_affinity::CoreId { id: core_id })?;
        
        // Set real-time priority for critical threads
        self.set_realtime_priority()?;
        
        Ok(())
    }
    
    fn set_realtime_priority(&self) -> Result<(), AffinityError> {
        #[cfg(target_os = "linux")]
        {
            use libc::{sched_setscheduler, SCHED_FIFO, sched_param};
            let param = sched_param { sched_priority: 99 };
            unsafe {
                sched_setscheduler(0, SCHED_FIFO, &param);
            }
        }
        Ok(())
    }
}
```

#### Akcijski koraki:
1. **Dan 1:** CPU topology detection in core assignment
2. **Dan 2:** Thread pinning implementation
3. **Dan 3:** Integration z engine startup

#### CPU Layout (16-core sistem):
- **Cores 0-3:** Critical threads (MEV, State, Mempool, Network)
- **Cores 4-11:** Worker threads
- **Cores 12-15:** OS in background tasks

---

## 📊 **FAZA 2: IZBOLJŠAVE** (Teden 3)

### 2.1 NUMA-Aware Scheduling
**Lokacija:** `crates/core/src/engine/numa.rs`
**Prioriteta:** VISOKA
**Časovni okvir:** 4 dni

#### Implementacija:
```rust
pub struct NumaScheduler {
    nodes: Vec<NumaNode>,
    thread_pool: Vec<WorkerPool>,
    memory_pools: HashMap<usize, MemoryPool>,
}

impl NumaScheduler {
    pub fn schedule_task(&self, task: Task) -> Result<(), SchedulingError> {
        let optimal_node = self.find_optimal_numa_node(&task)?;
        let worker_pool = &self.thread_pool[optimal_node];
        
        // Ensure memory allocation on same NUMA node
        let memory = self.memory_pools[&optimal_node].allocate(task.memory_requirement())?;
        
        worker_pool.submit_with_memory(task, memory)
    }
}
```

### 2.2 Memory Pool Pre-allocation
**Lokacija:** `crates/core/src/memory/pool.rs`
**Prioriteta:** VISOKA
**Časovni okvir:** 3 dni

#### Implementacija:
```rust
pub struct PreAllocatedMemoryPool {
    small_blocks: LockFreeStack<MemoryBlock<64>>,    // 64B blocks
    medium_blocks: LockFreeStack<MemoryBlock<1024>>, // 1KB blocks
    large_blocks: LockFreeStack<MemoryBlock<4096>>,  // 4KB blocks
    huge_pages: LockFreeStack<HugePage>,             // 2MB pages
}

impl PreAllocatedMemoryPool {
    pub fn new() -> Self {
        let mut pool = Self::default();
        
        // Pre-allocate pools at startup
        pool.pre_allocate_blocks(1000, BlockSize::Small);
        pool.pre_allocate_blocks(500, BlockSize::Medium);
        pool.pre_allocate_blocks(100, BlockSize::Large);
        pool.pre_allocate_huge_pages(10);
        
        pool
    }
    
    #[inline(always)]
    pub fn allocate_fast(&self, size: usize) -> Option<*mut u8> {
        match size {
            0..=64 => self.small_blocks.pop().map(|b| b.as_ptr()),
            65..=1024 => self.medium_blocks.pop().map(|b| b.as_ptr()),
            1025..=4096 => self.large_blocks.pop().map(|b| b.as_ptr()),
            _ => self.allocate_huge(size),
        }
    }
}
```

---

## 🎛️ **FAZA 3: FINE-TUNING** (Teden 4)

### 3.1 Jemalloc Tuning
**Lokacija:** `Cargo.toml` in `src/main.rs`

#### Konfiguracija:
```toml
[dependencies]
jemallocator = { version = "0.5", features = ["profiling", "stats"] }

[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```rust
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

// Jemalloc tuning za ultra-low latency
fn configure_jemalloc() {
    std::env::set_var("MALLOC_CONF", 
        "background_thread:true,\
         metadata_thp:auto,\
         dirty_decay_ms:1000,\
         muzzy_decay_ms:1000,\
         narenas:4");
}
```

### 3.2 Cache Prefetching
**Lokacija:** `crates/core/src/optimization/prefetch.rs`

#### Implementacija:
```rust
#[inline(always)]
pub fn prefetch_for_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

pub struct PrefetchingIterator<T> {
    data: *const T,
    len: usize,
    pos: usize,
    prefetch_distance: usize,
}

impl<T> Iterator for PrefetchingIterator<T> {
    type Item = &'static T;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        
        // Prefetch next cache line
        if self.pos + self.prefetch_distance < self.len {
            unsafe {
                let prefetch_ptr = self.data.add(self.pos + self.prefetch_distance);
                prefetch_for_read(prefetch_ptr);
            }
        }
        
        unsafe {
            let item = &*self.data.add(self.pos);
            self.pos += 1;
            Some(item)
        }
    }
}
```

### 3.3 SIMD Optimizations
**Lokacija:** `crates/core/src/optimization/simd.rs`

#### Implementacija:
```rust
#[cfg(target_feature = "avx2")]
pub mod avx2_ops {
    use std::arch::x86_64::*;
    
    #[inline(always)]
    pub unsafe fn hash_batch_avx2(data: &[u64; 4]) -> [u64; 4] {
        let input = _mm256_loadu_si256(data.as_ptr() as *const __m256i);
        
        // Fast hash using AVX2 instructions
        let hash1 = _mm256_mullo_epi32(input, _mm256_set1_epi32(0x9e3779b9));
        let hash2 = _mm256_xor_si256(hash1, _mm256_srli_epi32(hash1, 16));
        
        let mut result = [0u64; 4];
        _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, hash2);
        result
    }
    
    #[inline(always)]
    pub unsafe fn compare_batch_avx2(a: &[u64; 4], b: &[u64; 4]) -> u32 {
        let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
        
        let cmp = _mm256_cmpeq_epi64(va, vb);
        _mm256_movemask_pd(_mm256_castsi256_pd(cmp)) as u32
    }
}
```

---

## 📊 **IMPLEMENTACIJSKI ČASOVNI NAČRT**

| Teden | Faza | Naloge | Pričakovani rezultat |
|-------|------|--------|---------------------|
| **1** | Kritično | Circuit Breaker + Batch Processing (del) | 50% zmanjšanje latency spike-ov |
| **2** | Kritično | Batch Processing + CPU Affinity | 90% operacij pod 1ms |
| **3** | Izboljšave | NUMA + Memory Pool + Lock-free | 30% izboljšanje throughput-a |
| **4** | Fine-tuning | Jemalloc + Cache + SIMD | 10-20% dodatne optimizacije |

---

## 🎯 **MERITVE USPEHA**

### Kritični KPI-ji:
- **Latency P99:** <1ms (trenutno: ~5-50ms)
- **Latency P95:** <500µs
- **Latency P50:** <100µs
- **Throughput:** >100k ops/sec
- **Memory usage:** <2GB steady state

### Benchmark ciljne vrednosti:
```
ultra_low_latency/single_enqueue_dequeue: <50ns ✅
ultra_low_latency/cache_get_operation: <60ns ✅
latency_under_load/load_factor/1.0: <1ms ❌ (trenutno: ~1.07ms)
latency_under_load/load_factor/5.0: <1ms ❌ (trenutno: ~37ms)
concurrent_operations/threads/8: <1ms ❌ (trenutno: ~1.08ms)
```

---

## 🔧 **IMPLEMENTACIJSKI PRISTOP**

### 1. Iterativni razvoj:
- Implementacija po fazah z vmesnim testiranjem
- Benchmark validacija po vsaki fazi
- Rollback plan za vsako optimizacijo

### 2. A/B testiranje:
- Primerjava performance pred/po optimizaciji
- Regression testing za obstoječo funkcionalnost
- Load testing v production-like okolju

### 3. Monitoring:
- Real-time latency monitoring
- Memory usage tracking
- CPU utilization analysis
- Throughput metrics

---

## 🚨 **TVEGANJA IN MITIGACIJE**

| Tveganje | Verjetnost | Vpliv | Mitigacija |
|----------|------------|-------|------------|
| Regression v performance | Srednja | Visok | Obsežno benchmark testiranje |
| Kompleksnost implementacije | Visoka | Srednji | Fazni pristop z rollback |
| Platform-specific issues | Nizka | Visok | Cross-platform testiranje |
| Memory leaks | Srednja | Visok | Valgrind/AddressSanitizer |

---

## 📋 **NASLEDNJI KORAKI**

1. **Takoj:** Začetek implementacije Circuit Breaker-ja
2. **Dan 3:** Prva benchmark validacija
3. **Teden 2:** Vmesna ocena napredka
4. **Teden 4:** Finalna validacija in production deployment

**Odgovorna oseba:** Senior Rust Developer  
**Review:** Tedensko s tehničnim vodjem  
**Deadline:** 4 tedne od začetka implementacije
