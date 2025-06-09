# TallyIO Production-Ready Rust Guidelines

**10/10 Standard za finančne aplikacije z <1ms latenco**

## 🎯 **CORE PRINCIPLES**

### **1. Error Handling Strategy**
```rust
// ✅ Konstruktorji - lahko failajo zaradi validacije
impl Module {
    pub fn new(config: Config) -> Result<Self, ModuleError> {
        config.validate()?;
        Ok(Self { config })
    }
}

// ✅ Operations - lahko failajo
impl Module {
    pub async fn process(&self, data: &Data) -> Result<Output, ModuleError> {
        // Business logic
    }
}

// ✅ Getters - infallible
impl Module {
    pub fn config(&self) -> &Config { &self.config }
    pub fn status(&self) -> Status { self.status }
}
```

### **2. Module Template**
```rust
//! Module description

use std::sync::Arc;
use dashmap::DashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModuleError {
    #[error("Operation failed: {reason}")]
    OperationFailed { reason: String },
    
    #[error("Invalid input: {field}")]
    InvalidInput { field: String },
}

pub type ModuleResult<T> = Result<T, ModuleError>;

pub struct Module {
    config: Arc<Config>,
    cache: DashMap<Key, Value>,
    counter: AtomicU64,
}

impl Module {
    /// Creates new instance
    /// 
    /// # Errors
    /// Returns error if config validation fails
    pub fn new(config: Config) -> ModuleResult<Self> {
        config.validate()?;
        
        Ok(Self {
            config: Arc::new(config),
            cache: DashMap::new(),
            counter: AtomicU64::new(0),
        })
    }
    
    /// Critical path function - MUST be <1ms
    #[inline(always)]
    pub fn critical_operation(&self, input: &Input) -> ModuleResult<Output> {
        // Validate input
        if !input.is_valid() {
            return Err(ModuleError::InvalidInput { 
                field: "input".to_string() 
            });
        }
        
        // Process with NO allocations
        let result = self.process_no_alloc(input)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_critical_operation() -> ModuleResult<()> {
        let module = Module::new(Config::default())?;
        let input = Input::default();
        
        let start = std::time::Instant::now();
        let result = module.critical_operation(&input)?;
        let elapsed = start.elapsed();
        
        assert!(elapsed.as_millis() < 1, "Too slow: {:?}", elapsed);
        assert!(result.is_valid());
        
        Ok(())
    }
}
```

## 🚨 **ABSOLUTE RULES**

### **FORBIDDEN**
- `.unwrap()` - NEVER
- `.expect()` - NEVER  
- `panic!()` - NEVER
- `todo!()` - NEVER
- `unimplemented!()` - NEVER

### **REQUIRED**
- `Result<T, E>` za sve operations
- `thiserror` za error types
- `#[inline(always)]` za critical paths
- Comprehensive tests z performance validation

## ⚡ **PERFORMANCE PATTERNS**

### **Lock-Free Concurrency**
```rust
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct HighPerformanceCache<K, V> {
    data: Arc<DashMap<K, V>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<K, V> HighPerformanceCache<K, V> 
where 
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    #[inline(always)]
    pub fn get(&self, key: &K) -> Option<V> {
        match self.data.get(key) {
            Some(value) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(value.clone())
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }
}
```

### **Zero-Allocation Hot Paths**
```rust
pub struct HotPath {
    buffer: [u8; 1024],  // Pre-allocated
    scratch: Vec<u8>,    // Reusable
}

impl HotPath {
    #[inline(always)]
    pub fn process(&mut self, data: &[u8]) -> ModuleResult<&[u8]> {
        let len = data.len().min(self.buffer.len());
        self.buffer[..len].copy_from_slice(&data[..len]);
        
        // Process in-place - NO allocations
        self.process_in_place(&mut self.buffer[..len])?;
        
        Ok(&self.buffer[..len])
    }
}
```

## 🔒 **SECURITY PATTERNS**

### **Input Validation**
```rust
use garde::{Validate, Report};

#[derive(Validate)]
pub struct CreateUserRequest {
    #[garde(email)]
    pub email: String,
    
    #[garde(length(min = 8, max = 128))]
    pub password: String,
    
    #[garde(range(min = 13, max = 120))]
    pub age: u32,
}

pub async fn create_user(request: CreateUserRequest) -> ModuleResult<User> {
    request.validate(&())?;
    // Process validated input
}
```

## 📊 **TESTING EXCELLENCE**

### **Performance Testing**
```rust
#[test]
fn test_latency_requirement() -> ModuleResult<()> {
    let module = Module::new(Config::default())?;
    let input = Input::default();
    
    let start = std::time::Instant::now();
    let _result = module.critical_operation(&input)?;
    let elapsed = start.elapsed();
    
    assert!(elapsed.as_millis() < 1, "Latency violation: {:?}", elapsed);
    Ok(())
}
```

---

**Sledite tem vzorcem za 10/10 production-ready kod.**
