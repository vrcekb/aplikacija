//! Memory Management - Ultra-Performance Memory Pool for Financial Applications
//!
//! Implements enterprise-grade memory management with:
//! - Custom memory pools for zero-allocation task processing
//! - Memory pressure monitoring and adaptive allocation
//! - NUMA-aware memory placement for optimal cache locality
//! - Real-time memory usage tracking and leak detection
//! - Financial-grade robustness with comprehensive error handling
//!
//! This module provides the foundation for ultra-low latency memory operations
//! required in high-frequency trading and MEV applications.

pub mod allocator;
pub mod pool;
pub mod pressure;
pub mod stats;

pub use allocator::{AllocatorError, AllocatorStats, NumaAllocator};
pub use pool::{MemoryPool, MemoryPoolConfig, MemoryPoolError};
pub use pressure::{MemoryPressureMonitor, PressureLevel, PressureThreshold};
pub use stats::{MemoryMetrics, MemoryStats, MemoryUsage};
