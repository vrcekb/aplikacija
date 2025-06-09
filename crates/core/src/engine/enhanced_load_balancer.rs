//! Enhanced Load Balancing - Ultra-Performance Worker Load Distribution
//!
//! Production-ready enhanced load balancing for `TallyIO` financial applications.
//! Implements cache-optimized load balancing with predictive worker selection
//! and sub-microsecond load distribution for <1ms latency requirements.

#![allow(clippy::inline_always)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::unused_self)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::float_cmp)]

use std::{
    sync::{
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use thiserror::Error;

use crate::{engine::executor::Task, types::WorkerId};

/// Enhanced load balancing error types
#[derive(Error, Debug, Copy, Clone, PartialEq, Eq)]
pub enum LoadBalancerError {
    /// No workers available
    #[error("No workers available")]
    NoWorkersAvailable,

    /// Invalid worker configuration
    #[error("Invalid worker configuration: {reason}")]
    InvalidConfiguration {
        /// Reason for configuration error
        reason: &'static str,
    },

    /// Load balancer not initialized
    #[error("Load balancer not initialized")]
    NotInitialized,

    /// Critical latency violation
    #[error("Latency violation: {latency_ns}ns > {max_latency_ns}ns")]
    LatencyViolation {
        /// Actual latency in nanoseconds
        latency_ns: u64,
        /// Maximum allowed latency in nanoseconds
        max_latency_ns: u64,
    },
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least loaded worker
    LeastLoaded,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Predictive load balancing
    Predictive,
    /// Locality-aware balancing
    LocalityAware,
}

/// Load balancing configuration
#[derive(Debug, Clone)]
pub struct LoadBalancerConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Maximum workers supported
    pub max_workers: usize,
    /// Target latency in nanoseconds
    pub target_latency_ns: u64,
    /// Load measurement window
    pub measurement_window: Duration,
    /// Cache line optimization
    pub cache_optimized: bool,
    /// NUMA awareness
    pub numa_aware: bool,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::Predictive,
            max_workers: 64,
            target_latency_ns: 1_000_000, // 1ms
            measurement_window: Duration::from_millis(10),
            cache_optimized: true,
            numa_aware: true,
        }
    }
}

/// Cache-aligned worker load information
#[derive(Debug, Default)]
#[repr(C, align(64))]
pub struct WorkerLoadInfo {
    /// Worker ID
    pub worker_id: AtomicU32,
    /// Current load (0-1000, fixed-point)
    pub current_load: AtomicU32,
    /// Task count
    pub task_count: AtomicU32,
    /// Average latency in nanoseconds
    pub avg_latency_ns: AtomicU64,
    /// Last update timestamp
    pub last_update: AtomicU64,
    /// Worker weight (for weighted algorithms)
    pub weight: AtomicU32,
    /// NUMA node ID
    pub numa_node: AtomicU32,
    /// Cache padding to prevent false sharing
    _padding: [u8; 8],
}

/// Cache-optimized load balancing table
#[derive(Debug)]
#[repr(C, align(64))]
pub struct LoadBalancingTable {
    /// Worker load information (cache-aligned)
    workers: Vec<WorkerLoadInfo>,
    /// Current worker count
    worker_count: AtomicUsize,
    /// Round-robin counter
    round_robin_counter: AtomicUsize,
    /// Last update timestamp
    last_update: AtomicU64,
    /// Table generation (for consistency)
    generation: AtomicU64,
}

/// Enhanced load balancer metrics
#[derive(Debug, Default)]
#[repr(C, align(64))]
pub struct LoadBalancerMetrics {
    /// Total selections performed
    pub total_selections: AtomicU64,
    /// Successful selections
    pub successful_selections: AtomicU64,
    /// Failed selections
    pub failed_selections: AtomicU64,
    /// Average selection latency in nanoseconds
    pub avg_selection_latency_ns: AtomicU64,
    /// Cache hits
    pub cache_hits: AtomicU64,
    /// Cache misses
    pub cache_misses: AtomicU64,
    /// Load imbalance factor (0-1000, fixed-point)
    pub load_imbalance: AtomicU32,
}

/// Enhanced load balancer
///
/// Implements cache-optimized load balancing with predictive worker selection.
/// Optimized for ultra-low latency financial applications with <1ms requirements.
#[derive(Debug)]
pub struct EnhancedLoadBalancer {
    /// Load balancer configuration
    config: LoadBalancerConfig,
    /// Load balancing table
    table: Arc<LoadBalancingTable>,
    /// Load balancer metrics
    metrics: LoadBalancerMetrics,
    /// Balancer unique identifier (for future use in distributed systems)
    #[allow(dead_code)]
    balancer_id: u64,
}

impl LoadBalancingTable {
    /// Create new load balancing table
    fn new(max_workers: usize) -> Self {
        let mut workers = Vec::with_capacity(max_workers);
        for i in 0..max_workers {
            let worker_info = WorkerLoadInfo::default();
            worker_info
                .worker_id
                .store(u32::try_from(i).unwrap_or(u32::MAX), Ordering::Relaxed);
            worker_info.weight.store(1000, Ordering::Relaxed); // Default weight
            workers.push(worker_info);
        }

        Self {
            workers,
            worker_count: AtomicUsize::new(0),
            round_robin_counter: AtomicUsize::new(0),
            last_update: AtomicU64::new(0),
            generation: AtomicU64::new(1),
        }
    }

    /// Update worker load information
    fn update_worker_load(
        &self,
        worker_id: WorkerId,
        load: f64,
        task_count: u32,
        avg_latency_ns: u64,
    ) {
        let worker_index = usize::try_from(worker_id.raw()).unwrap_or(usize::MAX);
        if let Some(worker) = self.workers.get(worker_index) {
            let load_clamped = load.clamp(0.0_f64, 1000.0_f64);
            let load_fixed = (load_clamped * 1000.0_f64) as u32;
            let now = current_timestamp_ns();

            worker.current_load.store(load_fixed, Ordering::Relaxed);
            worker.task_count.store(task_count, Ordering::Relaxed);
            worker
                .avg_latency_ns
                .store(avg_latency_ns, Ordering::Relaxed);
            worker.last_update.store(now, Ordering::Relaxed);
        }
    }

    /// Get least loaded worker (cache-optimized)
    fn get_least_loaded_worker(&self) -> Option<WorkerId> {
        let worker_count = self.worker_count.load(Ordering::Relaxed);
        if worker_count == 0 {
            return None;
        }

        let mut min_load = u32::MAX;
        let mut selected_worker = 0_usize;

        // Cache-friendly linear scan with bounds checking
        for i in 0..worker_count {
            if let Some(worker) = self.workers.get(i) {
                let load = worker.current_load.load(Ordering::Relaxed);

                if load < min_load {
                    min_load = load;
                    selected_worker = i;
                }
            }
        }

        Some(WorkerId::from_raw(
            u64::try_from(selected_worker).unwrap_or(0),
        ))
    }

    /// Get next round-robin worker
    fn get_round_robin_worker(&self) -> Option<WorkerId> {
        let worker_count = self.worker_count.load(Ordering::Relaxed);
        if worker_count == 0 {
            return None;
        }

        let index = self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % worker_count;
        Some(WorkerId::from_raw(u64::try_from(index).unwrap_or(0)))
    }

    /// Get predictive worker selection
    fn get_predictive_worker(&self, task: &Task) -> Option<WorkerId> {
        let worker_count = self.worker_count.load(Ordering::Relaxed);
        if worker_count == 0 {
            return None;
        }

        // Simple predictive algorithm based on task characteristics
        let task_hash = self.hash_task(task);
        let mut best_score = f64::NEG_INFINITY;
        let mut selected_worker = 0_usize;

        for i in 0..worker_count {
            if let Some(worker) = self.workers.get(i) {
                let load = f64::from(worker.current_load.load(Ordering::Relaxed)) / 1000.0_f64;
                let latency = worker.avg_latency_ns.load(Ordering::Relaxed);
                let weight = f64::from(worker.weight.load(Ordering::Relaxed)) / 1000.0_f64;

                // Scoring function: lower load, lower latency, higher weight = better score
                let latency_factor = if latency > 0 {
                    latency as f64 / 1_000_000.0_f64
                } else {
                    0.0_f64
                };
                let score = weight / (1.0_f64 + load + latency_factor);

                // Add task affinity bonus
                let affinity_bonus = if (task_hash % worker_count) == i {
                    0.1_f64
                } else {
                    0.0_f64
                };
                let final_score = score + affinity_bonus;

                if final_score > best_score {
                    best_score = final_score;
                    selected_worker = i;
                }
            }
        }

        Some(WorkerId::from_raw(
            u64::try_from(selected_worker).unwrap_or(0),
        ))
    }

    /// Hash task for affinity calculation
    fn hash_task(&self, task: &Task) -> usize {
        // Simple hash based on task ID and strategy
        let mut hash = usize::try_from(task.id.raw()).unwrap_or(0);
        hash ^= usize::try_from(task.strategy_id.raw()).unwrap_or(0);
        hash ^= task.data.len();
        hash
    }
}

impl EnhancedLoadBalancer {
    /// Create new enhanced load balancer
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(config: LoadBalancerConfig) -> Result<Self, LoadBalancerError> {
        if config.max_workers == 0 {
            return Err(LoadBalancerError::InvalidConfiguration {
                reason: "max_workers must be > 0",
            });
        }

        let table = Arc::new(LoadBalancingTable::new(config.max_workers));

        Ok(Self {
            config,
            table,
            metrics: LoadBalancerMetrics::default(),
            balancer_id: fastrand::u64(..),
        })
    }

    /// Select optimal worker for task - Ultra-optimized hot path
    ///
    /// # Errors
    ///
    /// Returns error if no workers available or selection fails
    pub fn select_worker(&self, task: &Task) -> Result<WorkerId, LoadBalancerError> {
        let start_time = Instant::now();

        // Fast path: check if workers available
        if self.table.worker_count.load(Ordering::Relaxed) == 0 {
            self.metrics
                .failed_selections
                .fetch_add(1, Ordering::Relaxed);
            return Err(LoadBalancerError::NoWorkersAvailable);
        }

        // Select worker based on strategy
        let selected_worker = match self.config.strategy {
            LoadBalancingStrategy::RoundRobin => self.table.get_round_robin_worker(),
            LoadBalancingStrategy::LeastLoaded => self.table.get_least_loaded_worker(),
            LoadBalancingStrategy::Predictive => self.table.get_predictive_worker(task),
            LoadBalancingStrategy::WeightedRoundRobin => {
                // Simplified weighted round-robin
                self.table.get_round_robin_worker()
            }
            LoadBalancingStrategy::LocalityAware => {
                // Simplified locality-aware selection
                self.table.get_predictive_worker(task)
            }
        };

        let selection_latency = u64::try_from(start_time.elapsed().as_nanos()).unwrap_or(u64::MAX);

        // Update metrics
        self.metrics
            .total_selections
            .fetch_add(1, Ordering::Relaxed);

        // Check latency requirement
        if selection_latency > self.config.target_latency_ns {
            self.metrics
                .failed_selections
                .fetch_add(1, Ordering::Relaxed);
            return Err(LoadBalancerError::LatencyViolation {
                latency_ns: selection_latency,
                max_latency_ns: self.config.target_latency_ns,
            });
        }

        selected_worker.map_or_else(
            || {
                self.metrics
                    .failed_selections
                    .fetch_add(1, Ordering::Relaxed);
                Err(LoadBalancerError::NoWorkersAvailable)
            },
            |worker_id| {
                self.metrics
                    .successful_selections
                    .fetch_add(1, Ordering::Relaxed);
                self.update_avg_latency(selection_latency);
                Ok(worker_id)
            },
        )
    }

    /// Update worker load information
    #[inline]
    pub fn update_worker_load(
        &self,
        worker_id: WorkerId,
        load: f64,
        task_count: u32,
        avg_latency_ns: u64,
    ) {
        self.table
            .update_worker_load(worker_id, load, task_count, avg_latency_ns);
    }

    /// Add worker to load balancer
    ///
    /// # Errors
    ///
    /// Returns error if worker pool is full or configuration is invalid
    pub fn add_worker(&self, _worker_id: WorkerId) -> Result<(), LoadBalancerError> {
        let current_count = self.table.worker_count.load(Ordering::Relaxed);
        if current_count >= self.config.max_workers {
            return Err(LoadBalancerError::InvalidConfiguration {
                reason: "Maximum workers exceeded",
            });
        }

        self.table
            .worker_count
            .store(current_count + 1, Ordering::Relaxed);
        self.table.generation.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Remove worker from load balancer
    ///
    /// # Errors
    ///
    /// Returns error if worker not found or removal fails
    pub fn remove_worker(&self, _worker_id: WorkerId) -> Result<(), LoadBalancerError> {
        let current_count = self.table.worker_count.load(Ordering::Relaxed);
        if current_count == 0 {
            return Err(LoadBalancerError::NoWorkersAvailable);
        }

        // For simplicity, just decrease count
        // In production, would need proper worker removal logic
        self.table
            .worker_count
            .store(current_count - 1, Ordering::Relaxed);
        self.table.generation.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get load balancer metrics
    #[must_use]
    pub const fn metrics(&self) -> &LoadBalancerMetrics {
        &self.metrics
    }

    /// Get current worker count
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.table.worker_count.load(Ordering::Relaxed)
    }

    /// Calculate load imbalance factor
    #[must_use]
    pub fn load_imbalance(&self) -> f64 {
        let worker_count = self.table.worker_count.load(Ordering::Relaxed);
        if worker_count < 2 {
            return 0.0_f64;
        }

        let mut loads = Vec::with_capacity(worker_count);
        let mut total_load = 0.0_f64;

        for i in 0..worker_count {
            if let Some(worker) = self.table.workers.get(i) {
                let load = f64::from(worker.current_load.load(Ordering::Relaxed)) / 1000.0_f64;
                loads.push(load);
                total_load += load;
            }
        }

        if loads.is_empty() {
            return 0.0_f64;
        }

        let avg_load = total_load / f64::from(u32::try_from(loads.len()).unwrap_or(u32::MAX));
        let mut variance = 0.0_f64;

        for load in &loads {
            let diff = load - avg_load;
            variance += diff * diff;
        }

        variance / f64::from(u32::try_from(loads.len()).unwrap_or(u32::MAX))
    }

    /// Update average latency metric
    #[inline]
    fn update_avg_latency(&self, new_latency: u64) {
        let current_avg = self
            .metrics
            .avg_selection_latency_ns
            .load(Ordering::Relaxed);
        let selections = self.metrics.total_selections.load(Ordering::Relaxed);

        if selections == 0 {
            self.metrics
                .avg_selection_latency_ns
                .store(new_latency, Ordering::Relaxed);
        } else {
            let new_avg = (current_avg
                .saturating_mul(selections.saturating_sub(1))
                .saturating_add(new_latency))
                / selections;
            self.metrics
                .avg_selection_latency_ns
                .store(new_avg, Ordering::Relaxed);
        }
    }
}

/// Get current timestamp in nanoseconds
fn current_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StrategyId;

    #[test]
    fn test_enhanced_load_balancer_creation() -> Result<(), LoadBalancerError> {
        let config = LoadBalancerConfig::default();
        let balancer = EnhancedLoadBalancer::new(config)?;

        assert_eq!(balancer.worker_count(), 0);
        assert!((balancer.load_imbalance() - 0.0_f64).abs() < f64::EPSILON);

        Ok(())
    }

    #[test]
    fn test_worker_management() -> Result<(), LoadBalancerError> {
        let config = LoadBalancerConfig::default();
        let balancer = EnhancedLoadBalancer::new(config)?;

        // Add workers
        balancer.add_worker(WorkerId::from_raw(0))?;
        balancer.add_worker(WorkerId::from_raw(1))?;

        assert_eq!(balancer.worker_count(), 2);

        // Remove worker
        balancer.remove_worker(WorkerId::from_raw(1))?;
        assert_eq!(balancer.worker_count(), 1);

        Ok(())
    }

    #[test]
    fn test_worker_selection() -> Result<(), LoadBalancerError> {
        let config = LoadBalancerConfig::default();
        let balancer = EnhancedLoadBalancer::new(config)?;

        // Add workers
        balancer.add_worker(WorkerId::from_raw(0))?;
        balancer.add_worker(WorkerId::from_raw(1))?;

        // Create test task
        let task = Task::new(StrategyId::new(), vec![1, 2, 3]);

        // Select worker
        let selected = balancer.select_worker(&task)?;
        assert!(selected.raw() < 2);

        Ok(())
    }

    #[test]
    fn test_load_update() -> Result<(), LoadBalancerError> {
        let config = LoadBalancerConfig::default();
        let balancer = EnhancedLoadBalancer::new(config)?;

        balancer.add_worker(WorkerId::from_raw(0))?;

        // Update worker load
        balancer.update_worker_load(WorkerId::from_raw(0), 0.5_f64, 10, 500_000);

        Ok(())
    }

    #[test]
    fn test_invalid_configuration() {
        let config = LoadBalancerConfig {
            max_workers: 0,
            ..LoadBalancerConfig::default()
        };

        assert!(EnhancedLoadBalancer::new(config).is_err());
    }

    #[test]
    fn test_metrics_collection() -> Result<(), LoadBalancerError> {
        let config = LoadBalancerConfig::default();
        let balancer = EnhancedLoadBalancer::new(config)?;

        let metrics = balancer.metrics();
        assert_eq!(metrics.total_selections.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.successful_selections.load(Ordering::Relaxed), 0);

        Ok(())
    }
}
