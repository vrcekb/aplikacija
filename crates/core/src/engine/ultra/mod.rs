//! Ultra-optimized engine components for <50µs task processing
//!
//! This module contains NUMA-aware work stealing schedulers and ultra-optimized
//! task processing components designed for MEV trading applications.

pub mod financial_scheduler;

pub use financial_scheduler::{FinancialConfig, FinancialMetrics, FinancialScheduler};

use thiserror::Error;

/// Ultra-optimized engine operation errors
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum UltraEngineError {
    /// Worker thread creation failed
    #[error("Worker thread creation failed")]
    WorkerCreationFailed,

    /// NUMA topology detection failed
    #[error("NUMA topology detection failed")]
    NumaDetectionFailed,

    /// CPU affinity setting failed
    #[error("CPU affinity setting failed")]
    AffinityFailed,

    /// Task submission failed - all queues full
    #[error("Task submission failed - all queues full")]
    TaskSubmissionFailed,

    /// No tasks available for stealing
    #[error("No tasks available for stealing")]
    NoTasksAvailable,

    /// Load balancer error
    #[error("Load balancer error")]
    LoadBalancerError,

    /// NUMA topology detection failed
    #[error("NUMA topology detection failed")]
    TopologyDetectionFailed,

    /// Worker thread join failed
    #[error("Worker thread join failed")]
    WorkerJoinFailed,

    /// Scheduler not running
    #[error("Scheduler not running")]
    NotRunning,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Thread creation failed
    #[error("Thread creation failed: {0}")]
    ThreadCreationFailed(String),
}

/// Result type for ultra engine operations
pub type UltraEngineResult<T> = Result<T, UltraEngineError>;

/// Task trait for work stealing scheduler
pub trait Task: Send + Sync + 'static {
    /// Execute the task
    fn execute(&self) -> TaskResult;

    /// Get task priority (higher = more important)
    fn priority(&self) -> u8 {
        0
    }

    /// Get estimated execution time in microseconds
    fn estimated_duration_us(&self) -> u64 {
        100
    }
}

/// Task execution result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskResult {
    /// Task completed successfully
    Success,
    /// Task failed with error message
    Failed(String),
    /// Task needs to be retried
    Retry,
}

/// NUMA node identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NumaNode(pub u8);

impl NumaNode {
    /// Creates new NUMA node identifier
    #[must_use]
    pub const fn new(id: u8) -> Self {
        Self(id)
    }

    /// Gets NUMA node ID
    #[must_use]
    pub const fn id(self) -> u8 {
        self.0
    }
}

/// CPU core identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CpuCore(pub u8);

impl CpuCore {
    /// Creates new CPU core identifier
    #[must_use]
    pub const fn new(id: u8) -> Self {
        Self(id)
    }

    /// Gets CPU core ID
    #[must_use]
    pub const fn id(self) -> u8 {
        self.0
    }
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Number of NUMA nodes
    pub numa_nodes: u8,
    /// Number of CPU cores per NUMA node
    pub cores_per_node: u8,
    /// Total number of CPU cores
    pub total_cores: u8,
    /// CPU to NUMA node mapping
    pub cpu_to_numa: Vec<NumaNode>,
}

impl NumaTopology {
    /// Detects NUMA topology of the system
    ///
    /// # Errors
    /// Returns error if topology detection fails
    pub fn detect() -> UltraEngineResult<Self> {
        let total_cores =
            u8::try_from(num_cpus::get()).map_err(|_| UltraEngineError::TopologyDetectionFailed)?;

        // For now, assume single NUMA node if detection fails
        // In production, use hwloc or similar for proper detection
        let numa_nodes = 1;
        let cores_per_node = total_cores;

        let cpu_to_numa = (0..total_cores).map(|_| NumaNode::new(0)).collect();

        Ok(Self {
            numa_nodes,
            cores_per_node,
            total_cores,
            cpu_to_numa,
        })
    }

    /// Gets NUMA node for given CPU core
    #[must_use]
    pub fn numa_node_for_cpu(&self, cpu: CpuCore) -> Option<NumaNode> {
        self.cpu_to_numa.get(cpu.id() as usize).copied()
    }

    /// Gets CPU cores for given NUMA node
    #[must_use]
    pub fn cpus_for_numa_node(&self, numa_node: NumaNode) -> Vec<CpuCore> {
        self.cpu_to_numa
            .iter()
            .enumerate()
            .filter_map(|(cpu_id, &node)| {
                if node == numa_node {
                    u8::try_from(cpu_id).ok().map(CpuCore::new)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Performance metrics for ultra engine
#[derive(Debug, Clone, Copy, Default)]
pub struct UltraEngineMetrics {
    /// Total tasks processed
    pub tasks_processed: u64,
    /// Total tasks stolen
    pub tasks_stolen: u64,
    /// Total task submission failures
    pub submission_failures: u64,
    /// Average task execution time in microseconds
    pub avg_execution_time_us: u64,
    /// Maximum task execution time in microseconds
    pub max_execution_time_us: u64,
    /// NUMA-local task ratio (0.0 - 1.0)
    pub numa_local_ratio: f64,
}

impl UltraEngineMetrics {
    /// Creates new metrics
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tasks_processed: 0,
            tasks_stolen: 0,
            submission_failures: 0,
            avg_execution_time_us: 0,
            max_execution_time_us: 0,
            numa_local_ratio: 0.0,
        }
    }

    /// Calculates steal ratio
    #[must_use]
    pub fn steal_ratio(&self) -> f64 {
        if self.tasks_processed == 0 {
            0.0_f64
        } else {
            #[allow(clippy::cast_precision_loss)]
            let tasks_stolen_f64 = self.tasks_stolen as f64;
            #[allow(clippy::cast_precision_loss)]
            let tasks_processed_f64 = self.tasks_processed as f64;
            tasks_stolen_f64 / tasks_processed_f64
        }
    }

    /// Calculates success ratio
    #[must_use]
    pub fn success_ratio(&self) -> f64 {
        let total_attempts = self.tasks_processed + self.submission_failures;
        if total_attempts == 0 {
            0.0_f64
        } else {
            #[allow(clippy::cast_precision_loss)]
            let tasks_processed_f64 = self.tasks_processed as f64;
            #[allow(clippy::cast_precision_loss)]
            let total_attempts_f64 = total_attempts as f64;
            tasks_processed_f64 / total_attempts_f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_topology_detection() -> UltraEngineResult<()> {
        let topology =
            NumaTopology::detect().map_err(|_| UltraEngineError::TopologyDetectionFailed)?;
        assert!(topology.total_cores > 0);
        assert!(topology.numa_nodes > 0);
        assert_eq!(topology.cpu_to_numa.len(), topology.total_cores as usize);
        Ok(())
    }

    #[test]
    fn test_numa_node_operations() {
        let node = NumaNode::new(42);
        assert_eq!(node.id(), 42);
    }

    #[test]
    fn test_cpu_core_operations() {
        let core = CpuCore::new(8);
        assert_eq!(core.id(), 8);
    }

    #[test]
    fn test_metrics_calculations() {
        let metrics = UltraEngineMetrics {
            tasks_processed: 100,
            tasks_stolen: 25,
            submission_failures: 5,
            ..UltraEngineMetrics::new()
        };

        assert!((metrics.steal_ratio() - 0.25).abs() < f64::EPSILON);
        assert!((metrics.success_ratio() - (100.0 / 105.0)).abs() < f64::EPSILON);
    }
}
