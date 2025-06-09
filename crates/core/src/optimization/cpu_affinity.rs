//! CPU Affinity Management for `TallyIO` - Ultra-Low Latency Thread Pinning
//!
//! Production-ready CPU affinity management for financial applications requiring <1ms latency

use super::{OptimizationError, OptimizationResult};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Critical thread types for CPU affinity assignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CriticalThreadType {
    /// MEV scanner - highest priority
    MevScanner,
    /// State manager - critical for consistency
    StateManager,
    /// Mempool processor - transaction handling
    MempoolProcessor,
    /// Network handler - I/O operations
    NetworkHandler,
    /// Metrics collector - monitoring
    MetricsCollector,
}

/// CPU affinity manager with enhanced thread management and NUMA awareness
pub struct CpuAffinityManager {
    /// Available CPU cores
    cpu_cores: Vec<u32>,
    /// Critical thread assignments
    critical_cores: HashMap<CriticalThreadType, u32>,
    /// Worker cores for general use
    worker_cores: Vec<u32>,
    /// Total cores available
    total_cores: usize,
    /// CPU isolation enabled
    isolation_enabled: bool,
    /// NUMA node assignments for cores
    numa_topology: HashMap<u32, usize>,
    /// Core usage statistics
    core_usage: Vec<AtomicUsize>,
    /// Preferred NUMA node for allocations
    preferred_numa_node: AtomicUsize,
}

impl CpuAffinityManager {
    /// Create new CPU affinity manager with automatic core assignment
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails or insufficient cores
    pub fn new(cpu_cores: Vec<u32>) -> OptimizationResult<Self> {
        if cpu_cores.is_empty() {
            return Err(OptimizationError::CpuAffinityError {
                reason: "No CPU cores specified".to_string(),
            });
        }

        let total_cores = cpu_cores.len();

        if total_cores < 4 {
            return Err(OptimizationError::CpuAffinityError {
                reason: format!("Insufficient cores: {total_cores} available, 4 required"),
            });
        }

        // Assign critical cores (first 4-5 cores) - safe indexing
        let mut critical_cores = HashMap::new();

        if let Some(&core) = cpu_cores.first() {
            critical_cores.insert(CriticalThreadType::MevScanner, core);
        }
        if let Some(&core) = cpu_cores.get(1) {
            critical_cores.insert(CriticalThreadType::StateManager, core);
        }
        if let Some(&core) = cpu_cores.get(2) {
            critical_cores.insert(CriticalThreadType::MempoolProcessor, core);
        }
        if let Some(&core) = cpu_cores.get(3) {
            critical_cores.insert(CriticalThreadType::NetworkHandler, core);
        }

        if total_cores > 4 {
            if let Some(&core) = cpu_cores.get(4) {
                critical_cores.insert(CriticalThreadType::MetricsCollector, core);
            }
        }

        // Remaining cores for workers (reserve last 2 for OS if enough cores)
        let worker_start = critical_cores.len();
        let worker_end = if total_cores > 8 {
            total_cores - 2
        } else {
            total_cores
        };
        let worker_cores: Vec<u32> = cpu_cores
            .get(worker_start..worker_end)
            .map_or_else(Vec::new, <[u32]>::to_vec);

        // Initialize NUMA topology (simplified for now)
        let mut numa_topology = HashMap::new();
        let mut core_usage = Vec::with_capacity(total_cores);

        // Assign cores to NUMA nodes (assume 2 nodes for dual-socket systems)
        for (i, &core) in cpu_cores.iter().enumerate() {
            let numa_node = if total_cores > 8 {
                i / (total_cores / 2)
            } else {
                0
            };
            numa_topology.insert(core, numa_node);
            core_usage.push(AtomicUsize::new(0));
        }

        Ok(Self {
            cpu_cores,
            critical_cores,
            worker_cores,
            total_cores,
            isolation_enabled: true,
            numa_topology,
            core_usage,
            preferred_numa_node: AtomicUsize::new(0),
        })
    }

    /// Pin critical thread to its assigned core
    ///
    /// # Errors
    ///
    /// Returns error if thread type not found or pinning fails
    pub fn pin_critical_thread(&self, thread_type: CriticalThreadType) -> OptimizationResult<()> {
        let core_id = self.critical_cores.get(&thread_type).ok_or_else(|| {
            OptimizationError::CpuAffinityError {
                reason: format!("Thread type {thread_type:?} not assigned to any core"),
            }
        })?;

        self.set_affinity(*core_id)?;
        Self::set_high_priority();

        tracing::info!("Pinned {:?} to core {}", thread_type, core_id);
        Ok(())
    }

    /// Pin worker thread to available worker core
    ///
    /// # Errors
    ///
    /// Returns error if no worker cores available or pinning fails
    pub fn pin_worker_thread(&self, worker_id: usize) -> OptimizationResult<()> {
        if self.worker_cores.is_empty() {
            return Err(OptimizationError::CpuAffinityError {
                reason: "No worker cores available".to_string(),
            });
        }

        let core_id = *self
            .worker_cores
            .get(worker_id % self.worker_cores.len())
            .ok_or_else(|| OptimizationError::CpuAffinityError {
                reason: "Worker core index out of bounds".to_string(),
            })?;
        self.set_affinity(core_id)?;

        tracing::debug!("Pinned worker {} to core {}", worker_id, core_id);
        Ok(())
    }

    /// Set CPU affinity for current thread
    ///
    /// # Errors
    ///
    /// Returns error if affinity setting fails
    pub fn set_affinity(&self, core_id: u32) -> OptimizationResult<()> {
        if !self.cpu_cores.contains(&core_id) {
            return Err(OptimizationError::CpuAffinityError {
                reason: format!("Core {core_id} not in available cores"),
            });
        }

        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            use core_affinity::{set_for_current, CoreId};
            if !set_for_current(CoreId {
                id: core_id as usize,
            }) {
                return Err(OptimizationError::CpuAffinityError {
                    reason: format!("Failed to set affinity to core {core_id}"),
                });
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // CPU affinity not supported on this platform
            tracing::warn!("CPU affinity not supported on this platform");
        }

        Ok(())
    }

    /// Set high priority for critical threads
    fn set_high_priority() {
        #[cfg(target_os = "linux")]
        {
            use libc::{sched_param, sched_setscheduler, SCHED_FIFO};

            let param = sched_param { sched_priority: 50 }; // High but not max priority

            unsafe {
                if sched_setscheduler(0, SCHED_FIFO, &param) != 0 {
                    tracing::warn!(
                        "Failed to set thread priority - continuing without high priority"
                    );
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            // Windows priority setting would require winapi crate
            // For now, log a warning and continue
            tracing::warn!("High priority setting requires winapi crate on Windows");
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            tracing::warn!("High priority setting not implemented for this platform");
        }
    }

    /// Get available CPU cores
    #[must_use]
    pub fn available_cores(&self) -> &[u32] {
        &self.cpu_cores
    }

    /// Get optimal core for thread
    #[must_use]
    pub fn get_optimal_core(&self, thread_id: usize) -> u32 {
        self.cpu_cores
            .get(thread_id % self.cpu_cores.len())
            .copied()
            .unwrap_or(0)
    }

    /// Get topology information for monitoring
    #[must_use]
    pub fn get_topology_info(&self) -> TopologyInfo {
        TopologyInfo {
            total_cores: self.total_cores,
            critical_cores: self.critical_cores.clone(),
            worker_cores: self.worker_cores.clone(),
            isolation_enabled: self.isolation_enabled,
        }
    }

    /// Enable or disable CPU isolation
    pub fn set_isolation(&mut self, enabled: bool) {
        self.isolation_enabled = enabled;
        if enabled {
            tracing::info!("CPU isolation enabled for critical cores");
        } else {
            tracing::info!("CPU isolation disabled");
        }
    }

    /// Get critical core assignment for thread type
    #[must_use]
    pub fn get_critical_core(&self, thread_type: CriticalThreadType) -> Option<u32> {
        self.critical_cores.get(&thread_type).copied()
    }

    /// Get worker cores list
    #[must_use]
    pub fn get_worker_cores(&self) -> &[u32] {
        &self.worker_cores
    }

    /// Check if isolation is enabled
    #[must_use]
    pub const fn is_isolation_enabled(&self) -> bool {
        self.isolation_enabled
    }

    /// Get NUMA node for CPU core
    #[must_use]
    pub fn get_numa_node(&self, core: u32) -> Option<usize> {
        self.numa_topology.get(&core).copied()
    }

    /// Get cores on specific NUMA node
    #[must_use]
    pub fn get_cores_on_numa_node(&self, numa_node: usize) -> Vec<u32> {
        self.numa_topology
            .iter()
            .filter_map(
                |(&core, &node)| {
                    if node == numa_node {
                        Some(core)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Pin thread to core on specific NUMA node
    ///
    /// # Errors
    ///
    /// Returns error if no cores available on NUMA node or pinning fails
    pub fn pin_to_numa_node(&self, numa_node: usize, thread_id: usize) -> OptimizationResult<u32> {
        let cores_on_node = self.get_cores_on_numa_node(numa_node);

        if cores_on_node.is_empty() {
            return Err(OptimizationError::CpuAffinityError {
                reason: format!("No cores available on NUMA node {numa_node}"),
            });
        }

        let core = *cores_on_node
            .get(thread_id % cores_on_node.len())
            .ok_or_else(|| OptimizationError::CpuAffinityError {
                reason: "Core index out of bounds".to_string(),
            })?;
        self.set_affinity(core)?;

        // Update usage statistics
        if let Some(index) = self.cpu_cores.iter().position(|&c| c == core) {
            if let Some(usage) = self.core_usage.get(index) {
                usage.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(core)
    }

    /// Get least used core on NUMA node
    #[must_use]
    pub fn get_least_used_core_on_numa(&self, numa_node: usize) -> Option<u32> {
        let cores_on_node = self.get_cores_on_numa_node(numa_node);

        cores_on_node.into_iter().min_by_key(|&core| {
            self.cpu_cores
                .iter()
                .position(|&c| c == core)
                .and_then(|index| self.core_usage.get(index))
                .map_or(usize::MAX, |usage| usage.load(Ordering::Relaxed))
        })
    }

    /// Set preferred NUMA node
    pub fn set_preferred_numa_node(&self, numa_node: usize) {
        self.preferred_numa_node.store(numa_node, Ordering::Relaxed);
    }

    /// Get preferred NUMA node
    #[must_use]
    pub fn get_preferred_numa_node(&self) -> usize {
        self.preferred_numa_node.load(Ordering::Relaxed)
    }

    /// Get core usage statistics
    #[must_use]
    pub fn get_core_usage(&self, core: u32) -> Option<usize> {
        self.cpu_cores
            .iter()
            .position(|&c| c == core)
            .and_then(|index| self.core_usage.get(index))
            .map(|usage| usage.load(Ordering::Relaxed))
    }

    /// Reset core usage statistics
    pub fn reset_usage_stats(&self) {
        for usage in &self.core_usage {
            usage.store(0, Ordering::Relaxed);
        }
    }
}

/// Topology information for monitoring
#[derive(Debug, Clone)]
pub struct TopologyInfo {
    /// Total number of CPU cores
    pub total_cores: usize,
    /// Critical thread to core assignments
    pub critical_cores: HashMap<CriticalThreadType, u32>,
    /// Available worker cores
    pub worker_cores: Vec<u32>,
    /// Whether CPU isolation is enabled
    pub isolation_enabled: bool,
}
