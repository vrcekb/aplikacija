//! NUMA-Aware Scheduler for `TallyIO` - Ultra-Low Latency Task Distribution
//!
//! Production-ready NUMA topology-aware task scheduling for financial applications

use crate::optimization::cpu_affinity::CpuAffinityManager;
use crate::types::{TaskId, WorkerId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;

/// NUMA node information with performance metrics
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: usize,
    /// CPU cores on this node
    pub cores: Vec<usize>,
    /// Memory size in GB
    pub memory_gb: u64,
    /// Current load (0.0 - 1.0)
    pub load: f64,
    /// Average latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Number of active tasks
    pub active_tasks: usize,
}

/// NUMA topology with performance tracking
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// Available NUMA nodes
    pub nodes: Vec<NumaNode>,
    /// Total number of nodes
    pub total_nodes: usize,
    /// Cross-node latency matrix (ns)
    pub cross_node_latency: Vec<Vec<u64>>,
}

/// Task with NUMA affinity requirements
#[derive(Debug, Clone)]
pub struct NumaTask {
    /// Task identifier
    pub id: TaskId,
    /// Preferred NUMA node (None = any)
    pub preferred_node: Option<usize>,
    /// Memory requirement in bytes
    pub memory_requirement: usize,
    /// CPU intensity (0.0 - 1.0)
    pub cpu_intensity: f64,
    /// Memory access pattern (Sequential, Random, Mixed)
    pub memory_pattern: MemoryPattern,
    /// Task priority (0 = lowest, 255 = highest)
    pub priority: u8,
    /// Creation timestamp
    pub created_at: Instant,
}

/// Memory access patterns for NUMA optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPattern {
    /// Sequential memory access
    Sequential,
    /// Random memory access
    Random,
    /// Mixed access pattern
    Mixed,
}

/// NUMA-aware scheduler with load balancing
pub struct NumaScheduler {
    /// NUMA topology
    topology: NumaTopology,
    /// Node load tracking
    node_loads: Vec<AtomicU64>,
    /// Node task counts
    node_task_counts: Vec<AtomicUsize>,
    /// Node latency tracking
    node_latencies: Vec<AtomicU64>,
    /// Worker assignments per node
    worker_assignments: HashMap<usize, Vec<WorkerId>>,
    /// CPU affinity manager
    cpu_affinity: Arc<CpuAffinityManager>,
    /// Statistics
    stats: Arc<NumaSchedulerStats>,
}

/// NUMA scheduler statistics
#[derive(Debug, Default)]
pub struct NumaSchedulerStats {
    /// Total tasks scheduled
    pub total_tasks: AtomicU64,
    /// Tasks scheduled locally (same node)
    pub local_tasks: AtomicU64,
    /// Tasks scheduled remotely (cross-node)
    pub remote_tasks: AtomicU64,
    /// Load balancing operations
    pub load_balance_ops: AtomicU64,
    /// Average scheduling latency (ns)
    pub avg_scheduling_latency_ns: AtomicU64,
    /// Node utilization per node
    pub node_utilization: Vec<AtomicU64>,
}

impl NumaScheduler {
    /// Create new NUMA-aware scheduler
    ///
    /// # Errors
    ///
    /// Returns error if NUMA topology detection fails or insufficient resources
    pub fn new(cpu_affinity: Arc<CpuAffinityManager>) -> Result<Self, NumaSchedulerError> {
        let topology = Self::detect_numa_topology()?;
        let node_count = topology.total_nodes;

        if node_count == 0 {
            return Err(NumaSchedulerError::NoNumaNodes);
        }

        let node_loads = (0..node_count).map(|_| AtomicU64::new(0)).collect();
        let node_task_counts = (0..node_count).map(|_| AtomicUsize::new(0)).collect();
        let node_latencies = (0..node_count).map(|_| AtomicU64::new(0)).collect();
        let node_utilization = (0..node_count).map(|_| AtomicU64::new(0)).collect();

        let stats = Arc::new(NumaSchedulerStats {
            node_utilization,
            ..NumaSchedulerStats::default()
        });

        Ok(Self {
            topology,
            node_loads,
            node_task_counts,
            node_latencies,
            worker_assignments: HashMap::new(),
            cpu_affinity,
            stats,
        })
    }

    /// Schedule task to optimal NUMA node
    ///
    /// # Errors
    ///
    /// Returns error if no suitable node found or scheduling fails
    pub fn schedule_task(&self, task: &NumaTask) -> Result<usize, NumaSchedulerError> {
        let start_time = Instant::now();

        // Find optimal node based on task requirements
        let optimal_node = self.find_optimal_node(task)?;

        // Update node statistics - safe indexing
        if let Some(task_count) = self.node_task_counts.get(optimal_node) {
            task_count.fetch_add(1, Ordering::Relaxed);
        }
        self.update_node_load(optimal_node, task.cpu_intensity);

        // Update global statistics
        self.stats.total_tasks.fetch_add(1, Ordering::Relaxed);

        if let Some(preferred) = task.preferred_node {
            if optimal_node == preferred {
                self.stats.local_tasks.fetch_add(1, Ordering::Relaxed);
            } else {
                self.stats.remote_tasks.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Record scheduling latency - safe conversion
        let scheduling_latency = u64::try_from(start_time.elapsed().as_nanos()).unwrap_or(u64::MAX); // Cap at max if overflow
        self.update_avg_latency(scheduling_latency);

        Ok(optimal_node)
    }

    /// Find optimal NUMA node for task
    fn find_optimal_node(&self, task: &NumaTask) -> Result<usize, NumaSchedulerError> {
        // If task has preferred node and it's available, use it
        if let Some(preferred) = task.preferred_node {
            if preferred < self.topology.total_nodes && self.is_node_available(preferred, task) {
                return Ok(preferred);
            }
        }

        // Find best node based on multiple criteria
        let mut best_node = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (node_id, node) in self.topology.nodes.iter().enumerate() {
            if !self.is_node_available(node_id, task) {
                continue;
            }

            let score = self.calculate_node_score(node_id, node, task);
            if score > best_score {
                best_score = score;
                best_node = node_id;
            }
        }

        if best_score == f64::NEG_INFINITY {
            return Err(NumaSchedulerError::NoSuitableNode);
        }

        Ok(best_node)
    }

    /// Calculate node suitability score
    fn calculate_node_score(&self, node_id: usize, node: &NumaNode, task: &NumaTask) -> f64 {
        let load_factor = self.get_node_load(node_id);
        let memory_factor = Self::calculate_memory_factor(node, task);
        let latency_factor = self.calculate_latency_factor(node_id);
        let affinity_factor = self.calculate_affinity_factor(node_id, task);

        // Weighted scoring (lower is better for load/latency, higher for memory/affinity)
        let score = affinity_factor.mul_add(
            0.4_f64,
            memory_factor.mul_add(
                0.3_f64,
                load_factor.mul_add(-0.2_f64, -latency_factor * 0.1_f64),
            ),
        );

        // Priority boost
        let priority_boost = f64::from(task.priority) / 255.0_f64 * 0.1_f64;

        score + priority_boost
    }

    /// Check if node is available for task
    fn is_node_available(&self, node_id: usize, _task: &NumaTask) -> bool {
        if node_id >= self.topology.total_nodes {
            return false;
        }

        let current_tasks = self
            .node_task_counts
            .get(node_id)
            .map_or(usize::MAX, |count| count.load(Ordering::Relaxed));
        let Some(node) = self.topology.nodes.get(node_id) else {
            return false;
        };

        // Check if node has capacity (max 2 tasks per core)
        let max_tasks = node.cores.len() * 2;
        current_tasks < max_tasks
    }

    /// Get current node load (0.0 - 1.0)
    fn get_node_load(&self, node_id: usize) -> f64 {
        let load_raw = self
            .node_loads
            .get(node_id)
            .map_or(0, |load| load.load(Ordering::Relaxed));
        f64::from_bits(load_raw)
    }

    /// Update node load
    fn update_node_load(&self, node_id: usize, additional_load: f64) {
        let current_load = self.get_node_load(node_id);
        let new_load = (current_load + additional_load).min(1.0_f64);
        if let Some(load_atomic) = self.node_loads.get(node_id) {
            load_atomic.store(new_load.to_bits(), Ordering::Relaxed);
        }
    }

    /// Calculate memory factor (higher = better)
    #[allow(clippy::cast_precision_loss)] // Acceptable for NUMA scheduling heuristics
    fn calculate_memory_factor(node: &NumaNode, task: &NumaTask) -> f64 {
        // Safe conversion with explicit types - use saturating conversion for large values
        let max_precise_usize = usize::try_from(1u64 << 53_i32).unwrap_or(usize::MAX);
        let required_gb = if task.memory_requirement > max_precise_usize {
            // For very large values, use approximation to avoid precision loss
            let shifted = task.memory_requirement >> 30_i32;
            // Safe conversion with bounds check
            if shifted <= (1usize << 53_i32) {
                shifted as f64 // Approximate GB conversion
            } else {
                f64::INFINITY // Indicate overflow
            }
        } else {
            // Safe conversion for smaller values
            let mem_f64 = task.memory_requirement as f64;
            mem_f64 / (1_024.0_f64 * 1_024.0_f64 * 1_024.0_f64)
        };

        let max_precise_u64 = 1u64 << 53_i32;
        let node_memory_f64 = if node.memory_gb > max_precise_u64 {
            // For very large values, use safe approximation
            f64::INFINITY // Indicate very large memory
        } else {
            // Safe conversion for normal values
            node.memory_gb as f64
        };
        let available_ratio = (node_memory_f64 - required_gb) / node_memory_f64;
        available_ratio.max(0.0_f64)
    }

    /// Calculate latency factor (lower = better)
    #[allow(clippy::cast_precision_loss)] // Acceptable for NUMA scheduling heuristics
    fn calculate_latency_factor(&self, node_id: usize) -> f64 {
        let latency_ns = self
            .node_latencies
            .get(node_id)
            .map_or(0, |latency| latency.load(Ordering::Relaxed));
        if latency_ns == 0 {
            return 0.0_f64;
        }

        // Normalize to 0.0-1.0 range (assuming max 1ms latency) - safe conversion
        let max_precise_u64 = 1u64 << 53_i32;
        let latency_f64 = if latency_ns > max_precise_u64 {
            // For very large values, use approximation
            let shifted = latency_ns >> 20_i32;
            if shifted <= max_precise_u64 {
                (shifted as f64) / 1000.0_f64 // Approximate conversion
            } else {
                1.0_f64 // Cap at maximum
            }
        } else {
            (latency_ns as f64) / 1_000_000.0_f64
        };
        latency_f64.min(1.0_f64)
    }

    /// Calculate CPU affinity factor
    #[allow(clippy::cast_precision_loss)] // Acceptable for NUMA scheduling heuristics
    fn calculate_affinity_factor(&self, node_id: usize, _task: &NumaTask) -> f64 {
        // Higher factor for nodes with available cores
        let Some(node) = self.topology.nodes.get(node_id) else {
            return 0.0_f64;
        };

        // Safe conversion for core count with explicit types
        let max_precise_usize = 1usize << 53_i32;
        let available_cores = if node.cores.len() > max_precise_usize {
            // For very large core counts, use approximation
            let shifted = node.cores.len() >> 10_i32;
            if shifted <= max_precise_usize {
                (shifted as f64) * 1024.0_f64
            } else {
                f64::INFINITY // Indicate overflow
            }
        } else {
            node.cores.len() as f64
        };

        let current_tasks = self.node_task_counts.get(node_id).map_or(0.0_f64, |count| {
            let task_count = count.load(Ordering::Relaxed);
            if task_count > max_precise_usize {
                let shifted = task_count >> 10_i32;
                if shifted <= max_precise_usize {
                    (shifted as f64) * 1024.0_f64
                } else {
                    f64::INFINITY
                }
            } else {
                task_count as f64
            }
        });

        if current_tasks == 0.0_f64 {
            1.0_f64
        } else {
            (available_cores / current_tasks).min(1.0_f64)
        }
    }

    /// Update average scheduling latency
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // Acceptable for latency calculations
    fn update_avg_latency(&self, new_latency_ns: u64) {
        let current_avg = self.stats.avg_scheduling_latency_ns.load(Ordering::Relaxed);
        let total_tasks = self.stats.total_tasks.load(Ordering::Relaxed);

        if total_tasks <= 1 {
            self.stats
                .avg_scheduling_latency_ns
                .store(new_latency_ns, Ordering::Relaxed);
        } else {
            // Exponential moving average with safe conversion
            let alpha = 0.1_f64; // Smoothing factor

            // Safe conversion to f64 with precision handling and explicit types
            let max_precise_u64 = 1u64 << 53_i32;
            let current_avg_f64 = if current_avg > max_precise_u64 {
                let shifted = current_avg >> 10_i32;
                if shifted <= max_precise_u64 {
                    (shifted as f64) * 1024.0_f64
                } else {
                    current_avg as f64 // Fallback for extreme values
                }
            } else {
                current_avg as f64
            };

            let new_latency_f64 = if new_latency_ns > max_precise_u64 {
                let shifted = new_latency_ns >> 10_i32;
                if shifted <= max_precise_u64 {
                    (shifted as f64) * 1024.0_f64
                } else {
                    new_latency_ns as f64 // Fallback for extreme values
                }
            } else {
                new_latency_ns as f64
            };

            let new_avg_f64 = (1.0_f64 - alpha).mul_add(current_avg_f64, alpha * new_latency_f64);

            // Safe conversion back to u64 with bounds checking
            let u64_max_f64 = 18_446_744_073_709_551_615.0_f64; // u64::MAX as f64 approximation
            let new_avg = if new_avg_f64.is_finite()
                && new_avg_f64 >= 0.0_f64
                && new_avg_f64 <= u64_max_f64
            {
                new_avg_f64 as u64
            } else {
                current_avg // Keep current if calculation is invalid
            };

            self.stats
                .avg_scheduling_latency_ns
                .store(new_avg, Ordering::Relaxed);
        }
    }

    /// Perform load balancing across nodes
    ///
    /// # Errors
    ///
    /// Returns error if load balancing operation fails
    pub fn balance_load(&self) -> Result<usize, NumaSchedulerError> {
        let mut moves = 0;

        // Find overloaded and underloaded nodes
        let mut overloaded = Vec::with_capacity(self.topology.total_nodes);
        let mut underloaded = Vec::with_capacity(self.topology.total_nodes);

        for node_id in 0..self.topology.total_nodes {
            let load = self.get_node_load(node_id);
            if load > 0.8_f64 {
                overloaded.push(node_id);
            } else if load < 0.3_f64 {
                underloaded.push(node_id);
            }
        }

        // Balance load between nodes
        for &overloaded_node in &overloaded {
            for &underloaded_node in &underloaded {
                if self.can_migrate_tasks(overloaded_node, underloaded_node) {
                    moves += 1;
                    // In real implementation, would migrate actual tasks
                    self.simulate_task_migration(overloaded_node, underloaded_node);
                }
            }
        }

        if moves > 0 {
            self.stats
                .load_balance_ops
                .fetch_add(moves, Ordering::Relaxed);
        }

        usize::try_from(moves).map_err(|_| NumaSchedulerError::TopologyDetectionFailed {
            reason: "Move count conversion failed".to_string(),
        })
    }

    /// Check if tasks can be migrated between nodes
    fn can_migrate_tasks(&self, from_node: usize, to_node: usize) -> bool {
        let from_load = self.get_node_load(from_node);
        let to_load = self.get_node_load(to_node);

        // Only migrate if significant load difference
        from_load - to_load > 0.3_f64
    }

    /// Simulate task migration for load balancing
    fn simulate_task_migration(&self, from_node: usize, to_node: usize) {
        let migration_load = 0.1_f64; // Migrate 10% of load

        let from_load = self.get_node_load(from_node);
        let to_load = self.get_node_load(to_node);

        let new_from_load = (from_load - migration_load).max(0.0_f64);
        let new_to_load = (to_load + migration_load).min(1.0_f64);

        if let Some(from_atomic) = self.node_loads.get(from_node) {
            from_atomic.store(new_from_load.to_bits(), Ordering::Relaxed);
        }
        if let Some(to_atomic) = self.node_loads.get(to_node) {
            to_atomic.store(new_to_load.to_bits(), Ordering::Relaxed);
        }
    }

    /// Get scheduler statistics
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::if_same_then_else)] // Acceptable for statistics
    pub fn get_stats(&self) -> NumaSchedulerStatsSnapshot {
        let total_tasks = self.stats.total_tasks.load(Ordering::Relaxed);
        let local_tasks = self.stats.local_tasks.load(Ordering::Relaxed);
        let remote_tasks = self.stats.remote_tasks.load(Ordering::Relaxed);

        NumaSchedulerStatsSnapshot {
            total_tasks,
            local_tasks,
            remote_tasks,
            locality_ratio: if total_tasks > 0 {
                // Safe conversion with precision handling
                // Safe conversion - same for both branches due to precision limits
                let local_f64 = local_tasks as f64;
                let total_f64 = total_tasks as f64;
                local_f64 / total_f64
            } else {
                0.0_f64
            },
            load_balance_ops: self.stats.load_balance_ops.load(Ordering::Relaxed),
            avg_scheduling_latency_ns: self.stats.avg_scheduling_latency_ns.load(Ordering::Relaxed),
            node_utilization: self
                .stats
                .node_utilization
                .iter()
                .map(|u| u.load(Ordering::Relaxed))
                .collect(),
        }
    }

    /// Detect NUMA topology
    #[allow(clippy::unnecessary_wraps)] // Result needed for Linux NUMA detection
    fn detect_numa_topology() -> Result<NumaTopology, NumaSchedulerError> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux_numa_topology()
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Fallback to single-node topology
            Ok(NumaTopology {
                nodes: vec![NumaNode {
                    id: 0,
                    cores: (0..num_cpus::get()).collect(),
                    memory_gb: 16, // Default assumption
                    load: 0.0_f64,
                    avg_latency_ns: 0,
                    active_tasks: 0,
                }],
                total_nodes: 1,
                cross_node_latency: vec![vec![0]],
            })
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_numa_topology() -> Result<NumaTopology, NumaSchedulerError> {
        use std::fs;

        let numa_path = "/sys/devices/system/node";
        if !std::path::Path::new(numa_path).exists() {
            return Err(NumaSchedulerError::NumaNotSupported);
        }

        let mut nodes = Vec::new();

        if let Ok(entries) = fs::read_dir(numa_path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                if name_str.starts_with("node") {
                    if let Ok(node_id) = name_str[4..].parse::<usize>() {
                        if let Some(node) = Self::parse_numa_node(node_id) {
                            nodes.push(node);
                        }
                    }
                }
            }
        }

        if nodes.is_empty() {
            return Err(NumaSchedulerError::NoNumaNodes);
        }

        // Build cross-node latency matrix (simplified)
        let node_count = nodes.len();
        let cross_node_latency = vec![vec![100; node_count]; node_count]; // 100ns default

        Ok(NumaTopology {
            total_nodes: node_count,
            nodes,
            cross_node_latency,
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_numa_node(node_id: usize) -> Option<NumaNode> {
        use std::fs;

        let cpulist_path = format!("/sys/devices/system/node/node{node_id}/cpulist");
        let meminfo_path = format!("/sys/devices/system/node/node{node_id}/meminfo");

        let cores = if let Ok(cpulist) = fs::read_to_string(cpulist_path) {
            Self::parse_cpulist(cpulist.trim())
        } else {
            Vec::new()
        };

        let memory_gb = if let Ok(meminfo) = fs::read_to_string(meminfo_path) {
            Self::parse_memory_info(&meminfo)
        } else {
            4 // Default 4GB
        };

        Some(NumaNode {
            id: node_id,
            cores,
            memory_gb,
            load: 0.0,
            avg_latency_ns: 0,
            active_tasks: 0,
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_cpulist(cpulist: &str) -> Vec<usize> {
        let mut cores = Vec::new();

        for part in cpulist.split(',') {
            if let Some((start, end)) = part.split_once('-') {
                if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end.parse::<usize>()) {
                    cores.extend(start..=end);
                }
            } else if let Ok(core) = part.parse::<usize>() {
                cores.push(core);
            }
        }

        cores
    }

    #[cfg(target_os = "linux")]
    fn parse_memory_info(meminfo: &str) -> u64 {
        for line in meminfo.lines() {
            if line.contains("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Some(kb_str) = parts.get(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb / 1024 / 1024; // Convert KB to GB
                        }
                    }
                }
            }
        }
        4 // Default 4GB
    }

    /// Assign worker to NUMA node
    ///
    /// # Errors
    ///
    /// Returns error if worker assignment fails
    pub fn assign_worker(
        &mut self,
        worker_id: WorkerId,
        node_id: usize,
    ) -> Result<(), NumaSchedulerError> {
        if node_id >= self.topology.total_nodes {
            return Err(NumaSchedulerError::InvalidNodeId { node_id });
        }

        // Set CPU affinity for worker - use any critical thread type as placeholder
        if self
            .cpu_affinity
            .pin_critical_thread(crate::optimization::cpu_affinity::CriticalThreadType::MevScanner)
            .is_err()
        {
            tracing::warn!("Failed to set CPU affinity for worker {:?}", worker_id);
        }

        self.worker_assignments
            .entry(node_id)
            .or_default()
            .push(worker_id);
        Ok(())
    }

    /// Get workers assigned to node
    #[must_use]
    pub fn get_node_workers(&self, node_id: usize) -> Vec<WorkerId> {
        self.worker_assignments
            .get(&node_id)
            .cloned()
            .unwrap_or_default()
    }
}

/// NUMA scheduler statistics snapshot
#[derive(Debug, Clone)]
pub struct NumaSchedulerStatsSnapshot {
    /// Total tasks scheduled
    pub total_tasks: u64,
    /// Tasks scheduled locally
    pub local_tasks: u64,
    /// Tasks scheduled remotely
    pub remote_tasks: u64,
    /// Locality ratio (0.0 - 1.0)
    pub locality_ratio: f64,
    /// Load balancing operations
    pub load_balance_ops: u64,
    /// Average scheduling latency
    pub avg_scheduling_latency_ns: u64,
    /// Node utilization per node
    pub node_utilization: Vec<u64>,
}

/// NUMA scheduler errors
#[derive(Error, Debug)]
pub enum NumaSchedulerError {
    /// NUMA not supported on this system
    #[error("NUMA not supported on this system")]
    NumaNotSupported,
    /// No NUMA nodes detected
    #[error("No NUMA nodes detected")]
    NoNumaNodes,
    /// No suitable node found for task
    #[error("No suitable node found for task")]
    NoSuitableNode,
    /// Invalid node ID
    #[error("Invalid node ID: {node_id}")]
    InvalidNodeId {
        /// The invalid node ID
        node_id: usize,
    },
    /// Topology detection failed
    #[error("NUMA topology detection failed: {reason}")]
    TopologyDetectionFailed {
        /// Reason for detection failure
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_scheduler_creation() -> Result<(), NumaSchedulerError> {
        let cpu_cores = vec![0, 1, 2, 3];
        let cpu_affinity = Arc::new(CpuAffinityManager::new(cpu_cores).map_err(|_| {
            NumaSchedulerError::TopologyDetectionFailed {
                reason: "CPU affinity manager creation failed".to_string(),
            }
        })?);
        let scheduler = NumaScheduler::new(cpu_affinity)?;

        assert!(scheduler.topology.total_nodes > 0);
        Ok(())
    }

    #[test]
    fn test_task_scheduling() -> Result<(), NumaSchedulerError> {
        let cpu_cores = vec![0, 1, 2, 3];
        let cpu_affinity = Arc::new(CpuAffinityManager::new(cpu_cores).map_err(|_| {
            NumaSchedulerError::TopologyDetectionFailed {
                reason: "CPU affinity manager creation failed".to_string(),
            }
        })?);
        let scheduler = NumaScheduler::new(cpu_affinity)?;

        let task = NumaTask {
            id: TaskId::new(),
            preferred_node: None,
            memory_requirement: 1024 * 1024, // 1MB
            cpu_intensity: 0.5,
            memory_pattern: MemoryPattern::Sequential,
            priority: 128,
            created_at: Instant::now(),
        };

        let node_id = scheduler.schedule_task(&task)?;
        assert!(node_id < scheduler.topology.total_nodes);

        Ok(())
    }
}
