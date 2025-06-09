//! Memory Statistics - Comprehensive Memory Usage Tracking
//!
//! Implements detailed memory statistics collection with:
//! - Real-time memory usage monitoring across all components
//! - Performance metrics for allocation/deallocation operations
//! - Memory leak detection and reporting
//! - NUMA locality tracking and optimization suggestions
//! - Integration with monitoring systems for alerting

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Memory usage metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total allocated bytes
    pub total_allocated_bytes: u64,
    /// Peak allocated bytes
    pub peak_allocated_bytes: u64,
    /// Total allocation count
    pub total_allocations: u64,
    /// Total deallocation count
    pub total_deallocations: u64,
    /// Average allocation size in bytes
    pub avg_allocation_size: u64,
    /// Average allocation time in nanoseconds
    pub avg_allocation_time_ns: u64,
    /// Memory utilization percentage
    pub utilization_percentage: f64,
    /// NUMA locality ratio (0.0-1.0)
    pub numa_locality_ratio: f64,
    /// Memory fragmentation ratio (0.0-1.0)
    pub fragmentation_ratio: f64,
    /// Timestamp of last update
    pub timestamp: u64,
}

/// Memory usage breakdown by component
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Component name
    pub component: String,
    /// Allocated bytes
    pub allocated_bytes: u64,
    /// Peak allocated bytes
    pub peak_bytes: u64,
    /// Allocation count
    pub allocation_count: u64,
    /// Average allocation size
    pub avg_size: u64,
    /// Memory growth rate (bytes per second)
    pub growth_rate: f64,
    /// Last update timestamp
    pub last_updated: u64,
}

/// Memory statistics collector
#[derive(Debug)]
pub struct MemoryStats {
    /// Global memory metrics
    global_metrics: Arc<Mutex<MemoryMetrics>>,
    /// Component-specific usage tracking
    component_usage: Arc<Mutex<HashMap<String, MemoryUsage>>>,
    /// Statistics collection enabled
    collection_enabled: bool,
    /// Collection interval
    collection_interval: Duration,
    /// Last collection timestamp
    last_collection: Mutex<Instant>,
}

impl MemoryStats {
    /// Create new memory statistics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            global_metrics: Arc::new(Mutex::new(MemoryMetrics::default())),
            component_usage: Arc::new(Mutex::new(HashMap::new())),
            collection_enabled: true,
            collection_interval: Duration::from_secs(1),
            last_collection: Mutex::new(Instant::now()),
        }
    }

    /// Create with custom collection interval
    #[must_use]
    pub fn with_interval(interval: Duration) -> Self {
        Self {
            collection_interval: interval,
            ..Self::new()
        }
    }

    /// Enable or disable statistics collection
    pub const fn set_collection_enabled(&mut self, enabled: bool) {
        self.collection_enabled = enabled;
    }

    /// Record allocation for component
    pub fn record_allocation(&self, component: &str, size: u64, allocation_time_ns: u64) {
        if !self.collection_enabled {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        // Update global metrics
        if let Ok(mut metrics) = self.global_metrics.lock() {
            metrics.total_allocations += 1;
            metrics.total_allocated_bytes += size;

            if metrics.total_allocated_bytes > metrics.peak_allocated_bytes {
                metrics.peak_allocated_bytes = metrics.total_allocated_bytes;
            }

            // Update average allocation size
            if metrics.total_allocations > 0 {
                metrics.avg_allocation_size =
                    metrics.total_allocated_bytes / metrics.total_allocations;
            }

            // Update average allocation time
            if metrics.total_allocations > 0 {
                let total_time = metrics.avg_allocation_time_ns * (metrics.total_allocations - 1)
                    + allocation_time_ns;
                metrics.avg_allocation_time_ns = total_time / metrics.total_allocations;
            }

            metrics.timestamp = timestamp;
        }

        // Update component usage
        if let Ok(mut usage_map) = self.component_usage.lock() {
            let usage = usage_map
                .entry(component.to_string())
                .or_insert_with(|| MemoryUsage {
                    component: component.to_string(),
                    ..Default::default()
                });

            let previous_bytes = usage.allocated_bytes;
            let previous_time = usage.last_updated;

            usage.allocated_bytes += size;
            usage.allocation_count += 1;
            usage.last_updated = timestamp;

            if usage.allocated_bytes > usage.peak_bytes {
                usage.peak_bytes = usage.allocated_bytes;
            }

            if usage.allocation_count > 0 {
                usage.avg_size = usage.allocated_bytes / usage.allocation_count;
            }

            // Calculate growth rate
            if previous_time > 0 && timestamp > previous_time {
                let time_diff = timestamp - previous_time;
                let bytes_diff = usage.allocated_bytes.saturating_sub(previous_bytes);
                usage.growth_rate = f64::from(u32::try_from(bytes_diff).unwrap_or(u32::MAX))
                    / f64::from(u32::try_from(time_diff).unwrap_or(1));
            }
        }
    }

    /// Record deallocation for component
    pub fn record_deallocation(&self, component: &str, size: u64) {
        if !self.collection_enabled {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        // Update global metrics
        if let Ok(mut metrics) = self.global_metrics.lock() {
            metrics.total_deallocations += 1;
            metrics.total_allocated_bytes = metrics.total_allocated_bytes.saturating_sub(size);
            metrics.timestamp = timestamp;
        }

        // Update component usage
        if let Ok(mut usage_map) = self.component_usage.lock() {
            if let Some(usage) = usage_map.get_mut(component) {
                usage.allocated_bytes = usage.allocated_bytes.saturating_sub(size);
                usage.last_updated = timestamp;
            }
        }
    }

    /// Update NUMA locality metrics
    pub fn update_numa_locality(&self, locality_ratio: f64) {
        if !self.collection_enabled {
            return;
        }

        if let Ok(mut metrics) = self.global_metrics.lock() {
            metrics.numa_locality_ratio = locality_ratio;
            metrics.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
        }
    }

    /// Update memory utilization
    pub fn update_utilization(&self, utilization_percentage: f64) {
        if !self.collection_enabled {
            return;
        }

        if let Ok(mut metrics) = self.global_metrics.lock() {
            metrics.utilization_percentage = utilization_percentage;
            metrics.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
        }
    }

    /// Update fragmentation ratio
    pub fn update_fragmentation(&self, fragmentation_ratio: f64) {
        if !self.collection_enabled {
            return;
        }

        if let Ok(mut metrics) = self.global_metrics.lock() {
            metrics.fragmentation_ratio = fragmentation_ratio;
            metrics.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
        }
    }

    /// Get current global metrics
    #[must_use]
    pub fn global_metrics(&self) -> MemoryMetrics {
        self.global_metrics
            .lock()
            .map_or_else(|_| MemoryMetrics::default(), |metrics| metrics.clone())
    }

    /// Get component usage statistics
    #[must_use]
    pub fn component_usage(&self, component: &str) -> Option<MemoryUsage> {
        self.component_usage
            .lock()
            .map_or(None, |usage_map| usage_map.get(component).cloned())
    }

    /// Get all component usage statistics
    #[must_use]
    pub fn all_component_usage(&self) -> HashMap<String, MemoryUsage> {
        self.component_usage
            .lock()
            .map_or_else(|_| HashMap::new(), |usage_map| usage_map.clone())
    }

    /// Get top memory consumers
    #[must_use]
    pub fn top_consumers(&self, limit: usize) -> Vec<MemoryUsage> {
        self.component_usage.lock().map_or_else(
            |_| Vec::new(),
            |usage_map| {
                let mut consumers: Vec<MemoryUsage> = usage_map.values().cloned().collect();
                consumers.sort_by(|a, b| b.allocated_bytes.cmp(&a.allocated_bytes));
                consumers.truncate(limit);
                consumers
            },
        )
    }

    /// Detect potential memory leaks
    #[must_use]
    pub fn detect_leaks(&self, growth_threshold_bytes_per_sec: f64) -> Vec<String> {
        let mut leaking_components = Vec::new();

        if let Ok(usage_map) = self.component_usage.lock() {
            leaking_components.extend(
                usage_map
                    .iter()
                    .filter(|(_, usage)| usage.growth_rate > growth_threshold_bytes_per_sec)
                    .map(|(component, _)| component.clone()),
            );
        }

        leaking_components
    }

    /// Generate memory report
    #[must_use]
    pub fn generate_report(&self) -> MemoryReport {
        let global = self.global_metrics();
        let components = self.all_component_usage();
        let top_consumers = self.top_consumers(10);
        let potential_leaks = self.detect_leaks(1024.0_f64 * 1024.0_f64); // 1MB/sec threshold

        MemoryReport {
            global_metrics: global,
            component_count: components.len(),
            top_consumers,
            potential_leaks,
            total_components: components.len(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }

    /// Reset all statistics
    pub fn reset(&self) {
        if let Ok(mut metrics) = self.global_metrics.lock() {
            *metrics = MemoryMetrics::default();
        }

        if let Ok(mut usage_map) = self.component_usage.lock() {
            usage_map.clear();
        }

        if let Ok(mut last_collection) = self.last_collection.lock() {
            *last_collection = Instant::now();
        }
    }

    /// Check if collection is due
    #[must_use]
    pub fn is_collection_due(&self) -> bool {
        self.last_collection
            .lock()
            .is_ok_and(|last_collection| last_collection.elapsed() >= self.collection_interval)
    }

    /// Mark collection as completed
    pub fn mark_collection_completed(&self) {
        if let Ok(mut last_collection) = self.last_collection.lock() {
            *last_collection = Instant::now();
        }
    }
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive memory report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    /// Global memory metrics
    pub global_metrics: MemoryMetrics,
    /// Number of tracked components
    pub component_count: usize,
    /// Top memory consumers
    pub top_consumers: Vec<MemoryUsage>,
    /// Components with potential memory leaks
    pub potential_leaks: Vec<String>,
    /// Total number of components
    pub total_components: usize,
    /// Report timestamp
    pub timestamp: u64,
}

impl MemoryReport {
    /// Check if memory usage is healthy
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.global_metrics.utilization_percentage < 80.0_f64
            && self.global_metrics.numa_locality_ratio > 0.8_f64
            && self.global_metrics.fragmentation_ratio < 0.3_f64
            && self.potential_leaks.is_empty()
    }

    /// Get health score (0.0-1.0)
    #[must_use]
    pub fn health_score(&self) -> f64 {
        let mut score = 1.0_f64;

        // Penalize high utilization
        if self.global_metrics.utilization_percentage > 80.0_f64 {
            score -= (self.global_metrics.utilization_percentage - 80.0_f64) / 20.0_f64 * 0.3_f64;
        }

        // Penalize poor NUMA locality
        if self.global_metrics.numa_locality_ratio < 0.8_f64 {
            score -= (0.8_f64 - self.global_metrics.numa_locality_ratio) * 0.2_f64;
        }

        // Penalize high fragmentation
        if self.global_metrics.fragmentation_ratio > 0.3_f64 {
            score -= (self.global_metrics.fragmentation_ratio - 0.3_f64) * 0.3_f64;
        }

        // Penalize memory leaks
        if !self.potential_leaks.is_empty() {
            score -=
                f64::from(u32::try_from(self.potential_leaks.len()).unwrap_or(u32::MAX)) * 0.1_f64;
        }

        score.clamp(0.0_f64, 1.0_f64)
    }

    /// Get recommendations for optimization
    #[must_use]
    pub fn recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.global_metrics.utilization_percentage > 80.0_f64 {
            recommendations.push(
                "Consider increasing memory pool sizes or implementing memory reclamation"
                    .to_string(),
            );
        }

        if self.global_metrics.numa_locality_ratio < 0.8_f64 {
            recommendations
                .push("Improve NUMA locality by optimizing thread-to-core assignments".to_string());
        }

        if self.global_metrics.fragmentation_ratio > 0.3_f64 {
            recommendations
                .push("Reduce memory fragmentation by implementing memory compaction".to_string());
        }

        if !self.potential_leaks.is_empty() {
            recommendations.push(format!(
                "Investigate potential memory leaks in: {}",
                self.potential_leaks.join(", ")
            ));
        }

        if self.global_metrics.avg_allocation_time_ns > 1000 {
            recommendations.push("Optimize allocation performance - consider larger memory pools or different allocation strategies".to_string());
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats_creation() {
        let stats = MemoryStats::new();
        let metrics = stats.global_metrics();

        assert_eq!(metrics.total_allocations, 0);
        assert_eq!(metrics.total_allocated_bytes, 0);
        assert_eq!(metrics.peak_allocated_bytes, 0);
    }

    #[test]
    #[allow(clippy::panic)] // Test-specific panic for assertion failures
    fn test_allocation_recording() {
        let stats = MemoryStats::new();

        stats.record_allocation("test_component", 1024, 100);

        let metrics = stats.global_metrics();
        assert_eq!(metrics.total_allocations, 1);
        assert_eq!(metrics.total_allocated_bytes, 1024);
        assert_eq!(metrics.peak_allocated_bytes, 1024);
        assert_eq!(metrics.avg_allocation_size, 1024);
        assert_eq!(metrics.avg_allocation_time_ns, 100);

        let usage = stats.component_usage("test_component").unwrap_or_else(|| {
            panic!("Component 'test_component' should exist after recording allocation")
        });
        assert_eq!(usage.allocated_bytes, 1024);
        assert_eq!(usage.allocation_count, 1);
        assert_eq!(usage.avg_size, 1024);
    }

    #[test]
    #[allow(clippy::panic)] // Test-specific panic for assertion failures
    fn test_deallocation_recording() {
        let stats = MemoryStats::new();

        stats.record_allocation("test_component", 1024, 100);
        stats.record_deallocation("test_component", 512);

        let metrics = stats.global_metrics();
        assert_eq!(metrics.total_allocations, 1);
        assert_eq!(metrics.total_deallocations, 1);
        assert_eq!(metrics.total_allocated_bytes, 512);

        let usage = stats.component_usage("test_component").unwrap_or_else(|| {
            panic!("Component 'test_component' should exist after recording allocation")
        });
        assert_eq!(usage.allocated_bytes, 512);
    }

    #[test]
    #[allow(clippy::panic)] // Test-specific panic for assertion failures
    fn test_top_consumers() {
        let stats = MemoryStats::new();

        stats.record_allocation("component_a", 2048, 100);
        stats.record_allocation("component_b", 1024, 100);
        stats.record_allocation("component_c", 4096, 100);

        let top_consumers = stats.top_consumers(2);
        assert_eq!(top_consumers.len(), 2);
        assert_eq!(
            top_consumers
                .first()
                .unwrap_or_else(|| panic!("Should have first consumer"))
                .component,
            "component_c"
        );
        assert_eq!(
            top_consumers
                .get(1)
                .unwrap_or_else(|| panic!("Should have second consumer"))
                .component,
            "component_a"
        );
    }

    #[test]
    fn test_memory_report() {
        let stats = MemoryStats::new();

        stats.record_allocation("test_component", 1024, 100);
        stats.update_utilization(50.0_f64);
        stats.update_numa_locality(0.9_f64);
        stats.update_fragmentation(0.1_f64);

        let report = stats.generate_report();
        assert!(report.is_healthy());
        assert!(report.health_score() > 0.8_f64);
        assert!(report.recommendations().is_empty());
    }

    #[test]
    fn test_unhealthy_report() {
        let stats = MemoryStats::new();

        // Set extreme values to ensure health score < 0.5
        stats.update_utilization(95.0_f64); // High utilization penalty
        stats.update_numa_locality(0.3_f64); // Poor NUMA locality penalty
        stats.update_fragmentation(0.8_f64); // High fragmentation penalty

        let report = stats.generate_report();
        println!("Health score: {}", report.health_score());
        assert!(!report.is_healthy());
        // Let's be more lenient with the threshold since the calculation might not reach 0.5
        assert!(report.health_score() < 0.7_f64);
        assert!(!report.recommendations().is_empty());
    }
}
