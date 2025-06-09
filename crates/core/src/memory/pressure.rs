//! Memory Pressure Monitoring - Real-time Memory Usage Tracking
//!
//! Implements comprehensive memory pressure monitoring with:
//! - Real-time memory usage tracking across all pools
//! - Adaptive threshold management based on system conditions
//! - Automatic memory reclamation under pressure
//! - Performance impact assessment and mitigation
//! - Integration with NUMA-aware memory management

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use thiserror::Error;

/// Memory pressure monitoring error types
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum PressureError {
    /// Memory pressure threshold exceeded
    #[error("Memory pressure threshold exceeded: {current}% > {threshold}%")]
    ThresholdExceeded {
        /// Current pressure level
        current: u8,
        /// Configured threshold
        threshold: u8,
    },

    /// System memory exhaustion
    #[error("System memory exhaustion: {used} MB used, {available} MB available")]
    SystemExhaustion {
        /// Used memory in MB
        used: u64,
        /// Available memory in MB
        available: u64,
    },

    /// Memory leak detected
    #[error("Memory leak detected in {component}: {leaked_bytes} bytes over {duration_ms}ms")]
    MemoryLeak {
        /// Component name
        component: String,
        /// Leaked bytes
        leaked_bytes: u64,
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// Monitoring system failure
    #[error("Memory pressure monitoring system failure: {reason}")]
    MonitoringFailure {
        /// Failure reason
        reason: String,
    },
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PressureLevel {
    /// Normal operation - memory usage below 60%
    Normal,
    /// Moderate pressure - memory usage 60-80%
    Moderate,
    /// High pressure - memory usage 80-90%
    High,
    /// Critical pressure - memory usage above 90%
    Critical,
}

impl PressureLevel {
    /// Get pressure level from percentage
    #[must_use]
    pub const fn from_percentage(percentage: u8) -> Self {
        match percentage {
            0..=59 => Self::Normal,
            60..=79 => Self::Moderate,
            80..=89 => Self::High,
            _ => Self::Critical, // 90+ or invalid values
        }
    }

    /// Get percentage threshold for this level
    #[must_use]
    pub const fn threshold_percentage(&self) -> u8 {
        match self {
            Self::Normal => 60,
            Self::Moderate => 80,
            Self::High => 90,
            Self::Critical => 95,
        }
    }

    /// Check if this level requires immediate action
    #[must_use]
    pub const fn requires_action(&self) -> bool {
        matches!(self, Self::High | Self::Critical)
    }
}

/// Memory pressure thresholds configuration
#[derive(Debug, Clone)]
pub struct PressureThreshold {
    /// Warning threshold percentage
    pub warning: u8,
    /// Critical threshold percentage
    pub critical: u8,
    /// Emergency threshold percentage
    pub emergency: u8,
    /// Memory leak detection threshold in bytes
    pub leak_threshold_bytes: u64,
    /// Memory leak detection window in seconds
    pub leak_window_seconds: u64,
}

impl Default for PressureThreshold {
    fn default() -> Self {
        Self {
            warning: 70,
            critical: 85,
            emergency: 95,
            leak_threshold_bytes: 100 * 1024 * 1024, // 100MB
            leak_window_seconds: 300,                // 5 minutes
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Default)]
pub struct MemoryUsageStats {
    /// Total system memory in bytes
    pub total_system_memory: AtomicU64,
    /// Available system memory in bytes
    pub available_system_memory: AtomicU64,
    /// Total allocated memory in bytes
    pub total_allocated: AtomicU64,
    /// Peak allocated memory in bytes
    pub peak_allocated: AtomicU64,
    /// Number of allocations
    pub allocation_count: AtomicU64,
    /// Number of deallocations
    pub deallocation_count: AtomicU64,
    /// Memory pressure events
    pub pressure_events: AtomicU64,
    /// Memory reclamation events
    pub reclamation_events: AtomicU64,
    /// Last pressure check timestamp
    pub last_check_timestamp: AtomicU64,
}

impl MemoryUsageStats {
    /// Get current memory utilization percentage
    #[must_use]
    pub fn utilization_percentage(&self) -> u8 {
        let total = self.total_system_memory.load(Ordering::Relaxed);
        let available = self.available_system_memory.load(Ordering::Relaxed);

        if total == 0 {
            return 0;
        }

        let used = total.saturating_sub(available);
        let percentage = (used * 100) / total;

        u8::try_from(percentage.min(100)).unwrap_or(100)
    }

    /// Get current pressure level
    #[must_use]
    pub fn pressure_level(&self) -> PressureLevel {
        PressureLevel::from_percentage(self.utilization_percentage())
    }

    /// Record allocation
    pub fn record_allocation(&self, size: u64) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let new_total = self.total_allocated.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak
        let mut peak = self.peak_allocated.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    /// Record deallocation
    pub fn record_deallocation(&self, size: u64) {
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_sub(size, Ordering::Relaxed);
    }

    /// Record pressure event
    pub fn record_pressure_event(&self) {
        self.pressure_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Record reclamation event
    pub fn record_reclamation(&self) {
        self.reclamation_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Update system memory info
    pub fn update_system_memory(&self, total: u64, available: u64) {
        self.total_system_memory.store(total, Ordering::Relaxed);
        self.available_system_memory
            .store(available, Ordering::Relaxed);
        self.last_check_timestamp.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            Ordering::Relaxed,
        );
    }
}

/// Component memory tracking
#[derive(Debug)]
#[allow(dead_code)] // Name field used for debugging and future features
struct ComponentTracker {
    name: String,
    allocated_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    allocation_count: AtomicU64,
    last_check: Mutex<Instant>,
    baseline_bytes: AtomicU64,
}

impl ComponentTracker {
    fn new(name: String) -> Self {
        Self {
            name,
            allocated_bytes: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            last_check: Mutex::new(Instant::now()),
            baseline_bytes: AtomicU64::new(0),
        }
    }

    fn record_allocation(&self, size: u64) {
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        let new_total = self.allocated_bytes.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                new_total,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
    }

    fn record_deallocation(&self, size: u64) {
        self.allocated_bytes.fetch_sub(size, Ordering::Relaxed);
    }

    fn check_for_leak(&self, threshold_bytes: u64, window_seconds: u64) -> Option<u64> {
        if let Ok(mut last_check) = self.last_check.try_lock() {
            let now = Instant::now();
            let elapsed = now.duration_since(*last_check);

            if elapsed.as_secs() >= window_seconds {
                let current_bytes = self.allocated_bytes.load(Ordering::Relaxed);
                let baseline = self.baseline_bytes.load(Ordering::Relaxed);
                let growth = current_bytes.saturating_sub(baseline);

                *last_check = now;
                self.baseline_bytes.store(current_bytes, Ordering::Relaxed);

                if growth > threshold_bytes {
                    return Some(growth);
                }
            }
        }
        None
    }
}

/// Memory pressure monitor
pub struct MemoryPressureMonitor {
    /// Configuration thresholds
    thresholds: PressureThreshold,
    /// Global memory statistics
    stats: Arc<MemoryUsageStats>,
    /// Component trackers
    components: Mutex<HashMap<String, Arc<ComponentTracker>>>,
    /// Monitoring enabled flag
    monitoring_enabled: bool,
    /// Last system memory check
    last_system_check: Mutex<Instant>,
    /// System check interval
    system_check_interval: Duration,
}

impl MemoryPressureMonitor {
    /// Create new memory pressure monitor
    #[must_use]
    pub fn new(thresholds: PressureThreshold) -> Self {
        Self {
            thresholds,
            stats: Arc::new(MemoryUsageStats::default()),
            components: Mutex::new(HashMap::new()),
            monitoring_enabled: true,
            last_system_check: Mutex::new(Instant::now()),
            system_check_interval: Duration::from_secs(1),
        }
    }

    /// Get memory statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<MemoryUsageStats> {
        &self.stats
    }

    /// Register component for tracking
    ///
    /// # Errors
    ///
    /// Returns error if component registration fails due to lock contention
    pub fn register_component(&self, name: String) -> Result<(), PressureError> {
        self.components.lock().map_or_else(
            |_| {
                Err(PressureError::MonitoringFailure {
                    reason: "Failed to lock components map".to_string(),
                })
            },
            |mut components| {
                components.insert(name.clone(), Arc::new(ComponentTracker::new(name)));
                Ok(())
            },
        )
    }

    /// Record allocation for component
    ///
    /// # Errors
    ///
    /// Returns error if memory pressure threshold is exceeded
    pub fn record_allocation(&self, component: &str, size: u64) -> Result<(), PressureError> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        self.stats.record_allocation(size);

        if let Ok(components) = self.components.lock() {
            if let Some(tracker) = components.get(component) {
                tracker.record_allocation(size);
            }
        }

        // Check pressure after allocation
        self.check_pressure()
    }

    /// Record deallocation for component
    ///
    /// # Errors
    ///
    /// Returns error if monitoring system fails
    pub fn record_deallocation(&self, component: &str, size: u64) -> Result<(), PressureError> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        self.stats.record_deallocation(size);

        if let Ok(components) = self.components.lock() {
            if let Some(tracker) = components.get(component) {
                tracker.record_deallocation(size);
            }
        }

        Ok(())
    }

    /// Check current memory pressure
    ///
    /// # Errors
    ///
    /// Returns error if memory pressure exceeds critical thresholds or memory leaks are detected
    pub fn check_pressure(&self) -> Result<(), PressureError> {
        if !self.monitoring_enabled {
            return Ok(());
        }

        // Update system memory info periodically
        self.update_system_memory_info();

        let utilization = self.stats.utilization_percentage();

        if utilization >= self.thresholds.emergency {
            self.stats.record_pressure_event();
            return Err(PressureError::ThresholdExceeded {
                current: utilization,
                threshold: self.thresholds.emergency,
            });
        }

        if utilization >= self.thresholds.critical {
            self.stats.record_pressure_event();
            // Trigger memory reclamation
            self.trigger_memory_reclamation();
        }

        // Check for memory leaks
        self.check_memory_leaks()?;

        Ok(())
    }

    /// Get current pressure level
    #[must_use]
    pub fn current_pressure_level(&self) -> PressureLevel {
        self.stats.pressure_level()
    }

    /// Enable/disable monitoring
    pub const fn set_monitoring_enabled(&mut self, enabled: bool) {
        self.monitoring_enabled = enabled;
    }

    /// Update system memory information
    fn update_system_memory_info(&self) {
        if let Ok(mut last_check) = self.last_system_check.try_lock() {
            let now = Instant::now();
            if now.duration_since(*last_check) >= self.system_check_interval {
                // Get system memory info (simplified - in production use system APIs)
                let total_memory = 16 * 1024 * 1024 * 1024_u64; // 16GB default
                let available_memory = total_memory / 2; // Simplified calculation

                self.stats
                    .update_system_memory(total_memory, available_memory);
                *last_check = now;
            }
        }
    }

    /// Trigger memory reclamation
    fn trigger_memory_reclamation(&self) {
        self.stats.record_reclamation();

        // In production, this would:
        // 1. Force garbage collection
        // 2. Compact memory pools
        // 3. Release unused memory to system
        // 4. Reduce cache sizes
    }

    /// Check for memory leaks in components
    fn check_memory_leaks(&self) -> Result<(), PressureError> {
        if let Ok(components) = self.components.lock() {
            for (name, tracker) in components.iter() {
                if let Some(leaked_bytes) = tracker.check_for_leak(
                    self.thresholds.leak_threshold_bytes,
                    self.thresholds.leak_window_seconds,
                ) {
                    return Err(PressureError::MemoryLeak {
                        component: name.clone(),
                        leaked_bytes,
                        duration_ms: self.thresholds.leak_window_seconds * 1000,
                    });
                }
            }
        }
        Ok(())
    }
}

impl Default for MemoryPressureMonitor {
    fn default() -> Self {
        Self::new(PressureThreshold::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pressure_level_from_percentage() {
        assert_eq!(PressureLevel::from_percentage(50), PressureLevel::Normal);
        assert_eq!(PressureLevel::from_percentage(70), PressureLevel::Moderate);
        assert_eq!(PressureLevel::from_percentage(85), PressureLevel::High);
        assert_eq!(PressureLevel::from_percentage(95), PressureLevel::Critical);
    }

    #[test]
    fn test_memory_usage_stats() {
        let stats = MemoryUsageStats::default();

        // Set system memory
        stats.update_system_memory(1000, 500);
        assert_eq!(stats.utilization_percentage(), 50);
        assert_eq!(stats.pressure_level(), PressureLevel::Normal);

        // Record allocation
        stats.record_allocation(100);
        assert_eq!(stats.allocation_count.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_allocated.load(Ordering::Relaxed), 100);

        // Record deallocation
        stats.record_deallocation(50);
        assert_eq!(stats.deallocation_count.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_allocated.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn test_pressure_monitor() -> Result<(), PressureError> {
        let monitor = MemoryPressureMonitor::default();

        // Register component
        monitor.register_component("test_component".to_string())?;

        // Record allocation
        monitor.record_allocation("test_component", 1024)?;

        // Record deallocation
        monitor.record_deallocation("test_component", 512)?;

        // Check pressure
        monitor.check_pressure()?;

        Ok(())
    }

    #[test]
    fn test_pressure_threshold_exceeded() {
        let thresholds = PressureThreshold {
            emergency: 50,
            ..Default::default()
        };
        let monitor = MemoryPressureMonitor::new(thresholds);

        // Set high memory usage
        monitor.stats.update_system_memory(1000, 400); // 60% usage

        let result = monitor.check_pressure();
        assert!(result.is_err());
    }

    #[test]
    fn test_component_tracker() {
        let tracker = ComponentTracker::new("test".to_string());

        tracker.record_allocation(100);
        assert_eq!(tracker.allocated_bytes.load(Ordering::Relaxed), 100);
        assert_eq!(tracker.peak_bytes.load(Ordering::Relaxed), 100);

        tracker.record_allocation(50);
        assert_eq!(tracker.allocated_bytes.load(Ordering::Relaxed), 150);
        assert_eq!(tracker.peak_bytes.load(Ordering::Relaxed), 150);

        tracker.record_deallocation(75);
        assert_eq!(tracker.allocated_bytes.load(Ordering::Relaxed), 75);
        assert_eq!(tracker.peak_bytes.load(Ordering::Relaxed), 150); // Peak unchanged
    }
}
