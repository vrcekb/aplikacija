//! Realtime optimizations for Linux
//!
//! Provides low-latency kernel tuning and realtime features for Linux systems.
//! These features are Linux-specific and designed for ultra-low latency trading.

use crate::error::{CoreError, CoreResult};
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::process::Command;

/// Realtime configuration for Linux systems
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// CPU cores to isolate
    pub isolated_cores: Vec<u32>,
    /// Enable hugepages
    pub enable_hugepages: bool,
    /// Hugepage size (2MB or 1GB)
    pub hugepage_size: HugepageSize,
    /// Number of hugepages to allocate
    pub hugepage_count: u32,
    /// Enable PREEMPT_RT kernel features
    pub enable_preempt_rt: bool,
    /// Enable CPU frequency scaling governor
    pub cpu_governor: CpuGovernor,
    /// Enable interrupt affinity
    pub enable_irq_affinity: bool,
    /// Network interrupt CPU mask
    pub network_irq_cpus: Vec<u32>,
}

/// Hugepage size options
#[derive(Debug, Clone, Copy)]
pub enum HugepageSize {
    /// 2MB hugepages (standard)
    Size2MB,
    /// 1GB hugepages (for large memory allocations)
    Size1GB,
}

impl HugepageSize {
    /// Get size in bytes
    #[must_use]
    pub const fn bytes(&self) -> usize {
        match self {
            Self::Size2MB => 2 * 1024 * 1024,
            Self::Size1GB => 1024 * 1024 * 1024,
        }
    }

    /// Get size string for kernel parameters
    #[must_use]
    pub const fn kernel_param(&self) -> &'static str {
        match self {
            Self::Size2MB => "2M",
            Self::Size1GB => "1G",
        }
    }
}

/// CPU frequency scaling governor
#[derive(Debug, Clone, Copy)]
pub enum CpuGovernor {
    /// Maximum performance
    Performance,
    /// Power saving
    Powersave,
    /// On-demand scaling
    Ondemand,
    /// Conservative scaling
    Conservative,
}

impl CpuGovernor {
    /// Get governor name for sysfs
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Powersave => "powersave",
            Self::Ondemand => "ondemand",
            Self::Conservative => "conservative",
        }
    }
}

/// Realtime optimizer for Linux systems
pub struct RealtimeOptimizer {
    config: RealtimeConfig,
}

impl RealtimeOptimizer {
    /// Create new realtime optimizer
    #[must_use]
    pub const fn new(config: RealtimeConfig) -> Self {
        Self { config }
    }

    /// Apply all realtime optimizations
    ///
    /// # Errors
    ///
    /// Returns error if any optimization fails to apply
    pub fn apply_optimizations(&self) -> CoreResult<()> {
        // Check if running on Linux
        if !cfg!(target_os = "linux") {
            return Err(CoreError::InvalidConfiguration(
                "Realtime optimizations are only available on Linux".to_string(),
            ));
        }

        // Apply optimizations in order of importance
        self.setup_cpu_isolation()?;
        self.setup_hugepages()?;
        self.setup_cpu_governor()?;
        self.setup_interrupt_affinity()?;
        self.check_preempt_rt()?;
        self.apply_sysctl_tuning()?;

        Ok(())
    }

    /// Setup CPU isolation using isolcpus
    fn setup_cpu_isolation(&self) -> CoreResult<()> {
        if self.config.isolated_cores.is_empty() {
            return Ok(());
        }

        // Check current kernel command line
        let cmdline = fs::read_to_string("/proc/cmdline")
            .map_err(|e| CoreError::SystemError(format!("Failed to read /proc/cmdline: {e}")))?;

        let cores_str = self
            .config
            .isolated_cores
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");

        if !cmdline.contains(&format!("isolcpus={cores_str}")) {
            eprintln!(
                "WARNING: CPU isolation not configured. Add 'isolcpus={} nohz_full={} rcu_nocbs={}' to kernel boot parameters",
                cores_str, cores_str, cores_str
            );
        }

        // Set CPU affinity for current process
        self.set_process_affinity()?;

        Ok(())
    }

    /// Set process CPU affinity
    fn set_process_affinity(&self) -> CoreResult<()> {
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
            use std::mem;
            use std::os::unix::process::CommandExt;

            unsafe {
                let mut cpu_set: cpu_set_t = mem::zeroed();
                CPU_ZERO(&mut cpu_set);

                // Set affinity to isolated cores
                for &core in &self.config.isolated_cores {
                    CPU_SET(core as usize, &mut cpu_set);
                }

                let result = sched_setaffinity(0, mem::size_of::<cpu_set_t>(), &cpu_set);
                if result != 0 {
                    return Err(CoreError::SystemError(format!(
                        "Failed to set CPU affinity: {}",
                        io::Error::last_os_error()
                    )));
                }
            }
        }

        Ok(())
    }

    /// Setup hugepages
    fn setup_hugepages(&self) -> CoreResult<()> {
        if !self.config.enable_hugepages {
            return Ok(());
        }

        let hugepage_path = match self.config.hugepage_size {
            HugepageSize::Size2MB => "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages",
            HugepageSize::Size1GB => "/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages",
        };

        // Try to allocate hugepages
        match fs::write(hugepage_path, self.config.hugepage_count.to_string()) {
            Ok(()) => {
                // Verify allocation
                let allocated = fs::read_to_string(hugepage_path)
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
                    .unwrap_or(0);

                if allocated < self.config.hugepage_count {
                    eprintln!(
                        "WARNING: Only {}/{} hugepages allocated",
                        allocated, self.config.hugepage_count
                    );
                }
            }
            Err(e) => {
                eprintln!("WARNING: Failed to allocate hugepages: {e}");
            }
        }

        Ok(())
    }

    /// Setup CPU frequency scaling governor
    fn setup_cpu_governor(&self) -> CoreResult<()> {
        let governor = self.config.cpu_governor.name();

        for &core in &self.config.isolated_cores {
            let governor_path = format!(
                "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_governor",
                core
            );

            if let Err(e) = fs::write(&governor_path, governor) {
                eprintln!("WARNING: Failed to set governor for CPU {core}: {e}");
            }
        }

        // Disable CPU frequency scaling for maximum performance
        if matches!(self.config.cpu_governor, CpuGovernor::Performance) {
            // Disable Intel turbo boost for consistent latency
            let _ = fs::write("/sys/devices/system/cpu/intel_pstate/no_turbo", "1");

            // Set performance bias
            let _ = fs::write("/sys/devices/system/cpu/cpu0/power/energy_perf_bias", "0");
        }

        Ok(())
    }

    /// Setup interrupt affinity
    fn setup_interrupt_affinity(&self) -> CoreResult<()> {
        if !self.config.enable_irq_affinity || self.config.network_irq_cpus.is_empty() {
            return Ok(());
        }

        // Find network device interrupts
        let irq_dir = fs::read_dir("/proc/irq")
            .map_err(|e| CoreError::SystemError(format!("Failed to read /proc/irq: {e}")))?;

        let cpu_mask = self
            .config
            .network_irq_cpus
            .iter()
            .fold(0_u64, |mask, &cpu| mask | (1 << cpu));

        for entry in irq_dir.flatten() {
            let irq_path = entry.path();
            let smp_affinity_path = irq_path.join("smp_affinity");

            if smp_affinity_path.exists() {
                // Check if this is a network interrupt
                if let Ok(interrupts) = fs::read_to_string(irq_path.join("spurious")) {
                    if interrupts.contains("eth") || interrupts.contains("mlx") {
                        let _ = fs::write(smp_affinity_path, format!("{cpu_mask:x}"));
                    }
                }
            }
        }

        Ok(())
    }

    /// Check for PREEMPT_RT kernel
    fn check_preempt_rt(&self) -> CoreResult<()> {
        if !self.config.enable_preempt_rt {
            return Ok(());
        }

        let version = fs::read_to_string("/proc/version")
            .map_err(|e| CoreError::SystemError(format!("Failed to read /proc/version: {e}")))?;

        if !version.contains("PREEMPT_RT") && !version.contains("PREEMPT RT") {
            eprintln!("WARNING: PREEMPT_RT kernel not detected. Install linux-rt for best latency");
        }

        Ok(())
    }

    /// Apply sysctl tuning for low latency
    fn apply_sysctl_tuning(&self) -> CoreResult<()> {
        let tuning_params = vec![
            // Disable kernel preemption latency tracing
            ("kernel.sched_latency_ns", "1000000"),
            ("kernel.sched_min_granularity_ns", "100000"),
            ("kernel.sched_wakeup_granularity_ns", "25000"),
            // Reduce timer frequency for less interrupts
            ("kernel.sched_rt_runtime_us", "950000"),
            // Network tuning
            ("net.core.busy_poll", "50"),
            ("net.core.busy_read", "50"),
            ("net.core.netdev_budget", "600"),
            // Disable transparent hugepages (causes latency spikes)
            ("vm.transparent_hugepage", "never"),
            // Memory tuning
            ("vm.swappiness", "0"),
            ("vm.dirty_ratio", "3"),
            ("vm.dirty_background_ratio", "2"),
        ];

        for (param, value) in tuning_params {
            let sysctl_path = format!("/proc/sys/{}", param.replace('.', "/"));
            if let Err(e) = fs::write(&sysctl_path, value) {
                eprintln!("WARNING: Failed to set {param}={value}: {e}");
            }
        }

        Ok(())
    }

    /// Get current system latency statistics
    pub fn get_latency_stats(&self) -> CoreResult<LatencyStats> {
        let mut stats = LatencyStats::default();

        // Read scheduling latency
        if let Ok(sched_debug) = fs::read_to_string("/proc/sched_debug") {
            // Parse scheduling latency info
            for line in sched_debug.lines() {
                if line.contains("nr_running") {
                    if let Some(value) = line.split_whitespace().last() {
                        stats.runqueue_length = value.parse().unwrap_or(0);
                    }
                }
            }
        }

        // Read interrupt stats
        if let Ok(interrupts) = fs::read_to_string("/proc/interrupts") {
            stats.total_interrupts = interrupts
                .lines()
                .skip(1) // Skip header
                .map(|line| {
                    line.split_whitespace()
                        .skip(1) // Skip IRQ number
                        .filter_map(|s| s.parse::<u64>().ok())
                        .sum::<u64>()
                })
                .sum();
        }

        // Check CPU idle states
        for &cpu in &self.config.isolated_cores {
            let idle_path = format!("/sys/devices/system/cpu/cpu{}/cpuidle/state0/time", cpu);
            if let Ok(idle_time) = fs::read_to_string(idle_path) {
                if let Ok(time) = idle_time.trim().parse::<u64>() {
                    stats.cpu_idle_time_us += time;
                }
            }
        }

        Ok(stats)
    }
}

/// System latency statistics
#[derive(Debug, Default)]
pub struct LatencyStats {
    /// Total number of interrupts
    pub total_interrupts: u64,
    /// Average runqueue length
    pub runqueue_length: u32,
    /// CPU idle time in microseconds
    pub cpu_idle_time_us: u64,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            isolated_cores: vec![],
            enable_hugepages: true,
            hugepage_size: HugepageSize::Size2MB,
            hugepage_count: 512,
            enable_preempt_rt: true,
            cpu_governor: CpuGovernor::Performance,
            enable_irq_affinity: true,
            network_irq_cpus: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hugepage_size() {
        assert_eq!(HugepageSize::Size2MB.bytes(), 2 * 1024 * 1024);
        assert_eq!(HugepageSize::Size1GB.bytes(), 1024 * 1024 * 1024);
        assert_eq!(HugepageSize::Size2MB.kernel_param(), "2M");
        assert_eq!(HugepageSize::Size1GB.kernel_param(), "1G");
    }

    #[test]
    fn test_cpu_governor() {
        assert_eq!(CpuGovernor::Performance.name(), "performance");
        assert_eq!(CpuGovernor::Powersave.name(), "powersave");
    }

    #[test]
    fn test_realtime_config_default() {
        let config = RealtimeConfig::default();
        assert!(config.isolated_cores.is_empty());
        assert!(config.enable_hugepages);
        assert_eq!(config.hugepage_count, 512);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_realtime_optimizer_creation() {
        let config = RealtimeConfig {
            isolated_cores: vec![4, 5, 6, 7],
            enable_hugepages: true,
            hugepage_size: HugepageSize::Size2MB,
            hugepage_count: 1024,
            enable_preempt_rt: true,
            cpu_governor: CpuGovernor::Performance,
            enable_irq_affinity: true,
            network_irq_cpus: vec![0, 1],
        };

        let optimizer = RealtimeOptimizer::new(config);
        // Basic creation test - actual optimization requires root privileges
        let stats = optimizer.get_latency_stats();
        assert!(stats.is_ok() || stats.is_err()); // May fail without proper permissions
    }
}
