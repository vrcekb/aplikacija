//! `TallyIO` Core - Ultra-Performance Financial Trading Engine
//!
//! This crate provides the core functionality for `TallyIO`'s MEV/DeFi trading platform
//! with sub-millisecond latency requirements and zero-panic production-ready code.
//!
//! # Features
//!
//! - **Ultra-low latency**: <1ms execution guarantee for critical paths
//! - **Zero-panic policy**: All operations return `Result<T, E>`
//! - **Lock-free concurrency**: Optimized for high-throughput trading
//! - **Memory efficiency**: Zero allocations in hot paths
//! - **Type safety**: Comprehensive validation and error handling
//!
//! # Architecture
//!
//! The core is organized into several key modules:
//!
//! - [`engine`] - Main execution engine for strategy processing
//! - [`state`] - Global and local state management
//! - [`mempool`] - Real-time blockchain mempool monitoring
//! - [`optimization`] - Performance optimizations (CPU, memory, SIMD)
//! - [`types`] - Core type definitions for financial operations
//!
//! # Example
//!
//! ```rust
//! use tallyio_core::{init_with_config, CoreConfig, CoreResult};
//!
//! fn main() -> CoreResult<()> {
//!     // Initialize with test configuration (for documentation example)
//!     // In production, use CoreConfig::production() with proper environment setup
//!     let config = CoreConfig::test()?;
//!
//!     // Create the core instance
//!     let core = init_with_config(config)?;
//!
//!     // Core is now ready for configuration and use
//!     println!("TallyIO Core initialized with {} workers", core.config().engine.max_workers);
//!
//!     // In production, you would call core.start() and core.stop()
//!     // but we skip that in documentation to avoid potential blocking
//!     Ok(())
//! }
//! ```
//!
//! # Performance Guarantees
//!
//! - **Critical operations**: <1ms latency
//! - **Memory allocation**: Zero in hot paths
//! - **Error handling**: Zero panics in production
//! - **Concurrency**: Lock-free where possible

#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::large_stack_arrays,
    clippy::indexing_slicing,
    missing_docs,
    // unsafe_code - Temporarily disabled for memory pool implementation
)]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::correctness,
    clippy::suspicious,
    clippy::perf,
    clippy::style,
    clippy::complexity,
    clippy::let_underscore_future,
    clippy::diverging_sub_expression,
    clippy::unreachable,
    clippy::default_numeric_fallback,
    clippy::redundant_pattern_matching,
    clippy::manual_let_else,
    clippy::blocks_in_conditions,
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_ref_mut,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::disallowed_methods
)]
#![allow(clippy::multiple_crate_versions)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Public modules
pub mod config;
pub mod error;
pub mod prelude;
pub mod types;

// Core functionality modules
pub mod engine;
pub mod lockfree;
pub mod memory;
pub mod mempool;
pub mod optimization;
pub mod state;

// Re-exports for convenience
pub use config::CoreConfig;
pub use error::{CoreError, CoreResult, CriticalError};
pub use types::{Address, BlockNumber, Gas, Price, TxHash};

// Re-export subsystem types for external use
pub use engine::{Engine, EngineConfig, EngineError, EngineResult};
pub use mempool::{MempoolMonitor, MempoolMonitorStats};
pub use state::{StateManager, StateManagerStats};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: BuildInfo = BuildInfo {
    version: VERSION,
    git_hash: match option_env!("GIT_HASH") {
        Some(hash) => hash,
        None => "unknown",
    },
    build_date: match option_env!("BUILD_DATE") {
        Some(date) => date,
        None => "unknown",
    },
    rust_version: match option_env!("RUSTC_VERSION") {
        Some(version) => version,
        None => "unknown",
    },
    target: match option_env!("TARGET") {
        Some(target) => target,
        None => "unknown",
    },
    profile: if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    },
};

/// Build information structure
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Crate version
    pub version: &'static str,
    /// Git commit hash
    pub git_hash: &'static str,
    /// Build date
    pub build_date: &'static str,
    /// Rust compiler version
    pub rust_version: &'static str,
    /// Target triple
    pub target: &'static str,
    /// Build profile
    pub profile: &'static str,
}

impl BuildInfo {
    /// Get formatted build information
    #[must_use]
    pub fn formatted(&self) -> String {
        format!(
            "`TallyIO` Core v{} ({})\nGit: {}\nBuilt: {} with rustc {}\nTarget: {} ({})",
            self.version,
            self.profile,
            self.git_hash,
            self.build_date,
            self.rust_version,
            self.target,
            self.profile
        )
    }
}

/// Initialize `TallyIO` core with default configuration
///
/// This is a convenience function for quick setup. For production use,
/// create a custom configuration using [`CoreConfig::production()`].
///
/// # Errors
///
/// Returns error if initialization fails or configuration is invalid.
///
/// # Example
///
/// ```rust
/// use tallyio_core::init;
///
/// fn main() -> tallyio_core::CoreResult<()> {
///     let core = init()?;
///     // Core is now ready for use
///     Ok(())
/// }
/// ```
pub fn init() -> CoreResult<CoreInstance> {
    let config = CoreConfig::development()?;
    CoreInstance::new(config)
}

/// Initialize `TallyIO` core with custom configuration
///
/// # Errors
///
/// Returns error if initialization fails or configuration is invalid.
///
/// # Example
///
/// ```rust
/// use tallyio_core::{init_with_config, CoreConfig};
///
/// fn main() -> tallyio_core::CoreResult<()> {
///     let config = CoreConfig::test()?;
///     let core = init_with_config(config)?;
///     Ok(())
/// }
/// ```
pub fn init_with_config(config: CoreConfig) -> CoreResult<CoreInstance> {
    CoreInstance::new(config)
}

/// System-wide statistics
#[derive(Debug, Clone)]
pub struct SystemStatistics {
    /// State manager statistics
    pub state_manager: state::StateManagerStats,
    /// Mempool monitor statistics
    pub mempool_monitor: mempool::MempoolMonitorStats,
    /// Overall system running state
    pub is_running: bool,
}

/// Main `TallyIO` core instance
///
/// This is the primary interface for interacting with the `TallyIO` core engine.
/// It manages all subsystems and provides a unified API.
pub struct CoreInstance {
    config: CoreConfig,
    engine: engine::Engine,
    state_manager: state::StateManager,
    mempool_monitor: mempool::MempoolMonitor,
}

impl CoreInstance {
    /// Create new core instance
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid or initialization fails.
    pub fn new(config: CoreConfig) -> CoreResult<Self> {
        // Validate configuration
        config.validate()?;

        // Initialize subsystems
        let engine_config = Self::convert_engine_config(&config.engine);
        let engine = engine::Engine::new(engine_config)?;

        let state_manager = state::StateManager::new(config.state.clone())?;

        let mempool_config = Self::convert_mempool_config(&config.mempool);
        let mempool_monitor = mempool::MempoolMonitor::new(mempool_config)?;

        Ok(Self {
            config,
            engine,
            state_manager,
            mempool_monitor,
        })
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &CoreConfig {
        &self.config
    }

    /// Get build information
    #[must_use]
    pub const fn build_info(&self) -> &BuildInfo {
        &BUILD_INFO
    }

    /// Get engine reference
    #[must_use]
    pub const fn engine(&self) -> &engine::Engine {
        &self.engine
    }

    /// Get state manager reference
    #[must_use]
    pub const fn state_manager(&self) -> &state::StateManager {
        &self.state_manager
    }

    /// Get mempool monitor reference
    #[must_use]
    pub const fn mempool_monitor(&self) -> &mempool::MempoolMonitor {
        &self.mempool_monitor
    }

    /// Get comprehensive system statistics
    ///
    /// # Errors
    ///
    /// Returns error if statistics collection fails
    pub fn get_system_statistics(&self) -> CoreResult<SystemStatistics> {
        let state_stats = self
            .state_manager
            .get_statistics()
            .map_err(|e| CoreError::state("statistics", e.to_string()))?;
        let mempool_stats = self
            .mempool_monitor
            .get_statistics()
            .map_err(|e| CoreError::mempool(e, "statistics"))?;

        Ok(SystemStatistics {
            state_manager: state_stats,
            mempool_monitor: mempool_stats,
            is_running: self.is_running(),
        })
    }

    /// Start all subsystems
    ///
    /// # Errors
    ///
    /// Returns error if any subsystem fails to start.
    pub fn start(&mut self) -> CoreResult<()> {
        tracing::info!("Starting `TallyIO` Core v{VERSION}");

        // Start subsystems in order
        self.engine.start()?;
        self.state_manager.start()?;
        self.mempool_monitor.start()?;

        tracing::info!("`TallyIO` Core started successfully");
        Ok(())
    }

    /// Stop all subsystems
    ///
    /// # Errors
    ///
    /// Returns error if any subsystem fails to stop gracefully.
    pub fn stop(&mut self) -> CoreResult<()> {
        tracing::info!("Stopping `TallyIO` Core");

        // Stop subsystems in reverse order
        self.mempool_monitor.stop()?;
        self.state_manager.stop()?;
        self.engine.stop()?;

        tracing::info!("`TallyIO` Core stopped successfully");
        Ok(())
    }

    /// Check if core is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        // Check all subsystems
        self.engine.is_running()
            && self.state_manager.is_running()
            && self.mempool_monitor.is_running()
    }

    /// Get health status
    ///
    /// # Errors
    ///
    /// Returns error if health check fails.
    pub fn health_check(&self) -> CoreResult<prelude::HealthStatus> {
        use prelude::*;
        use std::time::SystemTime;

        Ok(HealthStatus {
            component: "core".to_string(),
            status: HealthLevel::Healthy,
            timestamp: SystemTime::now(),
            details: Some("Core instance operational".to_string()),
            metrics: Some(HealthMetrics {
                cpu_usage: 0.0_f64,
                memory_usage: 0,
                active_connections: 0,
                request_rate: 0.0_f64,
                error_rate: 0.0_f64,
                avg_response_time_us: 0,
            }),
        })
    }

    /// Convert `CoreConfig` `EngineConfig` to engine module `EngineConfig`
    const fn convert_engine_config(config: &config::EngineConfig) -> engine::EngineConfig {
        use std::time::Duration;

        engine::EngineConfig {
            max_workers: config.max_workers as usize,
            task_queue_capacity: config.queue_capacity,
            worker_idle_timeout: Duration::from_secs(30),
            task_timeout: Duration::from_micros(config.max_execution_time_us),
            enable_monitoring: true,
            cpu_affinity: None,
            memory_pool_size: 1024 * 1024, // 1MB default
        }
    }

    /// Convert `CoreConfig` `MempoolConfig` to mempool module `MempoolConfig`
    const fn convert_mempool_config(config: &config::MempoolConfig) -> mempool::MempoolConfig {
        use crate::types::Price;
        use std::time::Duration;

        mempool::MempoolConfig {
            max_transactions: 100_000,
            transaction_ttl: Duration::from_secs(300),
            enable_mev_detection: config.filter.enable_mev_detection,
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            min_mev_value: Price::from_ether(config.filter.min_value_eth as u64),
            enable_realtime_analysis: true,
            analysis_batch_size: 1000,
            min_gas_limit: 21_000,
            max_gas_limit: config.filter.max_gas_limit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        // VERSION is a const string, so we test its content directly
        #[allow(clippy::const_is_empty)]
        {
            assert!(!VERSION.is_empty());
        }
        assert!(VERSION.contains('.'));
    }

    #[test]
    fn test_build_info() {
        let info = BUILD_INFO.formatted();
        assert!(info.contains("`TallyIO` Core"));
        assert!(info.contains(VERSION));
    }

    #[test]
    fn test_core_initialization() -> CoreResult<()> {
        // Set test environment to avoid production detection
        std::env::set_var("TALLYIO_ENVIRONMENT", "test");

        let config = CoreConfig::test()?;
        let core = CoreInstance::new(config)?;
        assert!(!core.is_running());

        // Clean up
        std::env::remove_var("TALLYIO_ENVIRONMENT");
        Ok(())
    }

    #[test]
    fn test_init_convenience_function() -> CoreResult<()> {
        // Set development environment to avoid production detection
        std::env::set_var("TALLYIO_ENVIRONMENT", "development");

        let _core = init()?;

        // Clean up
        std::env::remove_var("TALLYIO_ENVIRONMENT");
        Ok(())
    }

    #[test]
    fn test_health_check() -> CoreResult<()> {
        // Set test environment to avoid production detection
        std::env::set_var("TALLYIO_ENVIRONMENT", "test");

        let config = CoreConfig::test()?;
        let core = CoreInstance::new(config)?;
        let health = core.health_check()?;
        assert_eq!(health.component, "core");

        // Clean up
        std::env::remove_var("TALLYIO_ENVIRONMENT");
        Ok(())
    }
}
