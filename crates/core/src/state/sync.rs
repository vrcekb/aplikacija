//! State Synchronization - Ultra-Performance State Sync
//!
//! Production-ready state synchronization primitives for `TallyIO`.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use super::{StateError, StateResult, StateVersion};

/// Synchronization barrier for state updates
pub struct SyncBarrier {
    /// Current version
    version: AtomicU64,

    /// Waiting threads count
    waiting: AtomicU64,

    /// Barrier is open
    is_open: AtomicBool,
}

impl SyncBarrier {
    /// Create new sync barrier
    #[must_use]
    pub const fn new() -> Self {
        Self {
            version: AtomicU64::new(0),
            waiting: AtomicU64::new(0),
            is_open: AtomicBool::new(true),
        }
    }

    /// Wait for version update
    ///
    /// # Errors
    ///
    /// Returns error if timeout is reached before target version is available
    pub fn wait_for_version(
        &self,
        target_version: StateVersion,
        timeout: Duration,
    ) -> StateResult<()> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            let current = self.version.load(Ordering::Acquire);
            if current >= target_version.raw() {
                return Ok(());
            }

            std::thread::yield_now();
        }

        Err(StateError::SyncTimeout {
            duration_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
        })
    }

    /// Update version and notify waiters
    pub fn update_version(&self, new_version: StateVersion) {
        self.version.store(new_version.raw(), Ordering::Release);
    }

    /// Get current version
    #[must_use]
    pub fn current_version(&self) -> StateVersion {
        StateVersion::new(self.version.load(Ordering::Acquire))
    }

    /// Get waiting threads count
    #[must_use]
    pub fn waiting_count(&self) -> u64 {
        self.waiting.load(Ordering::Relaxed)
    }

    /// Check if barrier is open
    #[must_use]
    pub fn is_barrier_open(&self) -> bool {
        self.is_open.load(Ordering::Relaxed)
    }
}

impl Default for SyncBarrier {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free state synchronizer
pub struct StateSynchronizer {
    /// Global sync barrier
    barrier: Arc<SyncBarrier>,

    /// Synchronization statistics
    stats: Arc<SyncStats>,
}

/// Synchronization statistics
#[derive(Debug, Default)]
pub struct SyncStats {
    /// Total sync operations
    pub total_syncs: AtomicU64,

    /// Successful syncs
    pub successful_syncs: AtomicU64,

    /// Failed syncs
    pub failed_syncs: AtomicU64,

    /// Total wait time in nanoseconds
    pub total_wait_time_ns: AtomicU64,
}

impl SyncStats {
    /// Get success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_syncs.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0_f64;
        }

        let successful = self.successful_syncs.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        {
            successful as f64 / total as f64
        }
    }

    /// Get average wait time in microseconds
    #[must_use]
    pub fn average_wait_time_us(&self) -> f64 {
        let total = self.total_syncs.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0_f64;
        }

        let total_wait_ns = self.total_wait_time_ns.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        {
            (total_wait_ns as f64 / total as f64) / 1000.0_f64
        }
    }
}

impl StateSynchronizer {
    /// Create new state synchronizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            barrier: Arc::new(SyncBarrier::new()),
            stats: Arc::new(SyncStats::default()),
        }
    }

    /// Synchronize to specific version
    ///
    /// # Errors
    ///
    /// Returns error if synchronization fails or timeout is reached
    pub fn sync_to_version(
        &self,
        target_version: StateVersion,
        timeout: Duration,
    ) -> StateResult<()> {
        let start = Instant::now();
        self.stats.total_syncs.fetch_add(1, Ordering::Relaxed);

        let result = self.barrier.wait_for_version(target_version, timeout);

        let wait_time = start.elapsed();
        self.stats.total_wait_time_ns.fetch_add(
            u64::try_from(wait_time.as_nanos()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );

        match result {
            Ok(()) => {
                self.stats.successful_syncs.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.stats.failed_syncs.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Update global version
    pub fn update_version(&self, new_version: StateVersion) {
        self.barrier.update_version(new_version);
    }

    /// Get current version
    #[must_use]
    pub fn current_version(&self) -> StateVersion {
        self.barrier.current_version()
    }

    /// Get synchronization statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<SyncStats> {
        &self.stats
    }
}

impl Default for StateSynchronizer {
    fn default() -> Self {
        Self::new()
    }
}
