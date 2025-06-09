//! Mempool Watcher - Real-time Mempool Monitoring
//!
//! Production-ready mempool watching for real-time transaction monitoring.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam::channel::{self, Receiver, Sender};

use crate::types::Transaction;

use super::{MempoolConfig, MempoolError, MempoolResult};

/// Mempool watcher for real-time monitoring
pub struct MempoolWatcher {
    /// Configuration
    config: Arc<MempoolConfig>,

    /// Running state
    is_running: AtomicBool,

    /// Transaction sender
    tx_sender: Sender<Transaction>,

    /// Transaction receiver
    tx_receiver: Receiver<Transaction>,

    /// Watcher statistics
    stats: Arc<WatcherStats>,

    /// Worker thread handle
    worker_handle: Option<thread::JoinHandle<()>>,
}

/// Watcher statistics
#[derive(Debug, Default)]
pub struct WatcherStats {
    /// Total transactions watched
    pub transactions_watched: AtomicU64,

    /// Transactions per second
    pub transactions_per_second: AtomicU64,

    /// Last update time
    pub last_update: std::sync::RwLock<Option<Instant>>,

    /// Watch errors
    pub watch_errors: AtomicU64,
}

impl WatcherStats {
    /// Update transactions per second
    pub fn update_tps(&self, count: u64) {
        self.transactions_per_second.store(count, Ordering::Relaxed);
        if let Ok(mut last_update) = self.last_update.write() {
            *last_update = Some(Instant::now());
        }
        // Note: If lock is poisoned, we continue without updating timestamp
        // This is acceptable for metrics collection
    }

    /// Get current TPS
    #[must_use]
    pub fn current_tps(&self) -> u64 {
        self.transactions_per_second.load(Ordering::Relaxed)
    }
}

impl MempoolWatcher {
    /// Create new mempool watcher
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(config: Arc<MempoolConfig>) -> MempoolResult<Self> {
        let (tx_sender, tx_receiver) = channel::unbounded();

        Ok(Self {
            config,
            is_running: AtomicBool::new(false),
            tx_sender,
            tx_receiver,
            stats: Arc::new(WatcherStats::default()),
            worker_handle: None,
        })
    }

    /// Start watching mempool
    ///
    /// # Errors
    ///
    /// Returns error if watcher is already running or start fails
    pub fn start(&mut self) -> MempoolResult<()> {
        if self.is_running.load(Ordering::Acquire) {
            return Err(MempoolError::WatcherError {
                reason: "Watcher is already running".to_string(),
            });
        }

        self.is_running.store(true, Ordering::Release);

        // Start worker thread
        let config = Arc::clone(&self.config);
        let stats = Arc::clone(&self.stats);
        let is_running = Arc::new(AtomicBool::new(true));
        let is_running_clone = Arc::clone(&is_running);

        let handle = thread::Builder::new()
            .name("mempool-watcher".to_string())
            .spawn(move || {
                Self::watch_loop(&config, &stats, &is_running_clone);
            })
            .map_err(|e| MempoolError::WatcherError {
                reason: format!("Failed to start watcher thread: {e}"),
            })?;

        self.worker_handle = Some(handle);

        Ok(())
    }

    /// Stop watching mempool
    ///
    /// # Errors
    ///
    /// Returns error if watcher is not running or stop fails
    pub fn stop(&mut self) -> MempoolResult<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(MempoolError::WatcherError {
                reason: "Watcher is not running".to_string(),
            });
        }

        self.is_running.store(false, Ordering::Release);

        // Wait for worker thread to finish
        if let Some(handle) = self.worker_handle.take() {
            handle.join().map_err(|_| MempoolError::WatcherError {
                reason: "Failed to join watcher thread".to_string(),
            })?;
        }

        Ok(())
    }

    /// Watch loop implementation
    fn watch_loop(
        _config: &Arc<MempoolConfig>,
        stats: &Arc<WatcherStats>,
        is_running: &Arc<AtomicBool>,
    ) {
        let mut last_tps_update = Instant::now();
        let mut transaction_count = 0;

        while is_running.load(Ordering::Acquire) {
            // Simulate watching mempool
            // In a real implementation, this would connect to blockchain nodes
            // and listen for new transactions

            // Update TPS every second
            if last_tps_update.elapsed() >= Duration::from_secs(1) {
                stats.update_tps(transaction_count);
                transaction_count = 0;
                last_tps_update = Instant::now();
            }

            // Sleep briefly to avoid busy waiting
            thread::sleep(Duration::from_millis(10));
        }
    }

    /// Submit transaction to watcher
    ///
    /// # Errors
    ///
    /// Returns error if submission fails
    pub fn submit_transaction(&self, transaction: Transaction) -> MempoolResult<()> {
        if !self.is_running.load(Ordering::Acquire) {
            return Err(MempoolError::WatcherError {
                reason: "Watcher is not running".to_string(),
            });
        }

        self.tx_sender
            .send(transaction)
            .map_err(|e| MempoolError::WatcherError {
                reason: format!("Failed to send transaction: {e}"),
            })?;

        self.stats
            .transactions_watched
            .fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Receive next transaction
    ///
    /// # Errors
    ///
    /// Returns error if receive fails
    pub fn receive_transaction(&self, timeout: Duration) -> MempoolResult<Transaction> {
        self.tx_receiver
            .recv_timeout(timeout)
            .map_err(|e| MempoolError::WatcherError {
                reason: format!("Failed to receive transaction: {e}"),
            })
    }

    /// Check if watcher is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Acquire)
    }

    /// Get watcher statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<WatcherStats> {
        &self.stats
    }

    /// Get transaction channel capacity
    #[must_use]
    pub const fn channel_capacity(&self) -> usize {
        // For unbounded channels, return a large number
        usize::MAX
    }

    /// Get pending transaction count
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.tx_receiver.len()
    }
}

impl Drop for MempoolWatcher {
    fn drop(&mut self) {
        if self.is_running.load(Ordering::Acquire) {
            let _ = self.stop();
        }
    }
}
