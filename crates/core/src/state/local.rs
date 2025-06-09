//! Thread-Local State Management - Ultra-Fast Thread-Local Caching
//!
//! Production-ready thread-local state caching for <1ms read operations.
//! Implements write-through caching with automatic invalidation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use crate::types::{Address, TxHash, WorkerId};

use super::{
    GlobalState, MarketState, MevOpportunity, PositionState, StateEntry, StateResult, StateVersion,
};

thread_local! {
    /// Thread-local state cache
    static LOCAL_CACHE: RefCell<LocalStateCache> = RefCell::new(LocalStateCache::new());
}

/// Thread-local state cache implementation
struct LocalStateCache {
    /// Cached market states
    markets: HashMap<(Address, Address), StateEntry<MarketState>>,

    /// Cached position states
    positions: HashMap<(Address, Address), StateEntry<PositionState>>,

    /// Cached MEV opportunities
    mev_opportunities: HashMap<TxHash, StateEntry<MevOpportunity>>,

    /// Last global version seen
    last_global_version: StateVersion,

    /// Cache statistics
    stats: LocalCacheStats,

    /// Cache creation time
    created_at: Instant,

    /// Maximum cache age
    max_age: Duration,
}

/// Local cache statistics
#[derive(Debug, Default)]
struct LocalCacheStats {
    /// Cache hits
    hits: u64,

    /// Cache misses
    misses: u64,

    /// Cache invalidations
    invalidations: u64,

    /// Last access time
    last_access: Option<Instant>,
}

impl LocalStateCache {
    /// Create new local cache
    fn new() -> Self {
        Self {
            markets: HashMap::new(),
            positions: HashMap::new(),
            mev_opportunities: HashMap::new(),
            last_global_version: StateVersion::default(),
            stats: LocalCacheStats::default(),
            created_at: Instant::now(),
            max_age: Duration::from_secs(60), // 1 minute max age
        }
    }

    /// Check if cache needs refresh
    fn needs_refresh(&self, global_version: StateVersion) -> bool {
        global_version.is_newer_than(self.last_global_version)
            || self.created_at.elapsed() > self.max_age
    }

    /// Invalidate cache
    fn invalidate(&mut self) {
        self.markets.clear();
        self.positions.clear();
        self.mev_opportunities.clear();
        self.stats.invalidations += 1;
    }

    /// Update global version
    const fn update_version(&mut self, version: StateVersion) {
        self.last_global_version = version;
    }

    /// Record cache hit
    fn record_hit(&mut self) {
        self.stats.hits += 1;
        self.stats.last_access = Some(Instant::now());
    }

    /// Record cache miss
    fn record_miss(&mut self) {
        self.stats.misses += 1;
        self.stats.last_access = Some(Instant::now());
    }

    /// Get cache hit ratio
    fn hit_ratio(&self) -> f64 {
        let total = self.stats.hits + self.stats.misses;
        if total == 0 {
            0.0_f64
        } else {
            // Safe conversion with precision awareness for performance metrics
            #[allow(clippy::cast_precision_loss)]
            {
                self.stats.hits as f64 / total as f64
            }
        }
    }
}

/// Thread-local state manager
pub struct LocalState {
    /// Reference to global state
    global_state: Arc<GlobalState>,

    /// Worker ID for this thread
    worker_id: Option<WorkerId>,

    /// Local statistics
    stats: Arc<LocalStateStats>,
}

/// Local state statistics (shared across threads)
#[derive(Debug, Default)]
pub struct LocalStateStats {
    /// Total cache operations
    pub total_operations: AtomicU64,

    /// Total cache hits across all threads
    pub total_hits: AtomicU64,

    /// Total cache misses across all threads
    pub total_misses: AtomicU64,

    /// Total cache invalidations
    pub total_invalidations: AtomicU64,
}

impl LocalStateStats {
    /// Get overall hit ratio
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let hits = self.total_hits.load(Ordering::Relaxed);
        let misses = self.total_misses.load(Ordering::Relaxed);
        let total = hits + misses;

        if total == 0 {
            0.0_f64
        } else {
            // Safe conversion with precision awareness for performance metrics
            #[allow(clippy::cast_precision_loss)]
            {
                hits as f64 / total as f64
            }
        }
    }
}

impl LocalState {
    /// Create new local state manager
    #[must_use]
    pub fn new(global_state: Arc<GlobalState>) -> Self {
        Self {
            global_state,
            worker_id: None,
            stats: Arc::new(LocalStateStats::default()),
        }
    }

    /// Create local state with worker ID
    #[must_use]
    pub fn with_worker_id(global_state: Arc<GlobalState>, worker_id: WorkerId) -> Self {
        Self {
            global_state,
            worker_id: Some(worker_id),
            stats: Arc::new(LocalStateStats::default()),
        }
    }

    /// Get market state (with caching)
    ///
    /// # Errors
    ///
    /// Returns error if market is not found or cache access fails
    pub fn get_market(&self, token_a: Address, token_b: Address) -> StateResult<MarketState> {
        self.stats.total_operations.fetch_add(1, Ordering::Relaxed);

        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            // Check if cache needs refresh
            let global_version = self.global_state.version();
            if cache.needs_refresh(global_version) {
                cache.invalidate();
                cache.update_version(global_version);
                self.stats
                    .total_invalidations
                    .fetch_add(1, Ordering::Relaxed);
            }

            let key = (token_a, token_b);

            // Try cache first
            let cache_result = cache.markets.get(&key).and_then(|entry| {
                if entry.is_expired() {
                    None
                } else {
                    Some(entry.value.clone())
                }
            });

            if let Some(market_state) = cache_result {
                cache.record_hit();
                self.stats.total_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(market_state);
            }
            cache.markets.remove(&key);

            // Cache miss - fetch from global state
            cache.record_miss();
            self.stats.total_misses.fetch_add(1, Ordering::Relaxed);

            match self.global_state.get_market(token_a, token_b) {
                Ok(market) => {
                    // Cache the result
                    let entry = StateEntry::new(market.clone());
                    cache.markets.insert(key, entry);
                    Ok(market)
                }
                Err(e) => Err(e),
            }
        })
    }

    /// Get position state (with caching)
    ///
    /// # Errors
    ///
    /// Returns error if position cannot be retrieved from global state
    pub fn get_position(&self, owner: Address, protocol: Address) -> StateResult<PositionState> {
        self.stats.total_operations.fetch_add(1, Ordering::Relaxed);

        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            // Check if cache needs refresh
            let global_version = self.global_state.version();
            if cache.needs_refresh(global_version) {
                cache.invalidate();
                cache.update_version(global_version);
                self.stats
                    .total_invalidations
                    .fetch_add(1, Ordering::Relaxed);
            }

            let key = (owner, protocol);

            // Try cache first
            let cache_result = cache.positions.get(&key).and_then(|entry| {
                if entry.is_expired() {
                    None
                } else {
                    Some(entry.value.clone())
                }
            });

            if let Some(position_state) = cache_result {
                cache.record_hit();
                self.stats.total_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(position_state);
            }
            cache.positions.remove(&key);

            // Cache miss - fetch from global state
            cache.record_miss();
            self.stats.total_misses.fetch_add(1, Ordering::Relaxed);

            match self.global_state.get_position(owner, protocol) {
                Ok(position) => {
                    // Cache the result
                    let entry = StateEntry::new(position.clone());
                    cache.positions.insert(key, entry);
                    Ok(position)
                }
                Err(e) => Err(e),
            }
        })
    }

    /// Get MEV opportunity (with caching)
    ///
    /// # Errors
    ///
    /// Returns error if MEV opportunity cannot be retrieved from global state
    pub fn get_mev_opportunity(&self, tx_hash: TxHash) -> StateResult<MevOpportunity> {
        self.stats.total_operations.fetch_add(1, Ordering::Relaxed);

        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();

            // Check if cache needs refresh
            let global_version = self.global_state.version();
            if cache.needs_refresh(global_version) {
                cache.invalidate();
                cache.update_version(global_version);
                self.stats
                    .total_invalidations
                    .fetch_add(1, Ordering::Relaxed);
            }

            // Try cache first
            let cache_result = cache.mev_opportunities.get(&tx_hash).and_then(|entry| {
                if !entry.is_expired() && entry.value.is_valid() {
                    Some(entry.value.clone())
                } else {
                    None
                }
            });

            if let Some(mev_opportunity) = cache_result {
                cache.record_hit();
                self.stats.total_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(mev_opportunity);
            }
            cache.mev_opportunities.remove(&tx_hash);

            // Cache miss - fetch from global state
            cache.record_miss();
            self.stats.total_misses.fetch_add(1, Ordering::Relaxed);

            match self.global_state.get_mev_opportunity(tx_hash) {
                Ok(opportunity) => {
                    // Cache the result
                    let entry = StateEntry::new(opportunity.clone());
                    cache.mev_opportunities.insert(tx_hash, entry);
                    Ok(opportunity)
                }
                Err(e) => Err(e),
            }
        })
    }

    /// Update market state (write-through)
    ///
    /// # Errors
    ///
    /// Returns error if market state cannot be updated in global state
    pub fn update_market(&self, market: MarketState) -> StateResult<()> {
        // Update global state first
        self.global_state.update_market(market.clone())?;

        // Update local cache
        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = (market.token_a, market.token_b);
            let entry = StateEntry::new(market);
            cache.markets.insert(key, entry);
        });

        Ok(())
    }

    /// Update position state (write-through)
    ///
    /// # Errors
    ///
    /// Returns error if position state cannot be updated in global state
    pub fn update_position(&self, position: PositionState) -> StateResult<()> {
        // Update global state first
        self.global_state.update_position(position.clone())?;

        // Update local cache
        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let key = (position.owner, position.protocol);
            let entry = StateEntry::new(position);
            cache.positions.insert(key, entry);
        });

        Ok(())
    }

    /// Add MEV opportunity (write-through)
    ///
    /// # Errors
    ///
    /// Returns error if MEV opportunity cannot be added to global state
    pub fn add_mev_opportunity(&self, opportunity: MevOpportunity) -> StateResult<()> {
        // Update global state first
        self.global_state.add_mev_opportunity(opportunity.clone())?;

        // Update local cache
        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let tx_hash = opportunity.target_tx;
            let entry = StateEntry::new(opportunity);
            cache.mev_opportunities.insert(tx_hash, entry);
        });

        Ok(())
    }

    /// Invalidate local cache
    pub fn invalidate_cache(&self) {
        LOCAL_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            cache.invalidate();
            self.stats
                .total_invalidations
                .fetch_add(1, Ordering::Relaxed);
        });
    }

    /// Get cache statistics
    #[must_use]
    pub fn cache_stats(&self) -> (f64, u64) {
        LOCAL_CACHE.with(|cache| {
            let cache = cache.borrow();
            (cache.hit_ratio(), cache.stats.invalidations)
        })
    }

    /// Get worker ID
    #[must_use]
    pub const fn worker_id(&self) -> Option<WorkerId> {
        self.worker_id
    }

    /// Get global state reference
    #[must_use]
    pub const fn global_state(&self) -> &Arc<GlobalState> {
        &self.global_state
    }

    /// Get local statistics
    #[must_use]
    pub const fn stats(&self) -> &Arc<LocalStateStats> {
        &self.stats
    }
}
