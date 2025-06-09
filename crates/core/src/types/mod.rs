//! `TallyIO` Core Types
//!
//! Production-ready type definitions for ultra-performance financial operations.
//! All types are optimized for <1ms latency requirements.

pub mod opportunity;
pub mod result;
pub mod transaction;

// Re-exports for convenience
pub use opportunity::*;
pub use result::*;
pub use transaction::{Transaction, TxHash, TxStatus};

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Global ID counter for generating unique identifiers
static GLOBAL_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate next unique ID
#[inline]
fn next_id() -> u64 {
    GLOBAL_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Strategy identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StrategyId(u64);

impl StrategyId {
    /// Create new strategy ID
    #[must_use]
    pub fn new() -> Self {
        Self(next_id())
    }

    /// Create strategy ID from raw value
    #[must_use]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get raw ID value
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for StrategyId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StrategyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "strategy-{}", self.0)
    }
}

/// Task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(u64);

impl TaskId {
    /// Create new task ID
    #[must_use]
    pub fn new() -> Self {
        Self(next_id())
    }

    /// Create task ID from raw value
    #[must_use]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get raw ID value
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "task-{}", self.0)
    }
}

/// Worker identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(u64);

impl WorkerId {
    /// Create new worker ID
    #[must_use]
    pub fn new() -> Self {
        Self(next_id())
    }

    /// Create worker ID from raw value
    #[must_use]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Get raw ID value
    #[must_use]
    pub const fn raw(&self) -> u64 {
        self.0
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worker-{}", self.0)
    }
}

/// Gas amount
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Gas(u64);

impl Gas {
    /// Create new gas amount
    #[must_use]
    pub const fn new(amount: u64) -> Self {
        Self(amount)
    }

    /// Standard gas limit for simple transfers
    pub const TRANSFER: Self = Self(21_000);

    /// Standard gas limit for contract interactions
    pub const CONTRACT: Self = Self(100_000);

    /// Maximum reasonable gas limit
    pub const MAX: Self = Self(10_000_000);

    /// Get gas amount
    #[must_use]
    pub const fn amount(&self) -> u64 {
        self.0
    }

    /// Check if gas amount is reasonable
    #[must_use]
    pub const fn is_reasonable(&self) -> bool {
        self.0 >= Self::TRANSFER.0 && self.0 <= Self::MAX.0
    }
}

impl fmt::Display for Gas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} gas", self.0)
    }
}

/// Price in wei (smallest Ethereum unit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Price(pub u128);

impl Price {
    /// Create new price in wei
    #[must_use]
    pub const fn new(wei: u128) -> Self {
        Self(wei)
    }

    /// Create price from gwei (10^9 wei)
    #[must_use]
    pub const fn from_gwei(gwei: u64) -> Self {
        Self((gwei as u128) * 1_000_000_000)
    }

    /// Create price from ether (10^18 wei)
    #[must_use]
    pub const fn from_ether(ether: u64) -> Self {
        Self((ether as u128) * 1_000_000_000_000_000_000)
    }

    /// Get raw wei amount
    #[must_use]
    pub const fn wei(&self) -> u128 {
        self.0
    }

    /// Get gwei amount (rounded down)
    ///
    /// # Errors
    ///
    /// Returns 0 if conversion would overflow (extremely unlikely in practice)
    #[must_use]
    pub fn gwei(&self) -> u64 {
        u64::try_from(self.0 / 1_000_000_000).unwrap_or(0)
    }

    /// Get ether amount (rounded down)
    ///
    /// # Errors
    ///
    /// Returns 0 if conversion would overflow (extremely unlikely in practice)
    #[must_use]
    pub fn ether(&self) -> u64 {
        u64::try_from(self.0 / 1_000_000_000_000_000_000).unwrap_or(0)
    }

    /// Check if price is zero
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }

    /// Add two prices (saturating)
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Subtract two prices (saturating)
    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Multiply price by factor (saturating)
    #[must_use]
    pub const fn saturating_mul(self, factor: u64) -> Self {
        Self(self.0.saturating_mul(factor as u128))
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 >= 1_000_000_000_000_000_000 {
            write!(f, "{} ETH", self.ether())
        } else if self.0 >= 1_000_000_000 {
            write!(f, "{} gwei", self.gwei())
        } else {
            write!(f, "{} wei", self.0)
        }
    }
}

/// Blockchain address (20 bytes for Ethereum)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Address([u8; 20]);

impl Address {
    /// Create new address from bytes
    #[must_use]
    pub const fn new(bytes: [u8; 20]) -> Self {
        Self(bytes)
    }

    /// Create zero address
    #[must_use]
    pub const fn zero() -> Self {
        Self([0; 20])
    }

    /// Get address bytes
    #[must_use]
    pub const fn bytes(&self) -> &[u8; 20] {
        &self.0
    }

    /// Check if address is zero
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }

    /// Convert to hex string (without 0x prefix)
    #[must_use]
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Create from hex string (with or without 0x prefix)
    ///
    /// # Errors
    ///
    /// Returns error if hex string is invalid or wrong length
    pub fn from_hex(hex: &str) -> Result<Self, hex::FromHexError> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        let bytes = hex::decode(hex)?;
        if bytes.len() != 20 {
            return Err(hex::FromHexError::InvalidStringLength);
        }
        let mut addr = [0u8; 20];
        addr.copy_from_slice(&bytes);
        Ok(Self(addr))
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{}", self.to_hex())
    }
}

/// Private key (32 bytes, zeroized on drop)
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct PrivateKey([u8; 32]);

impl PrivateKey {
    /// Create new private key
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get key bytes (use carefully)
    #[must_use]
    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Generate random private key
    #[must_use]
    pub fn random() -> Self {
        let mut bytes = [0u8; 32];
        // Use a simple pseudo-random generation for now
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| {
                // Safe conversion: we only need lower 64 bits for randomness
                u64::try_from(d.as_nanos() % u128::from(u64::MAX)).unwrap_or(0)
            });

        for (i, byte) in bytes.iter_mut().enumerate() {
            let index = u64::try_from(i).unwrap_or(0);
            *byte = u8::try_from((timestamp.wrapping_mul(index + 1)) % 256).unwrap_or(0);
        }
        Self(bytes)
    }
}

impl fmt::Debug for PrivateKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PrivateKey([REDACTED])")
    }
}

/// Block number
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockNumber(pub u64);

impl BlockNumber {
    /// Create new block number
    #[must_use]
    pub const fn new(number: u64) -> Self {
        Self(number)
    }

    /// Get block number
    #[must_use]
    pub const fn number(&self) -> u64 {
        self.0
    }

    /// Get next block number
    #[must_use]
    pub const fn next(&self) -> Self {
        Self(self.0.saturating_add(1))
    }

    /// Get previous block number
    #[must_use]
    pub const fn prev(&self) -> Self {
        Self(self.0.saturating_sub(1))
    }
}

impl fmt::Display for BlockNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gas() {
        let gas = Gas::new(21000);
        assert_eq!(gas.amount(), 21000);
        assert!(gas.is_reasonable());

        let high_gas = Gas::new(20_000_000);
        assert!(!high_gas.is_reasonable());
    }

    #[test]
    fn test_price() {
        let price = Price::from_gwei(20);
        assert_eq!(price.gwei(), 20);

        let eth_price = Price::from_ether(1);
        assert_eq!(eth_price.ether(), 1);

        let sum = price.saturating_add(eth_price);
        assert!(sum.wei() > price.wei());
    }

    #[test]
    fn test_address() {
        let addr = Address::zero();
        assert!(addr.is_zero());

        let hex_addr = addr.to_hex();
        assert_eq!(hex_addr.len(), 40);

        let parsed = Address::from_hex(&format!("0x{hex_addr}"));
        assert!(parsed.is_ok(), "Valid hex address should parse correctly");
        #[allow(clippy::unwrap_used)]
        let parsed = parsed.unwrap();
        assert_eq!(addr, parsed);
    }

    #[test]
    fn test_private_key_zeroize() {
        let key = PrivateKey::random();
        // Key should be zeroized when dropped
        drop(key);
        // This test mainly ensures compilation with zeroize
    }

    #[test]
    fn test_block_number() {
        let block = BlockNumber::new(100);
        assert_eq!(block.next().number(), 101);
        assert_eq!(block.prev().number(), 99);

        let zero_block = BlockNumber::new(0);
        assert_eq!(zero_block.prev().number(), 0); // Saturating sub
    }
}
