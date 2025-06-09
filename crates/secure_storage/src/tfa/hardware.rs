//! # Hardware Token Authentication
//!
//! Support for hardware security keys including `YubiKey`, FIDO2, and `WebAuthn`.

use super::TfaConfig;
use crate::error::SecureStorageResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Hardware token system
#[derive(Debug)]
pub struct HardwareTokenSystem {
    /// Registered tokens
    tokens: RwLock<HashMap<String, Vec<HardwareToken>>>,
    /// Performance counters
    authentications: AtomicU64,
    authentication_failures: AtomicU64,
}

/// Hardware token information
#[derive(Debug, Clone)]
pub struct HardwareToken {
    /// Token ID
    pub token_id: String,
    /// Token type
    pub token_type: TokenType,
    /// Public key
    pub public_key: Vec<u8>,
    /// Registration timestamp
    pub registered_at: SystemTime,
    /// Last used timestamp
    pub last_used: Option<SystemTime>,
}

/// Hardware token types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    /// `YubiKey`
    YubiKey,
    /// FIDO2/WebAuthn
    Fido2,
    /// Generic PKCS#11 token
    Pkcs11,
}

impl HardwareTokenSystem {
    /// Create new hardware token system
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(_config: TfaConfig) -> SecureStorageResult<Self> {
        info!("Initialized hardware token system");

        Ok(Self {
            tokens: RwLock::new(HashMap::new()),
            authentications: AtomicU64::new(0),
            authentication_failures: AtomicU64::new(0),
        })
    }

    /// Verify hardware token
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub async fn verify_token(
        &self,
        user_id: &str,
        _token_response: &str,
        _challenge_nonce: &[u8],
    ) -> SecureStorageResult<bool> {
        let tokens = self.tokens.read().await;

        if let Some(user_tokens) = tokens.get(user_id) {
            if !user_tokens.is_empty() {
                drop(tokens);
                // Simulate token verification
                self.authentications.fetch_add(1, Ordering::Relaxed);
                debug!("Hardware token verified for user: {}", user_id);
                return Ok(true);
            }
        }
        drop(tokens);

        self.authentication_failures.fetch_add(1, Ordering::Relaxed);
        debug!("Hardware token verification failed for user: {}", user_id);
        Ok(false)
    }

    /// Get statistics
    #[must_use]
    pub async fn get_stats(&self) -> HardwareTokenStats {
        let tokens_count = self.tokens.read().await.values().map(Vec::len).sum();

        HardwareTokenStats {
            registered_tokens: tokens_count,
            authentications: self.authentications.load(Ordering::Relaxed),
            authentication_failures: self.authentication_failures.load(Ordering::Relaxed),
        }
    }
}

/// Hardware token statistics
#[derive(Debug, Clone)]
pub struct HardwareTokenStats {
    /// Number of registered tokens
    pub registered_tokens: usize,
    /// Total authentications
    pub authentications: u64,
    /// Total authentication failures
    pub authentication_failures: u64,
}
