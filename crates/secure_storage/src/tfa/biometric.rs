//! # Biometric Authentication
//!
//! Biometric authentication support for fingerprint and facial recognition.

use super::TfaConfig;
use crate::error::SecureStorageResult;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Biometric system
#[derive(Debug)]
pub struct BiometricSystem {
    /// Biometric templates
    templates: RwLock<HashMap<String, Vec<BiometricTemplate>>>,
    /// Performance counters
    authentications: AtomicU64,
    authentication_failures: AtomicU64,
}

/// Biometric template
#[derive(Debug, Clone)]
pub struct BiometricTemplate {
    /// Template ID
    pub template_id: String,
    /// Biometric type
    pub biometric_type: BiometricType,
    /// Template hash
    pub template_hash: Vec<u8>,
    /// Registration timestamp
    pub registered_at: SystemTime,
    /// Last used timestamp
    pub last_used: Option<SystemTime>,
}

/// Biometric types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiometricType {
    /// Fingerprint
    Fingerprint,
    /// Facial recognition
    Face,
    /// Iris scan
    Iris,
    /// Voice recognition
    Voice,
}

impl BiometricSystem {
    /// Create new biometric system
    ///
    /// # Errors
    ///
    /// Returns error if initialization fails
    pub fn new(_config: TfaConfig) -> SecureStorageResult<Self> {
        info!("Initialized biometric system");

        Ok(Self {
            templates: RwLock::new(HashMap::new()),
            authentications: AtomicU64::new(0),
            authentication_failures: AtomicU64::new(0),
        })
    }

    /// Verify biometric
    ///
    /// # Errors
    ///
    /// Returns error if verification fails
    pub async fn verify_biometric(
        &self,
        user_id: &str,
        _biometric_data: &str,
    ) -> SecureStorageResult<bool> {
        let templates = self.templates.read().await;

        if let Some(user_templates) = templates.get(user_id) {
            if !user_templates.is_empty() {
                drop(templates);
                // Simulate biometric verification
                self.authentications.fetch_add(1, Ordering::Relaxed);
                debug!("Biometric verified for user: {}", user_id);
                return Ok(true);
            }
        }
        drop(templates);

        self.authentication_failures.fetch_add(1, Ordering::Relaxed);
        debug!("Biometric verification failed for user: {}", user_id);
        Ok(false)
    }

    /// Get statistics
    #[must_use]
    pub async fn get_stats(&self) -> BiometricStats {
        let templates_count = self.templates.read().await.values().map(Vec::len).sum();

        BiometricStats {
            registered_templates: templates_count,
            authentications: self.authentications.load(Ordering::Relaxed),
            authentication_failures: self.authentication_failures.load(Ordering::Relaxed),
        }
    }
}

/// Biometric statistics
#[derive(Debug, Clone)]
pub struct BiometricStats {
    /// Number of registered templates
    pub registered_templates: usize,
    /// Total authentications
    pub authentications: u64,
    /// Total authentication failures
    pub authentication_failures: u64,
}
