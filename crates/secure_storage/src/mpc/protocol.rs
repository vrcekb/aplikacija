//! # MPC Protocol Implementation
//!
//! Core protocol implementation for multi-party computation operations
//! including message handling, state management, and network coordination.

use super::{PartyId, ProtocolState, ThresholdConfig};
use crate::error::{SecureStorageError, SecureStorageResult};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};

/// MPC protocol message types
#[derive(Debug, Clone)]
pub enum MpcMessage {
    /// DKG commitment broadcast
    DkgCommitment {
        /// Operation identifier
        operation_id: String,
        /// Sender party
        sender: PartyId,
        /// Commitment values
        commitments: Vec<Vec<u8>>,
        /// Proof of knowledge
        proof: Vec<u8>,
    },
    /// DKG share distribution
    DkgShare {
        /// Operation identifier
        operation_id: String,
        /// Sender party
        sender: PartyId,
        /// Receiver party
        receiver: PartyId,
        /// Encrypted share data
        encrypted_share: Vec<u8>,
        /// Verification data
        verification_data: Vec<u8>,
    },
    /// Threshold signature request
    SignatureRequest {
        /// Operation identifier
        operation_id: String,
        /// Requesting party
        requester: PartyId,
        /// Message to sign
        message: Vec<u8>,
        /// Participating parties
        participants: Vec<PartyId>,
    },
    /// Partial signature response
    PartialSignature {
        /// Operation identifier
        operation_id: String,
        /// Sender party
        sender: PartyId,
        /// Signature data
        signature: Vec<u8>,
        /// Proof of correctness
        proof: Vec<u8>,
    },
    /// Protocol abort message
    Abort {
        /// Operation identifier
        operation_id: String,
        /// Sender party
        sender: PartyId,
        /// Abort reason
        reason: String,
    },
    /// Heartbeat message
    Heartbeat {
        /// Sender party
        sender: PartyId,
        /// Timestamp
        timestamp: Instant,
    },
}

impl MpcMessage {
    /// Get the operation ID if applicable
    #[must_use]
    pub fn operation_id(&self) -> Option<&str> {
        match self {
            Self::DkgCommitment { operation_id, .. }
            | Self::DkgShare { operation_id, .. }
            | Self::SignatureRequest { operation_id, .. }
            | Self::PartialSignature { operation_id, .. }
            | Self::Abort { operation_id, .. } => Some(operation_id),
            Self::Heartbeat { .. } => None,
        }
    }

    /// Get the sender party ID
    #[must_use]
    pub const fn sender(&self) -> PartyId {
        match self {
            Self::DkgCommitment { sender, .. }
            | Self::DkgShare { sender, .. }
            | Self::SignatureRequest {
                requester: sender, ..
            }
            | Self::PartialSignature { sender, .. }
            | Self::Abort { sender, .. }
            | Self::Heartbeat { sender, .. } => *sender,
        }
    }

    /// Validate message integrity
    ///
    /// # Errors
    ///
    /// Returns error if message validation fails
    pub fn validate(&self) -> SecureStorageResult<()> {
        match self {
            Self::DkgCommitment {
                commitments, proof, ..
            } => {
                if commitments.is_empty() || proof.is_empty() {
                    return Err(SecureStorageError::InvalidInput {
                        field: "dkg_commitment".to_string(),
                        reason: "Empty commitments or proof".to_string(),
                    });
                }
            }
            Self::DkgShare {
                encrypted_share,
                verification_data,
                ..
            } => {
                if encrypted_share.is_empty() || verification_data.is_empty() {
                    return Err(SecureStorageError::InvalidInput {
                        field: "dkg_share".to_string(),
                        reason: "Empty share or verification data".to_string(),
                    });
                }
            }
            Self::SignatureRequest {
                message,
                participants,
                ..
            } => {
                if message.is_empty() || participants.is_empty() {
                    return Err(SecureStorageError::InvalidInput {
                        field: "signature_request".to_string(),
                        reason: "Empty message or participants".to_string(),
                    });
                }
            }
            Self::PartialSignature {
                signature, proof, ..
            } => {
                if signature.is_empty() || proof.is_empty() {
                    return Err(SecureStorageError::InvalidInput {
                        field: "partial_signature".to_string(),
                        reason: "Empty signature or proof".to_string(),
                    });
                }
            }
            Self::Abort { reason, .. } => {
                if reason.is_empty() {
                    return Err(SecureStorageError::InvalidInput {
                        field: "abort".to_string(),
                        reason: "Empty abort reason".to_string(),
                    });
                }
            }
            Self::Heartbeat { .. } => {
                // Heartbeat messages are always valid
            }
        }
        Ok(())
    }
}

/// Protocol operation context
#[derive(Debug)]
pub struct ProtocolOperation {
    /// Operation ID
    pub operation_id: String,
    /// Current state
    pub state: ProtocolState,
    /// Participating parties
    pub participants: Vec<PartyId>,
    /// Required threshold
    pub threshold: u32,
    /// Collected messages
    pub messages: HashMap<PartyId, Vec<MpcMessage>>,
    /// Operation start time
    pub start_time: Instant,
    /// Operation timeout
    pub timeout: Duration,
    /// Last activity timestamp
    pub last_activity: Instant,
}

impl ProtocolOperation {
    /// Create a new protocol operation
    #[must_use]
    pub fn new(
        operation_id: String,
        participants: Vec<PartyId>,
        threshold: u32,
        timeout: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            operation_id,
            state: ProtocolState::Initializing,
            participants,
            threshold,
            messages: HashMap::new(),
            start_time: now,
            timeout,
            last_activity: now,
        }
    }

    /// Add a message to the operation
    ///
    /// # Errors
    ///
    /// Returns error if message validation fails
    pub fn add_message(&mut self, message: MpcMessage) -> SecureStorageResult<()> {
        message.validate()?;

        let sender = message.sender();
        self.messages.entry(sender).or_default().push(message);
        self.last_activity = Instant::now();

        Ok(())
    }

    /// Check if operation has timed out
    #[must_use]
    pub fn is_timed_out(&self) -> bool {
        self.start_time.elapsed() > self.timeout
    }

    /// Check if we have enough messages for the current phase
    #[must_use]
    pub fn has_sufficient_messages(&self, message_type: &str) -> bool {
        let count = self
            .messages
            .values()
            .flatten()
            .filter(|msg| Self::message_matches_type(msg, message_type))
            .count();

        count >= self.threshold as usize
    }

    /// Check if a message matches the given type
    fn message_matches_type(message: &MpcMessage, message_type: &str) -> bool {
        matches!(
            (message, message_type),
            (MpcMessage::DkgCommitment { .. }, "dkg_commitment")
                | (MpcMessage::DkgShare { .. }, "dkg_share")
                | (MpcMessage::PartialSignature { .. }, "partial_signature")
        )
    }

    /// Get messages of a specific type
    #[must_use]
    pub fn get_messages_of_type(&self, message_type: &str) -> Vec<&MpcMessage> {
        self.messages
            .values()
            .flatten()
            .filter(|msg| Self::message_matches_type(msg, message_type))
            .collect()
    }
}

/// MPC protocol coordinator
#[derive(Debug)]
pub struct MpcProtocol {
    /// Our party ID
    party_id: PartyId,
    /// Protocol configuration
    config: ThresholdConfig,
    /// Active operations
    operations: RwLock<HashMap<String, ProtocolOperation>>,
    /// Message sender channel
    message_sender: mpsc::UnboundedSender<MpcMessage>,
    /// Message receiver channel
    message_receiver: RwLock<Option<mpsc::UnboundedReceiver<MpcMessage>>>,
    /// Performance counters
    messages_sent: AtomicU64,
    messages_received: AtomicU64,
    operations_completed: AtomicU64,
    operations_failed: AtomicU64,
}

impl MpcProtocol {
    /// Create a new MPC protocol coordinator
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid
    pub fn new(party_id: PartyId, config: ThresholdConfig) -> SecureStorageResult<Self> {
        config.validate()?;

        let (message_sender, message_receiver) = mpsc::unbounded_channel();

        info!(
            "Initializing MPC protocol for party {} with {}/{} threshold",
            party_id.inner(),
            config.threshold,
            config.total_parties
        );

        Ok(Self {
            party_id,
            config,
            operations: RwLock::new(HashMap::new()),
            message_sender,
            message_receiver: RwLock::new(Some(message_receiver)),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            operations_completed: AtomicU64::new(0),
            operations_failed: AtomicU64::new(0),
        })
    }

    /// Start a new protocol operation
    ///
    /// # Errors
    ///
    /// Returns error if operation already exists or configuration is invalid
    pub async fn start_operation(
        &self,
        operation_id: String,
        participants: Vec<PartyId>,
    ) -> SecureStorageResult<()> {
        if participants.len() < self.config.threshold as usize {
            return Err(SecureStorageError::InvalidInput {
                field: "participants".to_string(),
                reason: format!(
                    "Insufficient participants: {} < {}",
                    participants.len(),
                    self.config.threshold
                ),
            });
        }

        let operation = ProtocolOperation::new(
            operation_id.clone(),
            participants,
            self.config.threshold,
            self.config.timeout,
        );

        let mut operations = self.operations.write().await;
        if operations.contains_key(&operation_id) {
            return Err(SecureStorageError::InvalidInput {
                field: "operation_id".to_string(),
                reason: "Operation already exists".to_string(),
            });
        }

        operations.insert(operation_id.clone(), operation);
        drop(operations);

        info!("Started protocol operation: {}", operation_id);
        Ok(())
    }

    /// Send a message to other parties
    ///
    /// # Errors
    ///
    /// Returns error if message is invalid or sending fails
    pub fn send_message(&self, message: MpcMessage) -> SecureStorageResult<()> {
        message.validate()?;

        // In a real implementation, this would send the message over the network
        // For now, we just log it and increment the counter
        debug!(
            "Sending message from party {} for operation {:?}",
            self.party_id.inner(),
            message.operation_id()
        );

        self.message_sender
            .send(message)
            .map_err(|e| SecureStorageError::InvalidInput {
                field: "message".to_string(),
                reason: format!("Failed to send message: {e}"),
            })?;

        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Process a received message
    ///
    /// # Errors
    ///
    /// Returns error if message is invalid or processing fails
    pub async fn process_message(&self, message: MpcMessage) -> SecureStorageResult<()> {
        message.validate()?;

        debug!(
            "Processing message from party {} for operation {:?}",
            message.sender().inner(),
            message.operation_id()
        );

        if let Some(operation_id) = message.operation_id() {
            let mut operations = self.operations.write().await;
            if let Some(operation) = operations.get_mut(operation_id) {
                operation.add_message(message)?;

                // Check if we can advance the protocol state
                self.check_state_transition(operation);
            } else {
                warn!("Received message for unknown operation: {}", operation_id);
            }
        }

        self.messages_received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Check if an operation can transition to the next state
    fn check_state_transition(&self, operation: &mut ProtocolOperation) {
        match operation.state {
            ProtocolState::Initializing => {
                Self::handle_initializing_state(operation);
            }
            ProtocolState::WaitingForParties => {
                Self::handle_waiting_state(operation);
            }
            ProtocolState::Computing => {
                self.handle_computing_state(operation);
            }
            ProtocolState::Completed | ProtocolState::Failed(_) | ProtocolState::TimedOut => {
                // Terminal states - no transitions
            }
        }
    }

    /// Handle initializing state transition
    fn handle_initializing_state(operation: &mut ProtocolOperation) {
        if operation.has_sufficient_messages("dkg_commitment") {
            operation.state = ProtocolState::WaitingForParties;
            info!(
                "Operation {} transitioned to WaitingForParties",
                operation.operation_id
            );
        }
    }

    /// Handle waiting for parties state transition
    fn handle_waiting_state(operation: &mut ProtocolOperation) {
        if operation.has_sufficient_messages("dkg_share") {
            operation.state = ProtocolState::Computing;
            info!(
                "Operation {} transitioned to Computing",
                operation.operation_id
            );
        }
    }

    /// Handle computing state transition
    fn handle_computing_state(&self, operation: &mut ProtocolOperation) {
        if operation.has_sufficient_messages("partial_signature") {
            operation.state = ProtocolState::Completed;
            self.operations_completed.fetch_add(1, Ordering::Relaxed);
            info!(
                "Operation {} completed successfully",
                operation.operation_id
            );
        }
    }

    /// Clean up timed out operations
    ///
    /// # Errors
    ///
    /// Returns error if cleanup fails
    pub async fn cleanup_timed_out_operations(&self) -> SecureStorageResult<()> {
        let mut operations = self.operations.write().await;
        let mut to_remove = Vec::with_capacity(operations.len());

        for (operation_id, operation) in operations.iter_mut() {
            if operation.is_timed_out() && operation.state != ProtocolState::TimedOut {
                operation.state = ProtocolState::TimedOut;
                to_remove.push(operation_id.clone());
                self.operations_failed.fetch_add(1, Ordering::Relaxed);
                warn!("Operation {} timed out", operation_id);
            }
        }

        for operation_id in to_remove {
            operations.remove(&operation_id);
        }
        drop(operations);

        Ok(())
    }

    /// Get operation status
    pub async fn get_operation_status(&self, operation_id: &str) -> Option<ProtocolState> {
        let operations = self.operations.read().await;
        operations.get(operation_id).map(|op| op.state.clone())
    }

    /// Start the message processing loop
    ///
    /// # Errors
    ///
    /// Returns error if message receiver is not available
    pub async fn start_message_loop(&self) -> SecureStorageResult<()> {
        let mut receiver = self.take_message_receiver().await?;
        self.log_message_loop_start();
        Self::start_cleanup_task();
        self.process_message_loop(&mut receiver).await?;
        Ok(())
    }

    /// Take the message receiver
    async fn take_message_receiver(
        &self,
    ) -> SecureStorageResult<tokio::sync::mpsc::UnboundedReceiver<MpcMessage>> {
        let mut receiver_guard = self.message_receiver.write().await;
        receiver_guard
            .take()
            .ok_or_else(|| SecureStorageError::InvalidInput {
                field: "message_receiver".to_string(),
                reason: "Message receiver already taken".to_string(),
            })
    }

    /// Log message loop start
    fn log_message_loop_start(&self) {
        info!(
            "Starting MPC protocol message loop for party {}",
            self.party_id.inner()
        );
    }

    /// Start cleanup task
    fn start_cleanup_task() {
        // Start cleanup task - removed unsafe code for now
        // TODO: Implement proper cleanup task with Arc<Self>
        info!("Message loop started - cleanup task disabled for safety");
    }

    /// Process the message loop
    async fn process_message_loop(
        &self,
        receiver: &mut tokio::sync::mpsc::UnboundedReceiver<MpcMessage>,
    ) -> SecureStorageResult<()> {
        while let Some(message) = receiver.recv().await {
            if let Err(e) = self.process_message(message).await {
                error!("Failed to process message: {}", e);
            }
        }
        Ok(())
    }

    /// Get protocol statistics
    #[must_use]
    pub async fn get_stats(&self) -> ProtocolStats {
        let operations_count = self.operations.read().await.len();

        ProtocolStats {
            party_id: self.party_id,
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            operations_completed: self.operations_completed.load(Ordering::Relaxed),
            operations_failed: self.operations_failed.load(Ordering::Relaxed),
            active_operations: operations_count,
        }
    }
}

/// Protocol statistics
#[derive(Debug, Clone)]
pub struct ProtocolStats {
    /// Our party ID
    pub party_id: PartyId,
    /// Number of messages sent
    pub messages_sent: u64,
    /// Number of messages received
    pub messages_received: u64,
    /// Number of operations completed
    pub operations_completed: u64,
    /// Number of operations failed
    pub operations_failed: u64,
    /// Number of active operations
    pub active_operations: usize,
}
