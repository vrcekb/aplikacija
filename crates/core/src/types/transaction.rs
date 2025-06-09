//! Transaction Types
//!
//! Production-ready transaction types for blockchain operations.

use serde::{Deserialize, Serialize};

use super::{Address, BlockNumber, Gas, Price};

/// Transaction hash
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TxHash([u8; 32]);

impl TxHash {
    /// Create new transaction hash
    #[must_use]
    pub const fn new(hash: [u8; 32]) -> Self {
        Self(hash)
    }

    /// Get hash bytes
    #[must_use]
    pub const fn bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Check if hash is zero
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&b| b == 0)
    }
}

impl std::fmt::Display for TxHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x")?;
        for byte in &self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxStatus {
    /// Transaction is pending in mempool
    Pending,
    /// Transaction is included in a block
    Confirmed,
    /// Transaction failed
    Failed,
    /// Transaction was dropped
    Dropped,
}

/// Transaction type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TxType {
    /// Legacy transaction
    Legacy,
    /// EIP-2930 transaction
    AccessList,
    /// EIP-1559 transaction
    DynamicFee,
}

/// Transaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// Transaction hash
    pub hash: TxHash,

    /// Sender address
    pub from: Address,

    /// Recipient address (None for contract creation)
    pub to: Option<Address>,

    /// Transaction value in wei
    pub value: Price,

    /// Gas limit
    pub gas_limit: Gas,

    /// Gas price (for legacy transactions)
    pub gas_price: Option<Price>,

    /// Max fee per gas (for EIP-1559)
    pub max_fee_per_gas: Option<Price>,

    /// Max priority fee per gas (for EIP-1559)
    pub max_priority_fee_per_gas: Option<Price>,

    /// Transaction data
    pub data: Vec<u8>,

    /// Transaction nonce
    pub nonce: u64,

    /// Transaction type
    pub tx_type: TxType,

    /// Block number (if confirmed)
    pub block_number: Option<BlockNumber>,

    /// Transaction index in block
    pub transaction_index: Option<u64>,

    /// Transaction status
    pub status: TxStatus,
}

/// Transaction builder parameters
#[derive(Debug, Clone)]
pub struct TransactionParams {
    /// Transaction hash
    pub hash: TxHash,
    /// Sender address
    pub from: Address,
    /// Recipient address (None for contract creation)
    pub to: Option<Address>,
    /// Value to transfer
    pub value: Price,
    /// Gas limit
    pub gas_limit: Gas,
    /// Transaction data
    pub data: Vec<u8>,
    /// Nonce
    pub nonce: u64,
}

impl Transaction {
    /// Create new transaction
    #[must_use]
    pub fn new(params: TransactionParams, gas_price: Price) -> Self {
        Self {
            hash: params.hash,
            from: params.from,
            to: params.to,
            value: params.value,
            gas_limit: params.gas_limit,
            gas_price: Some(gas_price),
            max_fee_per_gas: None,
            max_priority_fee_per_gas: None,
            data: params.data,
            nonce: params.nonce,
            tx_type: TxType::Legacy,
            block_number: None,
            transaction_index: None,
            status: TxStatus::Pending,
        }
    }

    /// Create EIP-1559 transaction
    #[must_use]
    pub fn new_eip1559(
        params: TransactionParams,
        max_fee_per_gas: Price,
        max_priority_fee_per_gas: Price,
    ) -> Self {
        Self {
            hash: params.hash,
            from: params.from,
            to: params.to,
            value: params.value,
            gas_limit: params.gas_limit,
            gas_price: None,
            max_fee_per_gas: Some(max_fee_per_gas),
            max_priority_fee_per_gas: Some(max_priority_fee_per_gas),
            data: params.data,
            nonce: params.nonce,
            tx_type: TxType::DynamicFee,
            block_number: None,
            transaction_index: None,
            status: TxStatus::Pending,
        }
    }

    /// Check if transaction is contract call
    #[must_use]
    pub const fn is_contract_call(&self) -> bool {
        self.to.is_some() && !self.data.is_empty()
    }

    /// Check if transaction is contract creation
    #[must_use]
    pub const fn is_contract_creation(&self) -> bool {
        self.to.is_none() && !self.data.is_empty()
    }

    /// Check if transaction is simple transfer
    #[must_use]
    pub const fn is_transfer(&self) -> bool {
        self.to.is_some() && self.data.is_empty()
    }

    /// Get effective gas price
    #[must_use]
    pub fn effective_gas_price(&self, base_fee: Option<Price>) -> Price {
        match self.tx_type {
            TxType::Legacy | TxType::AccessList => self.gas_price.unwrap_or(Price::new(0)),
            TxType::DynamicFee => {
                let max_fee = self.max_fee_per_gas.unwrap_or(Price::new(0));
                let priority_fee = self.max_priority_fee_per_gas.unwrap_or(Price::new(0));

                base_fee.map_or(max_fee, |base| {
                    let total_fee = Price::new(base.wei() + priority_fee.wei());
                    Price::new(total_fee.wei().min(max_fee.wei()))
                })
            }
        }
    }

    /// Calculate transaction fee
    #[must_use]
    pub fn calculate_fee(&self, base_fee: Option<Price>) -> Price {
        let gas_price = self.effective_gas_price(base_fee);
        Price::new(u128::from(self.gas_limit.amount()) * gas_price.wei())
    }

    /// Check if transaction is confirmed
    #[must_use]
    pub const fn is_confirmed(&self) -> bool {
        matches!(self.status, TxStatus::Confirmed)
    }

    /// Check if transaction is pending
    #[must_use]
    pub const fn is_pending(&self) -> bool {
        matches!(self.status, TxStatus::Pending)
    }

    /// Check if transaction failed
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        matches!(self.status, TxStatus::Failed)
    }
}
