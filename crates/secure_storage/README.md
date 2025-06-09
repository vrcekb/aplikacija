# TallyIO Secure Storage Module

🔐 **Ultra-secure storage for sensitive financial data with <10ms latency**

## Overview

The Secure Storage module provides enterprise-grade encryption and secure key management for TallyIO's financial trading platform. It implements multiple layers of security including AES-256-GCM encryption, Argon2id key derivation, HSM integration, and comprehensive access control.

## Features

### 🔒 **Encryption Layer**
- **AES-256-GCM** encryption for data at rest
- **ChaCha20-Poly1305** for high-performance encryption
- **Argon2id** key derivation function
- **Secure memory wiping** with zeroize
- **Constant-time operations** to prevent timing attacks

### 🏦 **Vault Interface**
- **Local encrypted vault** using SQLite
- **HashiCorp Vault** integration for production
- **HSM support** via PKCS#11 interface
- **Key rotation** and lifecycle management

### 🛡️ **Access Control**
- **RBAC** (Role-Based Access Control)
- **2FA** support for sensitive operations
- **Time-based access tokens**
- **Rate limiting** per role

### 📊 **Audit & Monitoring**
- **Immutable audit trail** for all operations
- **Real-time monitoring** of access patterns
- **Anomaly detection** for suspicious activity
- **Comprehensive logging** with structured data

## Performance Targets

| Operation | Target Latency | Achieved |
|-----------|---------------|----------|
| Key retrieval | < 10ms | ✅ |
| Encryption (1KB) | < 5ms | ✅ |
| Decryption (1KB) | < 5ms | ✅ |
| Vault operations | < 20ms | ✅ |

## Architecture

```rust
pub struct SecureStorage {
    vault: Arc<dyn Vault>,
    encryption: Arc<dyn Encryption>,
    access_control: Arc<AccessControl>,
    audit_log: Arc<AuditLog>,
}
```

### Core Traits

```rust
pub trait Vault: Send + Sync {
    async fn store(&self, key: &str, value: &[u8]) -> Result<()>;
    async fn retrieve(&self, key: &str) -> Result<Vec<u8>>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
}

pub trait Encryption: Send + Sync {
    fn encrypt(&self, data: &[u8], key_id: &str) -> Result<Vec<u8>>;
    fn decrypt(&self, data: &[u8], key_id: &str) -> Result<Vec<u8>>;
    fn generate_key(&self) -> Result<KeyMaterial>;
}
```

## Usage

### Basic Operations

```rust
use secure_storage::{SecureStorage, LocalVault, AesGcmEncryption};

// Initialize secure storage
let vault = LocalVault::new("./secure.db").await?;
let encryption = AesGcmEncryption::new()?;
let storage = SecureStorage::new(vault, encryption).await?;

// Store sensitive data
storage.store("api_key", b"secret_key_data").await?;

// Retrieve data
let data = storage.retrieve("api_key").await?;
```

### Production Setup with HashiCorp Vault

```rust
use secure_storage::{SecureStorage, HashiCorpVault, AesGcmEncryption};

let vault = HashiCorpVault::new("https://vault.company.com", token).await?;
let encryption = AesGcmEncryption::new()?;
let storage = SecureStorage::new(vault, encryption).await?;
```

## Security Features

### Memory Protection
- **mlock()** prevents swapping to disk
- **zeroize** clears sensitive data from memory
- **Secure allocators** for cryptographic operations

### Key Management
- **Key rotation** with configurable intervals
- **Key derivation** using Argon2id
- **Hardware security modules** (HSM) support
- **Multi-signature** key operations

### Access Control
- **Role-based permissions**
- **Time-limited access tokens**
- **IP-based restrictions**
- **Rate limiting** and throttling

## Testing

```bash
# Run all tests
cargo test

# Run with coverage
cargo test --features coverage

# Run benchmarks
cargo bench

# Security audit
cargo audit
```

## Configuration

```toml
[secure_storage]
vault_type = "local"  # or "hashicorp" or "hsm"
encryption_algorithm = "aes-256-gcm"
key_rotation_interval = "24h"
audit_retention = "1y"

[vault.local]
database_path = "./secure.db"
encryption_key_file = "./master.key"

[vault.hashicorp]
url = "https://vault.company.com"
mount_path = "secret"
```

## License

MIT License - see LICENSE file for details.
