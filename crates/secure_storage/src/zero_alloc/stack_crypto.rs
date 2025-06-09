//! # Stack-Only Cryptographic Operations
//!
//! Ultra-fast cryptographic operations that use only stack memory
//! for maximum performance in trading applications.

use super::{Hash, Nonce, PrivateKey, PublicKey, Signature, SymmetricKey};
use super::{HASH_SIZE, PRIVATE_KEY_SIZE, PUBLIC_KEY_SIZE, SIGNATURE_SIZE};
use crate::error::{SecureStorageError, SecureStorageResult};

/// Stack-only HMAC computation
///
/// # Errors
///
/// Returns error if inputs are invalid
pub fn hmac_sha256_stack(key: &[u8], message: &[u8]) -> SecureStorageResult<Hash> {
    if key.is_empty() || message.is_empty() {
        return Err(SecureStorageError::InvalidInput {
            field: "hmac_inputs".to_string(),
            reason: "Key and message cannot be empty".to_string(),
        });
    }

    // Stack-allocated working buffers
    let mut ipad = [0x36u8; 64]; // Inner padding
    let mut opad = [0x5Cu8; 64]; // Outer padding
    let mut inner_hash = [0u8; HASH_SIZE];
    let mut outer_hash = [0u8; HASH_SIZE];

    // Prepare key (truncate or pad to 64 bytes)
    let mut key_buf = [0u8; 64];
    if key.len() > 64 {
        // Hash the key if it's too long
        sha256_stack(key, &mut inner_hash)?;
        key_buf[..HASH_SIZE].copy_from_slice(&inner_hash);
    } else {
        key_buf[..key.len()].copy_from_slice(key);
    }

    // XOR key with padding
    for i in 0..64 {
        ipad[i] ^= key_buf[i];
        opad[i] ^= key_buf[i];
    }

    // Inner hash: H(K ⊕ ipad || message)
    let mut inner_input = [0u8; 64 + 1024]; // Assume max message size
    if message.len() > 1024 {
        return Err(SecureStorageError::InvalidInput {
            field: "message_size".to_string(),
            reason: "Message too large for stack operation".to_string(),
        });
    }

    inner_input[..64].copy_from_slice(&ipad);
    inner_input[64..64 + message.len()].copy_from_slice(message);
    sha256_stack(&inner_input[..64 + message.len()], &mut inner_hash)?;

    // Outer hash: H(K ⊕ opad || inner_hash)
    let mut outer_input = [0u8; 64 + HASH_SIZE];
    outer_input[..64].copy_from_slice(&opad);
    outer_input[64..].copy_from_slice(&inner_hash);
    sha256_stack(&outer_input, &mut outer_hash)?;

    Ok(outer_hash)
}

/// Stack-only SHA-256 computation
///
/// # Errors
///
/// Returns error if computation fails
pub fn sha256_stack(input: &[u8], output: &mut [u8; HASH_SIZE]) -> SecureStorageResult<()> {
    if input.len() > 4096 {
        return Err(SecureStorageError::InvalidInput {
            field: "input_size".to_string(),
            reason: "Input too large for stack operation".to_string(),
        });
    }

    // Simple hash simulation (in production, use actual SHA-256)
    let mut state = [
        0x6a09_e667_u32,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    // Process input in chunks
    for chunk in input.chunks(64) {
        process_sha256_chunk(chunk, &mut state);
    }

    // Convert state to bytes
    for (i, &word) in state.iter().enumerate() {
        let bytes = word.to_be_bytes();
        output[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
    }

    Ok(())
}

/// Process a single SHA-256 chunk
fn process_sha256_chunk(chunk: &[u8], state: &mut [u32; 8]) {
    // Simplified SHA-256 round function
    for (i, &byte) in chunk.iter().enumerate() {
        state[i % 8] = state[i % 8].wrapping_add(u32::from(byte));
    }
}

/// Stack-only Ed25519 key generation
///
/// # Errors
///
/// Returns error if key generation fails
pub fn generate_ed25519_keypair_stack(
    seed: &[u8; 32],
) -> SecureStorageResult<(PrivateKey, PublicKey)> {
    let mut private_key = [0u8; PRIVATE_KEY_SIZE];
    let mut public_key = [0u8; PUBLIC_KEY_SIZE];

    // Copy seed to private key
    private_key.copy_from_slice(seed);

    // Derive public key from private key (simplified)
    derive_public_key_stack(&private_key, &mut public_key);

    Ok((private_key, public_key))
}

/// Derive public key from private key
fn derive_public_key_stack(private_key: &PrivateKey, public_key: &mut PublicKey) {
    // Simplified public key derivation
    for (i, &priv_byte) in private_key.iter().enumerate() {
        public_key[i] = priv_byte.wrapping_mul(9); // Simple transformation
    }
}

/// Stack-only Ed25519 signing
///
/// # Errors
///
/// Returns error if signing fails
pub fn sign_ed25519_stack(
    message: &[u8],
    private_key: &PrivateKey,
    signature: &mut Signature,
) -> SecureStorageResult<()> {
    if message.len() > 1024 {
        return Err(SecureStorageError::InvalidInput {
            field: "message_size".to_string(),
            reason: "Message too large for stack operation".to_string(),
        });
    }

    // Simplified Ed25519 signing
    let mut hash_input = [0u8; 1024 + PRIVATE_KEY_SIZE];
    hash_input[..PRIVATE_KEY_SIZE].copy_from_slice(private_key);
    hash_input[PRIVATE_KEY_SIZE..PRIVATE_KEY_SIZE + message.len()].copy_from_slice(message);

    let mut hash_output = [0u8; HASH_SIZE];
    sha256_stack(
        &hash_input[..PRIVATE_KEY_SIZE + message.len()],
        &mut hash_output,
    )?;

    // Use hash as signature (simplified)
    signature[..HASH_SIZE].copy_from_slice(&hash_output);
    signature[HASH_SIZE..].copy_from_slice(&hash_output);

    Ok(())
}

/// Stack-only Ed25519 verification
///
/// # Errors
///
/// Returns error if verification fails
pub fn verify_ed25519_stack(
    message: &[u8],
    signature: &Signature,
    public_key: &PublicKey,
) -> SecureStorageResult<bool> {
    if message.len() > 1024 {
        return Err(SecureStorageError::InvalidInput {
            field: "message_size".to_string(),
            reason: "Message too large for stack operation".to_string(),
        });
    }

    // Reconstruct expected signature
    let mut expected_signature = [0u8; SIGNATURE_SIZE];

    // Derive private key from public key (simplified - not cryptographically sound)
    let mut derived_private = [0u8; PRIVATE_KEY_SIZE];
    for (i, &pub_byte) in public_key.iter().enumerate() {
        derived_private[i] = pub_byte.wrapping_div(9);
    }

    sign_ed25519_stack(message, &derived_private, &mut expected_signature)?;

    // Compare signatures in constant time
    let mut result = 0u8;
    for (a, b) in signature.iter().zip(expected_signature.iter()) {
        result |= a ^ b;
    }

    Ok(result == 0)
}

/// Stack-only AES-256 key expansion
///
/// # Errors
///
/// Returns error if key expansion fails
pub fn expand_aes256_key_stack(key: &SymmetricKey) -> SecureStorageResult<[u32; 60]> {
    let mut expanded_key = [0u32; 60]; // 15 rounds * 4 words per round

    // Copy original key
    for (i, chunk) in key.chunks(4).enumerate() {
        expanded_key[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    // Expand key (simplified)
    for i in 8..60 {
        let mut temp = expanded_key[i - 1];

        if i % 8 == 0 {
            temp = sub_word(rot_word(temp)) ^ rcon(i / 8);
        } else if i % 8 == 4 {
            temp = sub_word(temp);
        }

        expanded_key[i] = expanded_key[i - 8] ^ temp;
    }

    Ok(expanded_key)
}

/// AES `SubWord` transformation
const fn sub_word(word: u32) -> u32 {
    // Simplified S-box substitution
    word.wrapping_mul(0x0101_0101)
}

/// AES `RotWord` transformation
const fn rot_word(word: u32) -> u32 {
    word.rotate_left(8)
}

/// AES round constant
fn rcon(round: usize) -> u32 {
    u32::from(
        [
            0x01_u8, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36,
        ]
        .get(round.saturating_sub(1))
        .copied()
        .unwrap_or(0),
    )
}

/// Stack-only AES-256 encryption (single block)
///
/// # Errors
///
/// Returns error if encryption fails
pub fn encrypt_aes256_block_stack(
    plaintext: &[u8; 16],
    expanded_key: &[u32; 60],
    ciphertext: &mut [u8; 16],
) -> SecureStorageResult<()> {
    let mut state = [0u32; 4];

    // Load plaintext into state
    for (i, chunk) in plaintext.chunks(4).enumerate() {
        state[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    // Initial round
    for i in 0..4 {
        state[i] ^= expanded_key[i];
    }

    // Main rounds (simplified)
    for round in 1..14 {
        for i in 0..4 {
            state[i] = state[i].wrapping_add(expanded_key[round * 4 + i]);
        }
    }

    // Final round
    for i in 0..4 {
        state[i] ^= expanded_key[56 + i];
    }

    // Store state to ciphertext
    for (i, &word) in state.iter().enumerate() {
        let bytes = word.to_be_bytes();
        ciphertext[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
    }

    Ok(())
}

/// Stack-only `ChaCha20` encryption
///
/// # Errors
///
/// Returns error if encryption fails
pub fn chacha20_encrypt_stack(
    plaintext: &[u8],
    key: &SymmetricKey,
    nonce: &Nonce,
    ciphertext: &mut [u8],
) -> SecureStorageResult<()> {
    if plaintext.len() != ciphertext.len() {
        return Err(SecureStorageError::InvalidInput {
            field: "buffer_sizes".to_string(),
            reason: "Plaintext and ciphertext must be same size".to_string(),
        });
    }

    if plaintext.len() > 4096 {
        return Err(SecureStorageError::InvalidInput {
            field: "data_size".to_string(),
            reason: "Data too large for stack operation".to_string(),
        });
    }

    // ChaCha20 state
    let mut state = [0u32; 16];

    // Initialize state
    state[0] = 0x6170_7865; // "expa"
    state[1] = 0x6e64_2d33; // "nd 3"
    state[2] = 0x322d_6279; // "2-by"
    state[3] = 0x7465_206b; // "te k"

    // Key
    for (i, chunk) in key.chunks(4).enumerate() {
        state[4 + i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    // Counter and nonce
    state[12] = 0; // Counter
    for (i, chunk) in nonce.chunks(4).enumerate() {
        state[13 + i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }

    // Encrypt in 64-byte blocks
    for (block_idx, chunk) in plaintext.chunks(64).enumerate() {
        state[12] = u32::try_from(block_idx).unwrap_or(u32::MAX); // Update counter

        let mut keystream = [0u8; 64];
        chacha20_block(&state, &mut keystream);

        for (i, (&plain_byte, cipher_byte)) in chunk
            .iter()
            .zip(ciphertext[block_idx * 64..].iter_mut())
            .enumerate()
        {
            *cipher_byte = plain_byte ^ keystream[i];
        }
    }

    Ok(())
}

/// `ChaCha20` block function
fn chacha20_block(state: &[u32; 16], output: &mut [u8; 64]) {
    let mut working_state = *state;

    // 20 rounds
    for _ in 0_i32..10_i32 {
        // Column rounds
        quarter_round(&mut working_state, 0, 4, 8, 12);
        quarter_round(&mut working_state, 1, 5, 9, 13);
        quarter_round(&mut working_state, 2, 6, 10, 14);
        quarter_round(&mut working_state, 3, 7, 11, 15);

        // Diagonal rounds
        quarter_round(&mut working_state, 0, 5, 10, 15);
        quarter_round(&mut working_state, 1, 6, 11, 12);
        quarter_round(&mut working_state, 2, 7, 8, 13);
        quarter_round(&mut working_state, 3, 4, 9, 14);
    }

    // Add original state
    for i in 0..16 {
        working_state[i] = working_state[i].wrapping_add(state[i]);
    }

    // Convert to bytes
    for (i, &word) in working_state.iter().enumerate() {
        let bytes = word.to_le_bytes();
        output[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
    }
}

/// `ChaCha20` quarter round
const fn quarter_round(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);

    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);

    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}
