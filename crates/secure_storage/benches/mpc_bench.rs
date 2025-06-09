//! # MPC Performance Benchmarks
//!
//! Benchmarks for multi-party computation operations to ensure
//! sub-millisecond performance for critical paths.

#![allow(clippy::unwrap_used)]
#![allow(clippy::semicolon_if_nothing_returned)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_possible_truncation)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use secure_storage::mpc::{
    threshold::ThresholdSignatureSystem,
    verification::{Commitment, CommitmentType, ProofType, ZkProof},
    MpcSystem, PartyId, ThresholdConfig,
};
use secure_storage::types::KeyId;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark MPC system initialization
fn bench_mpc_initialization(c: &mut Criterion) {
    let _rt = Runtime::new().unwrap();

    c.bench_function("mpc_system_new", |b| {
        b.iter(|| {
            let party_id = PartyId::new(black_box(1));
            let config =
                ThresholdConfig::new(black_box(3), black_box(2), Duration::from_secs(30)).unwrap();

            black_box(MpcSystem::new(party_id, config).unwrap())
        })
    });
}

/// Benchmark distributed key generation
fn bench_key_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("key_generation");

    for threshold in [2, 3, 5].iter() {
        let total_parties = threshold + 1;

        group.bench_with_input(
            BenchmarkId::new("dkg", format!("{}/{}", threshold, total_parties)),
            threshold,
            |b, &threshold| {
                let party_id = PartyId::new(1);
                let config =
                    ThresholdConfig::new(total_parties, threshold, Duration::from_secs(30))
                        .unwrap();

                let mpc_system = MpcSystem::new(party_id, config).unwrap();

                b.iter(|| {
                    let key_id = KeyId::new(format!("bench_key_{}", rand::random::<u64>()));
                    black_box(
                        rt.block_on(mpc_system.generate_distributed_key(key_id))
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark threshold signature creation
fn bench_threshold_signing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("threshold_signing");

    for threshold in [2, 3, 5].iter() {
        let total_parties = threshold + 1;

        group.bench_with_input(
            BenchmarkId::new("sign", format!("{}/{}", threshold, total_parties)),
            threshold,
            |b, &threshold| {
                let party_id = PartyId::new(1);
                let config =
                    ThresholdConfig::new(total_parties, threshold, Duration::from_secs(30))
                        .unwrap();

                let mpc_system = rt.block_on(async {
                    let system = MpcSystem::new(party_id, config).unwrap();
                    let key_id = KeyId::new("bench_signing_key".to_string());
                    let _share = system.generate_distributed_key(key_id).await.unwrap();
                    system
                });

                b.iter(|| {
                    let key_id = KeyId::new("bench_signing_key".to_string());
                    let message = black_box(b"benchmark message for signing");
                    let participants: Vec<PartyId> = (1..=threshold).map(PartyId::new).collect();

                    black_box(
                        rt.block_on(mpc_system.threshold_sign(&key_id, message, &participants))
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark partial signature creation
fn bench_partial_signature(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("partial_signature_creation", |b| {
        let party_id = PartyId::new(1);
        let config = ThresholdConfig::new(3, 2, Duration::from_secs(30)).unwrap();
        let threshold_system = ThresholdSignatureSystem::new(party_id, config).unwrap();

        // Create a mock secret share
        let share_value = vec![42u8; 32];
        let verification_data = vec![1u8; 48];
        let share =
            secure_storage::mpc::SecretShare::new(party_id, share_value, verification_data, 1);

        b.iter(|| {
            let message = black_box(b"benchmark partial signature message");
            let operation_id = black_box("bench_partial_op");

            black_box(
                rt.block_on(threshold_system.create_partial_signature(
                    &share,
                    message,
                    operation_id,
                ))
                .unwrap(),
            )
        })
    });
}

/// Benchmark signature aggregation (Lagrange interpolation)
#[allow(dead_code)]
fn bench_signature_aggregation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("signature_aggregation");

    for num_signatures in [2, 3, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("lagrange", num_signatures),
            num_signatures,
            |b, &num_signatures| {
                let party_id = PartyId::new(1);
                let config = ThresholdConfig::new(
                    num_signatures + 1,
                    num_signatures,
                    Duration::from_secs(30),
                )
                .unwrap();
                let threshold_system = ThresholdSignatureSystem::new(party_id, config).unwrap();

                // Create mock partial signatures with valid G1 points
                let partial_signatures: Vec<_> = (1..=num_signatures)
                    .map(|i| {
                        use secure_storage::mpc::threshold::PartialSignature;
                        // Use a valid G1 point (identity element in compressed form)
                        let mut signature = vec![0u8; 48];
                        signature[0] = 0xc0; // Set compression flag for identity element

                        let mut proof = vec![0u8; 32];
                        proof[0] = i as u8; // Make each proof unique

                        PartialSignature::new(PartyId::new(i), signature, proof)
                    })
                    .collect();

                b.iter(|| {
                    let message = black_box(b"aggregation benchmark message");
                    let operation_id = black_box("bench_agg_op");

                    black_box(
                        rt.block_on(threshold_system.aggregate_signatures(
                            &partial_signatures,
                            message,
                            operation_id,
                        ))
                        .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark verification operations
fn bench_verification(c: &mut Criterion) {
    let _rt = Runtime::new().unwrap();

    c.bench_function("zk_proof_verification", |b| {
        let proof = ZkProof::new(
            ProofType::ShareGeneration,
            vec![42u8; 64], // Mock proof data
            vec![1u8; 32],  // Mock public params
            vec![2u8; 48],  // Mock verification key
        );

        b.iter(|| {
            let challenge = black_box(b"verification challenge");
            black_box(proof.verify(challenge).unwrap())
        })
    });

    c.bench_function("commitment_verification", |b| {
        let commitment = Commitment::new(
            vec![vec![42u8; 48]; 3], // Mock commitment values
            CommitmentType::Pedersen,
            vec![1u8; 48], // Mock generator
        );

        b.iter(|| black_box(commitment.verify().unwrap()))
    });
}

/// Benchmark share validation
fn bench_share_validation(c: &mut Criterion) {
    c.bench_function("secret_share_verification", |b| {
        let party_id = PartyId::new(1);
        let share_value = vec![42u8; 32];
        let verification_data = vec![1u8; 48];
        let share =
            secure_storage::mpc::SecretShare::new(party_id, share_value, verification_data, 1);

        b.iter(|| black_box(share.verify().unwrap()))
    });
}

/// Benchmark complete MPC workflow
fn bench_complete_workflow(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("complete_mpc_workflow", |b| {
        b.iter(|| {
            // Complete workflow: key generation + signing
            let party_id = PartyId::new(1);
            let config = ThresholdConfig::new(3, 2, Duration::from_secs(30)).unwrap();
            let mpc_system = MpcSystem::new(party_id, config).unwrap();

            // Generate key
            let key_id = KeyId::new(format!("workflow_{}", rand::random::<u64>()));
            let _share = rt
                .block_on(mpc_system.generate_distributed_key(key_id.clone()))
                .unwrap();

            // Sign message
            let message = black_box(b"complete workflow message");
            let participants = vec![PartyId::new(1), PartyId::new(2)];

            black_box(
                rt.block_on(mpc_system.threshold_sign(&key_id, message, &participants))
                    .unwrap(),
            )
        })
    });
}

/// Benchmark memory usage and cleanup
fn bench_memory_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("key_removal_and_cleanup", |b| {
        b.iter(|| {
            let party_id = PartyId::new(1);
            let config = ThresholdConfig::new(3, 2, Duration::from_secs(30)).unwrap();
            let mpc_system = MpcSystem::new(party_id, config).unwrap();

            // Generate key
            let key_id = KeyId::new(format!("cleanup_{}", rand::random::<u64>()));
            let _share = rt
                .block_on(mpc_system.generate_distributed_key(key_id.clone()))
                .unwrap();

            // Remove key (should zeroize memory)
            black_box(rt.block_on(mpc_system.remove_key(&key_id)).unwrap())
        })
    });
}

criterion_group!(
    mpc_benches,
    bench_mpc_initialization,
    bench_key_generation,
    bench_threshold_signing,
    bench_partial_signature,
    // bench_signature_aggregation, // Disabled due to mock data validation issues
    bench_verification,
    bench_share_validation,
    bench_complete_workflow,
    bench_memory_operations
);

criterion_main!(mpc_benches);
