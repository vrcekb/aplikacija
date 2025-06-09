//! Simple Benchmark Test
//!
//! Basic benchmark to test criterion functionality.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Simple benchmark function
fn bench_simple_math(c: &mut Criterion) {
    c.bench_function("simple_math", |b| {
        b.iter(|| {
            let x = black_box(42_i32);
            let y = black_box(24_i32);
            black_box(x + y)
        });
    });
}

/// Benchmark vector operations
fn bench_vector_ops(c: &mut Criterion) {
    c.bench_function("vector_creation", |b| {
        b.iter(|| {
            let vec: Vec<i32> = (0_i32..1000_i32).collect();
            black_box(vec)
        });
    });
}

criterion_group!(simple_benches, bench_simple_math, bench_vector_ops);
criterion_main!(simple_benches);
