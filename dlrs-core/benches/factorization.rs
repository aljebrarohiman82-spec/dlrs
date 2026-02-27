use criterion::{criterion_group, criterion_main, Criterion};
use dlrs_core::LowRankIdentity;
use nalgebra::DMatrix;

fn bench_factorization(c: &mut Criterion) {
    let k = DMatrix::new_random(100, 80);

    c.bench_function("factorize_100x80_rank8", |b| {
        b.iter(|| LowRankIdentity::from_matrix(&k, 8))
    });

    c.bench_function("factorize_100x80_rank16", |b| {
        b.iter(|| LowRankIdentity::from_matrix(&k, 16))
    });

    let lrim = LowRankIdentity::from_matrix(&k, 8);
    c.bench_function("reconstruct_100x80_rank8", |b| {
        b.iter(|| lrim.reconstruct())
    });

    c.bench_function("fingerprint_100x80_rank8", |b| {
        b.iter(|| lrim.fingerprint())
    });
}

criterion_group!(benches, bench_factorization);
criterion_main!(benches);
