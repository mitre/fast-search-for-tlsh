use crate::TlshDigest;

// Trivially-correct linear scan to support benchmarks. Not intended for actual use.
pub fn linear_scan(digests: &[TlshDigest], query: &TlshDigest, cutoff: i32) -> Vec<TlshDigest> {
    digests
        .iter()
        .filter(|&digest| {
            TlshDigest::distance_headers(query, digest) + TlshDigest::distance_bodies(query, digest)
                <= cutoff
        })
        .cloned()
        .collect()
}
