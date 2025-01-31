use mimalloc::MiMalloc; // Ideally, this would be a build flag. (It only matters for performance.)

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod common;
mod feature_reordering_trie;
mod linear_scan;
mod unwise_vantage_point_tree;
mod vantage_point_tree;

pub use common::{
    plaintext_distance_bodies, plaintext_distance_headers, TlshDigest, TLSH_DIGEST_LENGTH,
};

pub use feature_reordering_trie::{
    insert_into_tree, learn_schema, Schema, TreeNode, TrieWithBodyFunctionIndex,
};

pub use unwise_vantage_point_tree::UnwiseVantagePointTree;
pub use vantage_point_tree::VantagePointTree;

pub use linear_scan::linear_scan;
