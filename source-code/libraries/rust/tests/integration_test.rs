use fast_tlsh::{
    insert_into_tree, learn_schema, plaintext_distance_bodies, plaintext_distance_headers, Schema,
    TlshDigest, TreeNode, TrieWithBodyFunctionIndex, UnwiseVantagePointTree, VantagePointTree,
    TLSH_DIGEST_LENGTH,
};
use rand::prelude::*;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use tempfile::NamedTempFile;

// Helper functions
fn generate_random_hex_digest() -> [u8; TLSH_DIGEST_LENGTH] {
    let mut rng = rand::thread_rng();
    let mut plaintext_digest = [0u8; TLSH_DIGEST_LENGTH];
    for i in 0..TLSH_DIGEST_LENGTH {
        plaintext_digest[i] = rng.gen_range(0..16);
    }
    plaintext_digest
}

fn generate_random_digest() -> TlshDigest {
    TlshDigest::from_bytes(&generate_random_hex_digest())
}

// Distance method tests

#[test]
fn test_distance_headers() {
    let t_x = generate_random_digest();
    assert_eq!(TlshDigest::distance_headers(&t_x, &t_x), 0);

    let t_y = generate_random_digest();
    let t_z = generate_random_digest();
    assert!(TlshDigest::distance_headers(&t_y, &t_z) > 0);
}

#[test]
fn test_distance_bodies() {
    let t_x = generate_random_digest();
    assert_eq!(TlshDigest::distance_bodies(&t_x, &t_x), 0);

    let t_y = generate_random_digest();
    let t_z = generate_random_digest();
    assert!(TlshDigest::distance_bodies(&t_y, &t_z) > 0);
}

#[test]
fn test_distances_all_deterministic() {
    let digest1 = TlshDigest {
        checksum: 0x01,
        l: 0x02,
        q1: 0x03,
        q2: 0x04,
        body: [0x05; 32],
    };

    let digest2 = TlshDigest {
        checksum: 0x01,
        l: 0x02,
        q1: 0x03,
        q2: 0x04,
        body: [0x06; 32],
    };

    let expected_header_distance = 0;
    let expected_body_distance = 32;

    let header_distance = TlshDigest::distance_headers(&digest1, &digest2);
    let body_distance = TlshDigest::distance_bodies(&digest1, &digest2);

    assert_eq!(
        header_distance, expected_header_distance,
        "Header distance mismatch"
    );
    assert_eq!(
        body_distance, expected_body_distance,
        "Body distance mismatch"
    );
}

// Data-structure tests

// Supporting data-structure tests

#[test]
fn test_digest_compaction() {
    let digests: Vec<[u8; 70]> = (0..1000).map(|_| generate_random_hex_digest()).collect();
    for original_digest in digests {
        let compacted = TlshDigest::from_bytes(&original_digest);
        let decompacted = compacted.to_hex();
        assert_eq!(
            original_digest, decompacted,
            "Digests do not match after compaction and de-compaction."
        );
    }
}

#[test]
fn test_compacted_node_distance() {
    let digests: Vec<[u8; TLSH_DIGEST_LENGTH]> =
        (0..1000).map(|_| generate_random_hex_digest()).collect();

    for i in 0..digests.len() {
        for j in (i + 1)..digests.len() {
            let original_x = &digests[i];
            let original_y = &digests[j];

            let compacted_x = TlshDigest::from_bytes(original_x);
            let compacted_y = TlshDigest::from_bytes(original_y);

            let original_distance = plaintext_distance_headers(original_x, original_y)
                + plaintext_distance_bodies(original_x, original_y);
            let compacted_distance = TlshDigest::distance_headers(&compacted_x, &compacted_y)
                + TlshDigest::distance_bodies(&compacted_x, &compacted_y);

            assert_eq!(
                plaintext_distance_headers(original_x, original_y),
                TlshDigest::distance_headers(&compacted_x, &compacted_y),
                "Header distance mismatch between original and compacted digests."
            );

            assert_eq!(
                plaintext_distance_bodies(original_x, original_y),
                TlshDigest::distance_bodies(&compacted_x, &compacted_y),
                "Body distance mismatch between original and compacted digests."
            );

            assert_eq!(
                original_distance, compacted_distance,
                "Total mismatch between original and compacted digests"
            );
        }
    }
}

// Basic data-structure tests
#[test]
fn test_vantage_point_tree_basic() {
    let results = VantagePointTree::new(vec![]).query(&generate_random_digest(), 100);
    assert_eq!(results.len(), 0);

    let node = generate_random_digest();
    let results = VantagePointTree::new(vec![node.clone()]).query(&node, 2);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_trie_basic() {
    TrieWithBodyFunctionIndex::new(
        Schema {
            feature_order: vec![],
        },
        vec![],
    );
    let results = TrieWithBodyFunctionIndex::new(
        Schema {
            feature_order: vec![],
        },
        vec![],
    )
    .query(&generate_random_digest(), 100);
    assert_eq!(results.len(), 0);

    let node = generate_random_digest();
    let results = TrieWithBodyFunctionIndex::new(
        Schema {
            feature_order: vec![],
        },
        vec![node.clone()],
    )
    .query(&node, 2);
    assert_eq!(results.len(), 1);
}

// Test that they're ideal classifiers

#[test]
fn test_vantage_point_tree_idealness() {
    const CUTOFF: i32 = 200;

    let digests: Vec<TlshDigest> = (0..1000).map(|_| generate_random_digest()).collect();
    let tree = VantagePointTree::new(digests.clone());

    for digest in &digests {
        let results = tree.query(digest, CUTOFF);

        // Test for perfect specificity
        for result in &results {
            assert!(
                TlshDigest::distance_headers(digest, result)
                    + TlshDigest::distance_bodies(digest, result)
                    <= CUTOFF
            );
        }

        // Test for perfect sensitivity
        let all_results: HashSet<_> = results.into_iter().collect();
        for other_digest in &digests {
            if !all_results.contains(other_digest) {
                assert!(
                    TlshDigest::distance_headers(digest, other_digest)
                        + TlshDigest::distance_bodies(digest, other_digest)
                        > CUTOFF
                );
            }
        }
    }
}

#[test]
fn test_trie_idealness() {
    const CUTOFF: i32 = 500;

    let digests: Vec<TlshDigest> = (0..1000).map(|_| generate_random_digest()).collect();
    let schema = learn_schema(&*digests, CUTOFF, false);
    let index = TrieWithBodyFunctionIndex::new(schema.clone(), digests.clone());

    for digest in &digests {
        let results = index.query(digest, CUTOFF);

        for result in &results {
            assert!(
                TlshDigest::distance_headers(&digest, &result)
                    + TlshDigest::distance_bodies(&digest, &result)
                    <= CUTOFF
            );
        }

        let all_results: HashSet<_> = results.into_iter().collect();
        for other_digest in &digests {
            if !all_results.contains(other_digest) {
                assert!(
                    TlshDigest::distance_headers(&digest, &other_digest)
                        + TlshDigest::distance_bodies(&digest, &other_digest)
                        >= CUTOFF
                );
            }
        }
    }
}

#[test]
fn test_vantage_point_trees_boundary_conditions() {
    let digest1 = generate_random_digest();
    let digest2 = generate_random_digest();

    let distance = TlshDigest::distance_headers(&digest1, &digest2)
        + TlshDigest::distance_bodies(&digest1, &digest2);

    let digests = vec![digest1.clone(), digest2.clone()];
    let index = VantagePointTree::new(digests);

    let results_above = index.query(&digest1, distance + 1);
    assert!(
        results_above.contains(&digest2),
        "digest2 should be found with cutoff above distance"
    );

    let results_exact = index.query(&digest1, distance);
    assert!(
        results_exact.contains(&digest2),
        "digest2 should be found with cutoff equal to distance"
    );

    for result in &results_exact {
        let calculated_distance = TlshDigest::distance_headers(&digest1, result)
            + TlshDigest::distance_bodies(&digest1, result);
        assert!(
            calculated_distance <= distance,
            "Distance should be within cutoff"
        );
    }

    let results_below = index.query(&digest1, distance - 1);
    assert!(
        !results_below.contains(&digest2),
        "digest2 should not be found with cutoff below distance"
    );
}

#[test]
fn test_trie_boundary_conditions() {
    let digest1 = generate_random_digest();
    let digest2 = generate_random_digest();

    let distance = TlshDigest::distance_headers(&digest1, &digest2)
        + TlshDigest::distance_bodies(&digest1, &digest2);

    let digests = vec![digest1.clone(), digest2.clone()];
    let schema = learn_schema(&digests, distance, false);
    let index = TrieWithBodyFunctionIndex::new(schema, digests);

    let results_above = index.query(&digest1, distance + 1);
    assert!(
        results_above.contains(&digest2),
        "digest2 should be found with cutoff above distance"
    );

    let results_exact = index.query(&digest1, distance);
    assert!(
        results_exact.contains(&digest2),
        "digest2 should be found with cutoff equal to distance"
    );

    for result in &results_exact {
        let calculated_distance = TlshDigest::distance_headers(&digest1, result)
            + TlshDigest::distance_bodies(&digest1, result);
        assert!(
            calculated_distance <= distance,
            "Distance should be within cutoff"
        );
    }

    let results_below = index.query(&digest1, distance - 1);
    assert!(
        !results_below.contains(&digest2),
        "digest2 should not be found with cutoff below distance"
    );
}

#[test]
fn test_trie_feature_handling_in_query() {
    let digest1 = generate_random_digest();
    let digest2 = generate_random_digest();

    let schema = Schema {
        feature_order: vec![0, 1, 2, 3], // Test with a simple schema
    };

    let mut root = TreeNode::new();
    insert_into_tree(&mut root, &schema.feature_order, &digest1);
    insert_into_tree(&mut root, &schema.feature_order, &digest2);

    let index = TrieWithBodyFunctionIndex { schema, root };

    let results = index.query(&digest1, 100000);

    // Ensure that the query returns both digests
    assert!(results.contains(&digest1), "Query should return digest1");
    assert!(results.contains(&digest2), "Query should return digest2");
}

#[test]
fn test_trie_traversal_visits_all() {
    let digest1 = generate_random_digest();
    let digest2 = generate_random_digest();

    let schema = Schema {
        feature_order: vec![0, 1, 2, 3],
    };

    let mut root = TreeNode::new();
    insert_into_tree(&mut root, &schema.feature_order, &digest1);
    insert_into_tree(&mut root, &schema.feature_order, &digest2);

    let index = TrieWithBodyFunctionIndex { schema, root };

    let results = TrieWithBodyFunctionIndex::recursive_query(
        &index.root,
        &index.schema.feature_order,
        &digest1,
        0,
        i32::MAX,
    );

    assert!(
        results.contains(&digest1),
        "Recursive query should find digest1"
    );
    assert!(
        results.contains(&digest2),
        "Recursive query should find digest2"
    );
}

#[test]
fn test_trie_schema_learning() {
    let digests: Vec<TlshDigest> = (0..100).map(|_| generate_random_digest()).collect();
    let schema = learn_schema(&digests, 500, false);

    assert!(!schema.feature_order.is_empty(), "Schema not empty!");
    assert!(
        schema.feature_order.len() <= 36,
        "Schema larger than max allowable feature set size!"
    );

    for &feature in &schema.feature_order {
        assert!(feature < 36, "Feature indices seem corrupt");
    }
}

#[test]
fn test_trie_schema_io() {
    let digests: Vec<TlshDigest> = (0..4).map(|_| generate_random_digest()).collect();
    let schema = learn_schema(&digests, 500, false);

    let schema_file = NamedTempFile::new().unwrap();
    schema.save_index(&schema_file.as_file()).unwrap();

    assert_eq!(schema, Schema::load_index(&schema_file.reopen().unwrap()));
}

#[test]
fn test_trie_constructor_includes_all() {
    let root = &generate_random_digest();
    let leaf = &generate_random_digest();

    let digests = vec![root.clone(), leaf.clone()];
    let schema = learn_schema(&digests, 1, false);
    let index = TrieWithBodyFunctionIndex::new(schema.clone(), digests.clone());

    let results1 = index.query(&root, i32::MAX);
    let results2 = index.query(&leaf, i32::MAX);

    assert!(results1.contains(&root), "root should be found in results1");
    assert!(results2.contains(&leaf), "leaf should be found in results2");

    assert!(
        results2.contains(&root),
        "digest1 should be found in results2"
    );
    assert!(results1.contains(&leaf), "leaf should be found in results1");
}

#[test]
fn test_trie_node_insertion() {
    let digest = TlshDigest {
        checksum: 1,
        l: 2,
        q1: 3,
        q2: 4,
        body: [5; 32],
    };

    let mut root = TreeNode::new();
    assert_eq!(
        root.children.len(),
        0,
        "Expected empty TreeNode to have size 0"
    );

    let features = vec![0, 1, 2, 3, 4];
    insert_into_tree(&mut root, &features, &digest);

    assert!(
        root.children.len() == 1 || root.leaves.len() == 1,
        "Expected singleton TreeNode to have size 1"
    );
}

#[test]
fn test_trie_zero_accumulated_distance_handling() {
    let root = &generate_random_digest();
    let leaf = &generate_random_digest();

    let digests = vec![root.clone(), leaf.clone()];
    let schema = learn_schema(&digests, 1, false);
    let index = TrieWithBodyFunctionIndex::new(schema, digests);

    // I thought I had a bug where when I don't accumulate any distance traversing from the root,
    // strange things happened and results could get "lost". Here, I guarantee I'll get results by
    // making the radius infinite and searching for a leaf in a two-node trie, guaranteeing I don't
    // accumulate any distance and that I should match against the leaf.
    // This ended up not being a bug, but the test really doesn't hurt to have.
    let results = index.query(&root, i32::MAX);

    assert!(
        results.contains(&leaf),
        "digest2 should be found with zero accumulated distance"
    );
}

// Performance-related tests

#[test]
fn test_trie_performance_query() {
    use std::time::Instant;

    const NUM_DIGESTS_TO_GENERATE: usize = 1_000_000;
    const NUM_DIGESTS_TO_CHECK: usize = 100_000;
    const CUTOFF: i32 = 30;

    let digests: Vec<TlshDigest> = (0..NUM_DIGESTS_TO_GENERATE)
        .into_par_iter()
        .map(|_| generate_random_digest())
        .collect();

    let schema = learn_schema(&digests, CUTOFF, false);
    let index = TrieWithBodyFunctionIndex::new(schema, digests.clone());

    let start_time = Instant::now();
    for digest in &digests[..NUM_DIGESTS_TO_CHECK] {
        black_box(index.query(digest, CUTOFF));
    }
    let duration = start_time.elapsed();

    println!(
        "Querying {} digests took {:?}",
        NUM_DIGESTS_TO_CHECK, duration
    );
}

#[test]
fn test_vantage_point_tree_performance_query() {
    const NUM_DIGESTS_TO_GENERATE: usize = 1_000_000;
    const NUM_DIGESTS_TO_CHECK: usize = 100_000;
    const RADIUS: i32 = 30;

    let digests: Vec<TlshDigest> = (0..NUM_DIGESTS_TO_GENERATE)
        .into_par_iter()
        .map(|_| generate_random_digest())
        .collect();
    let tree = VantagePointTree::new(digests.clone());

    let start_time = Instant::now();
    for digest in &digests[..NUM_DIGESTS_TO_CHECK] {
        black_box(tree.query(digest, RADIUS));
    }

    let duration = start_time.elapsed();

    println!(
        "Querying {} digests took {:?}",
        NUM_DIGESTS_TO_CHECK, duration
    );
}

#[test]
fn test_vantage_point_tree_performance_build() {
    const NUM_DIGESTS_TO_GENERATE: usize = 100_000_000;

    let digests: Vec<TlshDigest> = (0..NUM_DIGESTS_TO_GENERATE)
        .into_par_iter()
        .map(|_| generate_random_digest())
        .collect();

    black_box(VantagePointTree::new(digests));
}

#[test]
fn test_trie_performance_build() {
    const NUM_DIGESTS_TO_GENERATE: usize = 100_000_000;

    let digests: Vec<TlshDigest> = (0..NUM_DIGESTS_TO_GENERATE)
        .into_par_iter()
        .map(|_| generate_random_digest())
        .collect();

    black_box(TrieWithBodyFunctionIndex::new(
        Schema {
            feature_order: vec![],
        },
        digests,
    ));
}

// Testing the method seen in the HAC-T paper.
const DEFAULT_INPUT_LIST_PATH: &str = "benchmark_inputs/virusshare_data.txt";

fn read_hashes_from_file(file_path: &str) -> Vec<[u8; TLSH_DIGEST_LENGTH]> {
    let file = File::open(file_path).expect("Unable to open file");
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| {
            let line = line.expect("Unable to read line");
            let mut digest = [0u8; TLSH_DIGEST_LENGTH];
            for (i, byte) in line.bytes().take(TLSH_DIGEST_LENGTH).enumerate() {
                digest[i] = byte;
            }
            digest
        })
        .collect()
}

#[test]
fn test_deliberately_wrong_vp_tree() {
    const RNG_SEED: u64 = 1337;
    const MAX_TRIANGLE_INEQUALITY_VIOLATION: i32 = 430;
    const WRONG_TRIANGLE_INEQUALITY_VIOLATION: i32 = 20;
    const CUTOFF: i32 = 30;
    const NUM_SAMPLES: usize = 1000;

    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let raw_digests = read_hashes_from_file(DEFAULT_INPUT_LIST_PATH);

    let corpus: Vec<TlshDigest> = raw_digests
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();

    let wise_tree = VantagePointTree::new(corpus.clone());
    let unwise_tree = UnwiseVantagePointTree::new(corpus.clone());

    let query_raw: Vec<_> = raw_digests
        .choose_multiple(&mut rng, NUM_SAMPLES)
        .cloned()
        .collect();

    let query_digests: Vec<TlshDigest> = query_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();

    let mut false_negative_count = 0;

    for digest in &query_digests {
        let wise_results: Vec<_> = wise_tree.query(digest, CUTOFF);
        let unwise_results_incorrect_bounds: Vec<_> =
            unwise_tree.query(digest, CUTOFF, WRONG_TRIANGLE_INEQUALITY_VIOLATION);
        let unwise_results_correct_bounds: Vec<_> =
            unwise_tree.query(digest, CUTOFF, MAX_TRIANGLE_INEQUALITY_VIOLATION);

        for result in &wise_results {
            if !unwise_results_incorrect_bounds.contains(result) {
                false_negative_count += 1;
            }
            assert!(
                unwise_results_correct_bounds.contains(result),
                "Correct bounds should have no false positives!"
            );
        }

        for result in &unwise_results_incorrect_bounds {
            assert!(
                wise_results.contains(result),
                "Unwise results (overly tightened) has false positives!"
            );
        }
        for result in &unwise_results_correct_bounds {
            assert!(
                wise_results.contains(result),
                "Unwise results (correctly tightened) has false positives!"
            );
        }
    }

    assert!(
        false_negative_count > 0,
        "Expected UnwiseVantagePointTree to miss some results"
    );
}

#[test]
fn test_deliberately_wrong_vp_tree_tighter() {
    const RNG_SEED: u64 = 1337;
    const MAX_TRIANGLE_INEQUALITY_VIOLATION: i32 = 430;
    const WRONG_TRIANGLE_INEQUALITY_VIOLATION: i32 = 20;
    const CUTOFF: i32 = 20;
    const NUM_SAMPLES: usize = 1000;

    let mut rng = StdRng::seed_from_u64(RNG_SEED);

    let raw_digests = read_hashes_from_file(DEFAULT_INPUT_LIST_PATH);

    let corpus: Vec<TlshDigest> = raw_digests
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();

    let wise_tree = VantagePointTree::new(corpus.clone());
    let unwise_tree = UnwiseVantagePointTree::new(corpus.clone());

    let query_raw: Vec<_> = raw_digests
        .choose_multiple(&mut rng, NUM_SAMPLES)
        .cloned()
        .collect();

    let query_digests: Vec<TlshDigest> = query_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();

    let mut false_negative_count = 0;

    for digest in &query_digests {
        let wise_results: Vec<_> = wise_tree.query(digest, CUTOFF);
        let unwise_results_incorrect_bounds: Vec<_> =
            unwise_tree.query(digest, CUTOFF, WRONG_TRIANGLE_INEQUALITY_VIOLATION);
        let unwise_results_correct_bounds: Vec<_> =
            unwise_tree.query(digest, CUTOFF, MAX_TRIANGLE_INEQUALITY_VIOLATION);

        // Count results found by wise tree but missing from unwise tree
        for result in &wise_results {
            if !unwise_results_incorrect_bounds.contains(result) {
                false_negative_count += 1;
            }
            assert!(
                unwise_results_correct_bounds.contains(result),
                "Correct bounds should have no false positives!"
            );
        }

        for result in &unwise_results_incorrect_bounds {
            assert!(
                wise_results.contains(result),
                "Unwise results (overly tightened) has false positives!"
            );
        }
        for result in &unwise_results_correct_bounds {
            assert!(
                wise_results.contains(result),
                "Unwise results (correctly tightened) has false positives!"
            );
        }
    }

    assert!(
        false_negative_count > 0,
        "Expected UnwiseVantagePointTree to miss some results"
    );
}
