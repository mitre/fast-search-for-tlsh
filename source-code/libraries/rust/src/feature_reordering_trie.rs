use crate::common::mod_diff;
use crate::TlshDigest;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Schema {
    pub feature_order: Vec<usize>,
}

impl Schema {
    pub fn save_index(&self, file: &File) -> io::Result<()> {
        serde_json::to_writer(file, &self.feature_order).expect("Failed to write schema");
        Ok(())
    }

    pub fn load_index(file: &File) -> Self {
        Schema {
            feature_order: serde_json::from_reader(file).expect("Failed to read schema"),
        }
    }
}

fn calculate_feature_difference(t_x: &TlshDigest, t_y: &TlshDigest, features: &[usize]) -> i32 {
    features
        .iter()
        .map(|&f| match f {
            0 => {
                // Checksum
                if t_x.checksum != t_y.checksum {
                    1
                } else {
                    0
                }
            }
            1 => {
                // L-value
                let l_diff = mod_diff(t_x.l as i32, t_y.l as i32, 256);
                match l_diff {
                    0 => 0,
                    1 => 1,
                    _ => l_diff * 12,
                }
            }
            2 => {
                // Q1
                // This is the slow-path. The fast-path should be much faster on most hardware.
                // let q1_diff = mod_diff(t_x.q1 as i32, t_y.q1 as i32, 16);
                // let result = if q1_diff <= 1 {
                //     q1_diff
                // } else {
                //     (q1_diff - 1) * 12
                // };
                // assert_eq!(result, TlshDigest::Q_DIFF_LOOKUP_TABLE[t_x.q1 as usize][t_y.q1 as usize]);
                TlshDigest::Q_DIFF_LOOKUP_TABLE[t_x.q1 as usize][t_y.q1 as usize]
            }
            3 => {
                // Q2
                // This is the slow-path. The fast-path should be faster on more hardware than not.
                // However, the code below is not actually that complex, and may be faster on some.
                // let q2_diff = mod_diff(t_x.q2 as i32, t_y.q2 as i32, 16);
                // let result = if q2_diff <= 1 {
                //     q2_diff
                // } else {
                //     (q2_diff - 1) * 12
                // };
                // assert_eq!(result, TlshDigest::Q_DIFF_LOOKUP_TABLE[t_x.q2 as usize][t_y.q2 as usize]);
                TlshDigest::Q_DIFF_LOOKUP_TABLE[t_x.q2 as usize][t_y.q2 as usize]
            }
            _ => {
                // Body

                // This is the slow-path. The fast-path should be much faster on most hardware.
                // let i = f - 4;
                // let x = t_x.body[i];
                // let y = t_y.body[i];
                //
                // let x_high = x >> 4;
                // let x_low = x & 0x0F;
                // let y_high = y >> 4;
                // let y_low = y & 0x0F;
                //
                // let x1 = x_high >> 2;
                // let x2 = x_high & 0b11;
                // let x3 = x_low >> 2;
                // let x4 = x_low & 0b11;
                //
                // let y1 = y_high >> 2;
                // let y2 = y_high & 0b11;
                // let y3 = y_low >> 2;
                // let y4 = y_low & 0b11;
                //
                // let d1 = (x1 as i32 - y1 as i32).abs();
                // let d2 = (x2 as i32 - y2 as i32).abs();
                // let d3 = (x3 as i32 - y3 as i32).abs();
                // let d4 = (x4 as i32 - y4 as i32).abs();
                //
                // let mut diff = 0;
                //
                // diff += if d1 == 3 { 6 } else { d1 };
                // diff += if d2 == 3 { 6 } else { d2 };
                // diff += if d3 == 3 { 6 } else { d3 };
                // diff += if d4 == 3 { 6 } else { d4 };
                //
                // assert_eq!(TlshDigest::BODY_DIFF_LOOKUP_TABLE[t_x.body[f - 4] as usize][t_y.body[f - 4] as usize], diff);

                TlshDigest::BODY_DIFF_LOOKUP_TABLE[t_x.body[f - 4] as usize]
                    [t_y.body[f - 4] as usize]
            }
        })
        .sum()
}

fn fast_calculate_feature_difference(t_x: u8, t_y: u8, feature: usize) -> i32 {
    match feature {
        0 => {
            // Checksum
            if t_x != t_y {
                1
            } else {
                0
            }
        }
        1 => {
            // L-value
            let l_diff = mod_diff(t_x as i32, t_y as i32, 256);
            match l_diff {
                0 => 0,
                1 => 1,
                _ => l_diff * 12,
            }
        }
        2 | 3 => {
            // Q1 or Q2
            TlshDigest::Q_DIFF_LOOKUP_TABLE[t_x as usize][t_y as usize]
        }
        _ => {
            // Body
            TlshDigest::BODY_DIFF_LOOKUP_TABLE[t_x as usize][t_y as usize]
        }
    }
}

pub fn learn_schema(data: &[TlshDigest], cutoff: i32, header_only: bool) -> Schema {
    let sample_size = min(64, data.len());
    let sampled_data: Vec<_> = data
        .choose_multiple(&mut rand::thread_rng(), sample_size)
        .collect();
    let mut selected_features = Vec::new();
    let mut max_cutoffs = 0;

    loop {
        let mut best_feature = None;
        let mut best_cutoff_count = max_cutoffs;
        let mut best_feature_by_sum = None;
        let mut best_sum = 0;

        let feature_range = if header_only { 0..4 } else { 0..36 };

        for feature_index in feature_range {
            if selected_features.contains(&feature_index) {
                continue;
            }

            let current_features = selected_features
                .iter()
                .chain(std::iter::once(&feature_index))
                .cloned()
                .collect::<Vec<_>>();
            let cutoff_count = sampled_data
                .iter()
                .map(|&v| {
                    sampled_data
                        .iter()
                        .filter(|&&target| {
                            calculate_feature_difference(v, target, &current_features) >= cutoff
                        })
                        .count()
                })
                .sum::<usize>();
            let sum = sampled_data
                .iter()
                .map(|&v| {
                    sampled_data
                        .iter()
                        .map(|&target| calculate_feature_difference(v, target, &current_features))
                        .sum::<i32>()
                })
                .sum();

            if sum > best_sum {
                best_sum = sum;
                best_feature_by_sum = Some(feature_index);
            }

            if cutoff_count > best_cutoff_count {
                best_cutoff_count = cutoff_count;
                best_feature = Some(feature_index);
            }
        }

        if let Some(feature) = best_feature {
            selected_features.push(feature);
        } else if let Some(feature) = best_feature_by_sum {
            selected_features.push(feature);
        } else {
            break;
        }

        max_cutoffs = best_cutoff_count;
    }

    Schema {
        feature_order: selected_features,
    }
}

#[derive(Clone, Debug)]
pub struct TreeNode {
    pub children: HashMap<u8, TreeNode>,
    pub leaves: Vec<TlshDigest>,
}

impl Default for TreeNode {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeNode {
    pub fn new() -> Self {
        TreeNode {
            children: HashMap::new(),
            leaves: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct TrieWithBodyFunctionIndex {
    pub schema: Schema,
    pub root: TreeNode,
}

impl TrieWithBodyFunctionIndex {
    pub fn query(&self, query_point: &TlshDigest, radius: i32) -> Vec<TlshDigest> {
        Self::recursive_query(
            &self.root,
            &self.schema.feature_order,
            query_point,
            0,
            radius,
        )
    }

    pub fn new(schema: Schema, data: Vec<TlshDigest>) -> Self {
        let mut root = TreeNode::new();
        for digest in data {
            insert_into_tree(&mut root, &schema.feature_order, &digest);
        }
        TrieWithBodyFunctionIndex { schema, root }
    }
}

pub fn insert_into_tree(node: &mut TreeNode, features: &[usize], digest: &TlshDigest) {
    const MAX_NUMBER_OF_FEATURES: i32 = 36; // Counting by hand.
    const MAX_NUMBER_OF_FEATURES_TO_USE: i32 = 9;

    if features.is_empty()
        || (MAX_NUMBER_OF_FEATURES - features.len() as i32) > MAX_NUMBER_OF_FEATURES_TO_USE
    {
        node.leaves.push(digest.clone());
        return;
    }

    let current_feature = features[0];
    let feature_value = match current_feature {
        0 => digest.checksum,
        1 => digest.l,
        2 => digest.q1,
        3 => digest.q2,
        _ => digest.body[current_feature - 4],
    };

    let child = node.children.entry(feature_value).or_default();

    insert_into_tree(child, &features[1..], digest);
}

impl TrieWithBodyFunctionIndex {
    pub fn recursive_query(
        node: &TreeNode,
        features: &[usize],
        query_point: &TlshDigest,
        accumulated_distance: i32,
        original_cutoff: i32,
    ) -> Vec<TlshDigest> {
        if node.children.is_empty() {
            return node
                .leaves
                .iter()
                .filter(|&point| {
                    accumulated_distance
                        + calculate_feature_difference(point, query_point, features)
                        <= original_cutoff
                })
                .cloned()
                .collect();
        }

        if features.is_empty() {
            return Vec::new();
        }

        let current_feature = features[0];
        let feature_value = match current_feature {
            0 => query_point.checksum,
            1 => query_point.l,
            2 => query_point.q1,
            3 => query_point.q2,
            _ => query_point.body[current_feature - 4],
        };

        // You could, for instance, further pruning branches by being smarter about modular
        // arithmetic. As it turns out, special-casing L-value by itself gets you 99% of the benefit
        // with essentially zero overhead, because the modulus is so high that the trivial, easily
        // compiler-optimized approach (splitting in half) is very fast, and because L-value diffs
        // matter so much more in practice than anything else.
        // let modulus = match current_feature {
        //     0 => 256,    // Checksum
        //     1 => 256,    // L-value
        //     2 | 3 => 16, // Q1 and Q2
        //     _ => 256,    // Body
        // };

        let remaining_cutoff = original_cutoff - accumulated_distance;
        let max_diff = match current_feature {
            0 => 255,                                         // Checksum
            1 => (remaining_cutoff / 12).clamp(1, 255) as u8, // L-value
            2 | 3 => (remaining_cutoff / 12).min(15) as u8,   // Q1 and Q2
            _ => (remaining_cutoff / 6).min(255) as u8,       // Body
        };

        let mut results = Vec::new();

        if current_feature == 1 {
            // Fast path
            let start = feature_value.saturating_sub(max_diff);
            let end = (feature_value as i32) + (max_diff as i32);

            for candidate_value in start..=255 {
                if let Some(child) = node.children.get(&candidate_value) {
                    let feature_contribution = fast_calculate_feature_difference(
                        candidate_value,
                        feature_value,
                        current_feature,
                    );

                    let new_accumulated_distance = accumulated_distance + feature_contribution;

                    if new_accumulated_distance <= original_cutoff {
                        let sub_results = Self::recursive_query(
                            child,
                            &features[1..],
                            query_point,
                            new_accumulated_distance,
                            original_cutoff,
                        );
                        results.extend(sub_results);
                    }
                }
            }

            if end > 255 {
                for candidate_value in 0..=((end - 256) as u8) {
                    if let Some(child) = node.children.get(&candidate_value) {
                        let feature_contribution = fast_calculate_feature_difference(
                            candidate_value,
                            feature_value,
                            current_feature,
                        );

                        let new_accumulated_distance = accumulated_distance + feature_contribution;

                        if new_accumulated_distance <= original_cutoff {
                            let sub_results = Self::recursive_query(
                                child,
                                &features[1..],
                                query_point,
                                new_accumulated_distance,
                                original_cutoff,
                            );
                            results.extend(sub_results);
                        }
                    }
                }
            }
        } else {
            // More general path
            for (&candidate_value, child) in &node.children {
                let feature_contribution = fast_calculate_feature_difference(
                    candidate_value,
                    feature_value,
                    current_feature,
                );

                let new_accumulated_distance = accumulated_distance + feature_contribution;

                if new_accumulated_distance <= original_cutoff {
                    let sub_results = Self::recursive_query(
                        child,
                        &features[1..],
                        query_point,
                        new_accumulated_distance,
                        original_cutoff,
                    );
                    results.extend(sub_results);
                }
            }
        }

        results
    }
}
