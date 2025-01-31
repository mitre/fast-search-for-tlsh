use crate::common::{TlshDigest, MAX_HEADER_TRIANGLE_INEQUALITY_VIOLATION};
use rand::seq::SliceRandom;
use std::cmp::min;

#[derive(Clone)]
struct VPNode {
    point: TlshDigest,
    threshold: i32,
    left: Option<Box<VPNode>>,
    right: Option<Box<VPNode>>,
}

struct VPTreeImpl {
    root: Option<Box<VPNode>>,
    size: usize,
}

impl VPTreeImpl {
    fn new() -> Self {
        VPTreeImpl {
            root: None,
            size: 0,
        }
    }

    fn insert(&mut self, point: TlshDigest) {
        let root = self.root.take();
        self.root = Self::_insert(root, point);
        self.size += 1;
        if self.size % 10_000 == 0 {
            self._rebalance();
        }
    }

    fn _insert(node: Option<Box<VPNode>>, point: TlshDigest) -> Option<Box<VPNode>> {
        match node {
            None => Some(Box::new(VPNode {
                point,
                threshold: 0,
                left: None,
                right: None,
            })),
            Some(mut n) => {
                let dist = TlshDigest::distance_headers(&n.point, &point);
                if dist <= n.threshold {
                    n.left = Self::_insert(n.left.take(), point);
                } else {
                    n.right = Self::_insert(n.right.take(), point);
                }
                Some(n)
            }
        }
    }

    fn _rebalance(&mut self) {
        let mut nodes = Vec::new();
        Self::_collect_nodes(&self.root, &mut nodes);
        self.root = Self::_build_tree(&mut nodes);
    }

    fn _collect_nodes(node: &Option<Box<VPNode>>, nodes: &mut Vec<TlshDigest>) {
        if let Some(n) = node {
            nodes.push(n.point.clone());
            Self::_collect_nodes(&n.left, nodes);
            Self::_collect_nodes(&n.right, nodes);
        }
    }

    fn _build_tree(nodes: &mut [TlshDigest]) -> Option<Box<VPNode>> {
        if nodes.is_empty() {
            return None;
        }

        let mut node = VPNode {
            point: nodes[0].clone(),
            threshold: 0,
            left: None,
            right: None,
        };

        if nodes.len() == 1 {
            return Some(Box::new(node));
        }

        let sample_size = min(16, nodes.len() - 1);
        let mut sampled_distances: Vec<(i32, usize)> = (1..=sample_size)
            .map(|i| (TlshDigest::distance_headers(&node.point, &nodes[i]), i))
            .collect();
        sampled_distances.select_nth_unstable_by_key(sample_size / 2, |k| k.0);
        let median_index = sample_size / 2;
        let estimated_median = sampled_distances[median_index].0;

        let mut left = 1;
        let mut right = nodes.len() - 1;

        while left <= right {
            while left <= right
                && TlshDigest::distance_headers(&node.point, &nodes[left]) < estimated_median
            {
                left += 1;
            }
            while left <= right
                && TlshDigest::distance_headers(&node.point, &nodes[right]) >= estimated_median
            {
                right -= 1;
            }
            if left < right {
                nodes.swap(left, right);
                left += 1;
                right -= 1;
            }
        }

        node.threshold = estimated_median;
        let (left_slice, right_slice) = nodes.split_at_mut(left);

        let (left_tree, right_tree) = rayon::join(
            || Self::_build_tree(&mut left_slice[1..]),
            || Self::_build_tree(right_slice),
        );

        node.left = left_tree;
        node.right = right_tree;

        Some(Box::new(node))
    }

    pub fn range_query(&self, query_point: &TlshDigest, radius: i32) -> Vec<TlshDigest> {
        let mut results = Vec::with_capacity(self.size / 6);
        let mut stack = Vec::with_capacity(100);
        let bounds_adjusted_radius = radius + MAX_HEADER_TRIANGLE_INEQUALITY_VIOLATION;

        stack.push(&self.root);

        while let Some(node) = stack.pop() {
            if let Some(n) = node {
                // Perhaps another day this would be worth experimenting with.
                // Stabilized cross-architecture prefetch options are limited.
                // unsafe { std::intrinsics::prefetch_read_instruction(&n.left, 3); }
                // unsafe { std::intrinsics::prefetch_read_instruction(&n.right, 3); }

                let header_distance = TlshDigest::distance_headers(&n.point, query_point);
                let body_distance = TlshDigest::distance_bodies(&n.point, query_point);

                if header_distance + body_distance <= radius {
                    results.push(n.point.clone());
                }

                if header_distance - bounds_adjusted_radius <= n.threshold {
                    stack.push(&n.left);
                }
                if header_distance + bounds_adjusted_radius >= n.threshold {
                    stack.push(&n.right);
                }
            }
        }

        results
    }

    fn bulk_insert(&mut self, new_nodes: Vec<TlshDigest>) {
        let mut all_nodes = Vec::new();
        Self::_collect_nodes(&self.root, &mut all_nodes);
        all_nodes.extend(new_nodes);
        all_nodes.shuffle(&mut rand::thread_rng());
        let new_root = Self::_build_tree(&mut all_nodes.clone());
        self.root = new_root;
        self.size = all_nodes.len();
    }
}

pub struct VantagePointTree {
    tree: VPTreeImpl,
}

impl VantagePointTree {
    pub fn new(nodes: Vec<TlshDigest>) -> Self {
        let mut tree = VPTreeImpl::new();
        tree.bulk_insert(nodes);
        VantagePointTree { tree }
    }

    pub fn insert(&mut self, point: TlshDigest) {
        self.tree.insert(point);
    }

    pub fn bulk_insert(&mut self, new_nodes: Vec<TlshDigest>) {
        self.tree.bulk_insert(new_nodes);
    }

    pub fn query(&self, query_point: &TlshDigest, radius: i32) -> Vec<TlshDigest> {
        self.tree.range_query(query_point, radius)
    }
}
