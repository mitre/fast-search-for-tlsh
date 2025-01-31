use fast_tlsh::{
    learn_schema, linear_scan, TlshDigest, TrieWithBodyFunctionIndex, UnwiseVantagePointTree,
    VantagePointTree, TLSH_DIGEST_LENGTH,
};
use itertools::Itertools;
use rand::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::create_dir_all;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::path::PathBuf;
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use sysinfo::{Components, System};

// Definitions
const SEED: u64 = 1337;
const DEFAULT_INPUT_LIST_PATH: &str = "benchmark_inputs/virusshare_data.txt";
const OUTPUT_ROOT: &str = "target/bench";
const NUM_SAMPLES: usize = 11;

struct WorkloadPattern {
    corpus_size: usize,
    query_size: usize,
}

// Based on Python test suite.
const WORKLOAD_PATTERNS: &[WorkloadPattern] = &[
    WorkloadPattern {
        corpus_size: 1_000_000,
        query_size: 1000,
    },
    WorkloadPattern {
        corpus_size: 10,
        query_size: 1_000_000,
    },
    WorkloadPattern {
        corpus_size: 1_000_000,
        query_size: 10,
    },
    WorkloadPattern {
        corpus_size: 10000,
        query_size: 10000,
    },
    WorkloadPattern {
        corpus_size: 5000,
        query_size: 5000,
    },
];

#[derive(Clone)]
enum InputSource {
    Random {
        num_digests: usize,
    },
    RandomFixedSize {
        fixed_size: usize,
        num_digests: usize,
    },
    File {
        path: &'static str,
    },
    FileFixedSize {
        fixed_size: usize,
        path: &'static str,
    },
}

struct IoMeasurement {
    preprocessing_time: std::time::Duration,
    schema_learn_time: Option<std::time::Duration>,
    data_clone_time: std::time::Duration,
    structure_build_time: std::time::Duration,
    query_time: std::time::Duration,
    total_time: std::time::Duration,
    num_matches: usize,
    timestamp: u128,
    median_cpu_core_temperature: f32,
    maximum_cpu_core_frequency: f32,
}

fn get_cpu_statistics() -> (f32, f32) {
    let system: System = System::new_all();
    let max_cpu_frequency = system
        .cpus()
        .iter()
        .map(|x| x.frequency())
        .max()
        .unwrap_or(0);

    let temperatures: Vec<f32> = Components::new_with_refreshed_list()
        .into_iter()
        .filter_map(|x| x.label().starts_with("PMU tdie").then(|| x.temperature())?)
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .collect();

    let median_temperature = temperatures[temperatures.len() / 2];

    (max_cpu_frequency as f32, median_temperature)
}

const ASCII_LOOKUP_TABLE: [u8; 16] = [
    b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'a', b'b', b'c', b'd', b'e', b'f',
];

fn generate_random_ascii_digest() -> [u8; TLSH_DIGEST_LENGTH] {
    let mut rng = thread_rng();
    let mut plaintext_digest = [0u8; TLSH_DIGEST_LENGTH];
    for i in 0..TLSH_DIGEST_LENGTH {
        let rand_val = rng.gen_range(0..16);
        plaintext_digest[i] = ASCII_LOOKUP_TABLE[rand_val];
    }
    plaintext_digest
}

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

fn benchmark_workload_vp_tree(
    raw_digests: &[[u8; TLSH_DIGEST_LENGTH]],
    pattern: &WorkloadPattern,
    cutoff: i32,
    rng: &mut StdRng,
) -> IoMeasurement {
    let total_start = Instant::now();

    let corpus_raw: Vec<_> = raw_digests
        .choose_multiple(rng, pattern.corpus_size)
        .cloned()
        .collect();

    let preprocess_start = Instant::now();
    let corpus: Vec<TlshDigest> = corpus_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let preprocessing_time = preprocess_start.elapsed();
    let clone_start = Instant::now();
    let corpus_clone = corpus.clone();
    let data_clone_time = clone_start.elapsed();
    let build_start = Instant::now();
    let tree = VantagePointTree::new(corpus_clone);
    let structure_build_time = build_start.elapsed();
    let query_raw: Vec<_> = raw_digests
        .choose_multiple(rng, pattern.query_size)
        .cloned()
        .collect();
    let query_digests: Vec<TlshDigest> = query_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let query_start = Instant::now();
    let mut num_matches = 0;
    for digest in &query_digests {
        num_matches += black_box(tree.query(digest, cutoff)).len();
    }
    let query_time = query_start.elapsed();
    let total_time = total_start.elapsed();
    let (maximum_cpu_core_frequency, median_cpu_core_temperature) = get_cpu_statistics();

    IoMeasurement {
        preprocessing_time: preprocessing_time,
        data_clone_time: data_clone_time,
        structure_build_time: structure_build_time,
        query_time: query_time,
        total_time: total_time,
        num_matches: num_matches,
        maximum_cpu_core_frequency: maximum_cpu_core_frequency,
        median_cpu_core_temperature: median_cpu_core_temperature,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        schema_learn_time: None,
    }
}

fn benchmark_workload_unwise_vp_tree(
    raw_digests: &[[u8; TLSH_DIGEST_LENGTH]],
    pattern: &WorkloadPattern,
    cutoff: i32,
    rng: &mut StdRng,
    constant_offset: i32,
) -> IoMeasurement {
    let total_start = Instant::now();

    let corpus_raw: Vec<_> = raw_digests
        .choose_multiple(rng, pattern.corpus_size)
        .cloned()
        .collect();

    let preprocess_start = Instant::now();
    let corpus: Vec<TlshDigest> = corpus_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let preprocessing_time = preprocess_start.elapsed();
    let clone_start = Instant::now();
    let corpus_clone = corpus.clone();
    let data_clone_time = clone_start.elapsed();
    let build_start = Instant::now();
    let tree = UnwiseVantagePointTree::new(corpus_clone);
    let structure_build_time = build_start.elapsed();
    let query_raw: Vec<_> = raw_digests
        .choose_multiple(rng, pattern.query_size)
        .cloned()
        .collect();
    let query_digests: Vec<TlshDigest> = query_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let query_start = Instant::now();
    let mut num_matches = 0;
    for digest in &query_digests {
        num_matches += black_box(tree.query(digest, cutoff, constant_offset)).len();
    }
    let query_time = query_start.elapsed();
    let total_time = total_start.elapsed();
    let (maximum_cpu_core_frequency, median_cpu_core_temperature) = get_cpu_statistics();

    IoMeasurement {
        preprocessing_time: preprocessing_time,
        data_clone_time: data_clone_time,
        structure_build_time: structure_build_time,
        query_time: query_time,
        total_time: total_time,
        num_matches: num_matches,
        maximum_cpu_core_frequency: maximum_cpu_core_frequency,
        median_cpu_core_temperature: median_cpu_core_temperature,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        schema_learn_time: None,
    }
}

fn benchmark_workload_trie(
    raw_digests: &[[u8; TLSH_DIGEST_LENGTH]],
    pattern: &WorkloadPattern,
    cutoff: i32,
    rng: &mut StdRng,
) -> IoMeasurement {
    let total_start = Instant::now();
    let corpus_raw = raw_digests
        .choose_multiple(rng, pattern.corpus_size)
        .cloned()
        .collect::<Vec<_>>();
    let preprocess_start = Instant::now();
    let corpus: Vec<TlshDigest> = corpus_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let preprocessing_time = preprocess_start.elapsed();
    let clone_start = Instant::now();
    let corpus_clone = corpus.clone();
    let data_clone_time = clone_start.elapsed();
    let schema_start = Instant::now();
    let schema = learn_schema(&corpus_clone, cutoff, false);
    let schema_learn_time = Some(schema_start.elapsed());
    let build_start = Instant::now();
    let index = TrieWithBodyFunctionIndex::new(schema, corpus_clone);
    let structure_build_time = build_start.elapsed();
    let query_raw = raw_digests
        .choose_multiple(rng, pattern.query_size)
        .cloned()
        .collect::<Vec<_>>();
    let query_digests: Vec<TlshDigest> = query_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let query_start = Instant::now();
    let mut num_matches = 0;
    for digest in &query_digests {
        num_matches += black_box(index.query(digest, cutoff)).len();
    }
    let query_time = query_start.elapsed();
    let total_time = total_start.elapsed();
    let (maximum_cpu_core_frequency, median_cpu_core_temperature) = get_cpu_statistics();

    IoMeasurement {
        preprocessing_time: preprocessing_time,
        schema_learn_time: schema_learn_time,
        data_clone_time: data_clone_time,
        structure_build_time: structure_build_time,
        query_time: query_time,
        total_time: total_time,
        num_matches: num_matches,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        maximum_cpu_core_frequency: maximum_cpu_core_frequency,
        median_cpu_core_temperature: median_cpu_core_temperature,
    }
}

fn benchmark_workload_linear_scan(
    raw_digests: &[[u8; TLSH_DIGEST_LENGTH]],
    pattern: &WorkloadPattern,
    cutoff: i32,
    rng: &mut StdRng,
) -> IoMeasurement {
    let total_start = Instant::now();
    let corpus_raw: Vec<_> = raw_digests
        .choose_multiple(rng, pattern.corpus_size)
        .cloned()
        .collect();
    let preprocess_start = Instant::now();
    let corpus: Vec<TlshDigest> = corpus_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let preprocessing_time = preprocess_start.elapsed();
    let clone_start = Instant::now();
    let corpus_clone = corpus.clone();
    let data_clone_time = clone_start.elapsed();
    let structure_build_time = std::time::Duration::from_secs(0);
    let query_raw: Vec<_> = raw_digests
        .choose_multiple(rng, pattern.query_size)
        .cloned()
        .collect();
    let query_digests: Vec<TlshDigest> = query_raw
        .iter()
        .map(|d| TlshDigest::from_plaintext(d))
        .collect();
    let query_start = Instant::now();
    let mut num_matches = 0;
    for digest in &query_digests {
        num_matches += black_box(linear_scan(&corpus_clone, digest, cutoff)).len();
    }
    let query_time = query_start.elapsed();
    let total_time = total_start.elapsed();
    let (maximum_cpu_core_frequency, median_cpu_core_temperature) = get_cpu_statistics();

    IoMeasurement {
        preprocessing_time: preprocessing_time,
        schema_learn_time: None,
        data_clone_time: data_clone_time,
        structure_build_time: structure_build_time,
        query_time: query_time,
        total_time: total_time,
        num_matches: num_matches,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        maximum_cpu_core_frequency: maximum_cpu_core_frequency,
        median_cpu_core_temperature: median_cpu_core_temperature,
    }
}

fn get_output_path(
    algorithm: &str,
    input_source: &str,
    pattern: &WorkloadPattern,
    cutoff: i32,
    num_digests: usize,
) -> PathBuf {
    let base_path = format!(
        "src{}_corpus{}_query{}_percent{}_cutoff{}",
        input_source,
        pattern.corpus_size,
        pattern.query_size,
        ((pattern.query_size as f64 / num_digests as f64) * 100.0).round() as i32,
        cutoff
    );

    PathBuf::from(OUTPUT_ROOT)
        .join(base_path)
        .join(algorithm)
        .join("report.csv")
}

fn write_workload_metrics_to_csv(
    algorithm: &str,
    input_source: &str,
    pattern: &WorkloadPattern,
    cutoff: i32,
    sample: usize,
    metrics: &IoMeasurement,
    is_first_open: bool,
    num_digests: usize,
) {
    let output_path = get_output_path(algorithm, input_source, pattern, cutoff, num_digests);
    create_dir_all(output_path.parent().unwrap()).expect("Failed to create output directory");

    let file = OpenOptions::new()
        .create(true)
        .append(!is_first_open)
        .truncate(is_first_open)
        .write(true)
        .open(&output_path)
        .expect("Couldn't open results CSV!");

    let mut writer = std::io::BufWriter::new(file);

    if is_first_open {
        writeln!(
            writer,
            "sample,timestamp_ns,preprocessing_ns,schema_learning_ns,clone_ns,build_ns,query_ns,total_ns,num_matches,query_to_corpus_ratio,max_cpu_frequency,median_cpu_temp"
        ).expect("Couldn't write headers!");
    }

    writeln!(
        writer,
        "{},{},{},{},{},{},{},{},{},{},{},{}",
        sample,
        metrics.timestamp,
        metrics.preprocessing_time.as_nanos(),
        metrics.schema_learn_time.map_or(0, |t| t.as_nanos()),
        metrics.data_clone_time.as_nanos(),
        metrics.structure_build_time.as_nanos(),
        metrics.query_time.as_nanos(),
        metrics.total_time.as_nanos(),
        metrics.num_matches,
        pattern.query_size as f64 / pattern.corpus_size as f64,
        metrics.maximum_cpu_core_frequency,
        metrics.median_cpu_core_temperature
    )
    .expect("Couldn't write data!");
}

fn compute_raw_digests(input: &InputSource) -> Vec<[u8; TLSH_DIGEST_LENGTH]> {
    let mut rng = StdRng::seed_from_u64(SEED);

    match &input {
        InputSource::Random { num_digests } => (0..*num_digests)
            .map(|_| generate_random_ascii_digest())
            .collect(),
        InputSource::RandomFixedSize { num_digests, .. } => (0..*num_digests)
            .map(|_| generate_random_ascii_digest())
            .collect(),
        InputSource::File { path } => {
            let mut digests = read_hashes_from_file(path);
            digests.shuffle(&mut rng);
            digests
        }
        InputSource::FileFixedSize { path, .. } => {
            let mut digests = read_hashes_from_file(path);
            digests.shuffle(&mut rng);
            digests
        }
    }
}

fn get_input_name(input: &InputSource) -> String {
    match &input {
        InputSource::Random { num_digests } => format!("random_{}", num_digests),
        InputSource::RandomFixedSize {
            fixed_size,
            num_digests,
        } => format!("random{}_fixed{}", num_digests, fixed_size),
        InputSource::File { .. } => "file".to_string(),
        InputSource::FileFixedSize { fixed_size, .. } => format!("file_fixed{}", fixed_size),
    }
}

fn run_regular_benchmarks(input: InputSource) {
    let rng = StdRng::seed_from_u64(SEED);
    let raw_digests = compute_raw_digests(&input);
    let num_digests = raw_digests.len();
    let fixed_size = match &input {
        InputSource::RandomFixedSize { fixed_size, .. } => Some(fixed_size),
        InputSource::FileFixedSize { fixed_size, .. } => Some(fixed_size),
        _ => None,
    };
    let input_name = get_input_name(&input);

    for percentage in (1..=10).chain((20..=100).step_by(10)).collect::<Vec<_>>() {
        let corpus_size = raw_digests.len() * percentage / 100;
        let query_size = *fixed_size.unwrap_or(&(corpus_size / 10));

        let pattern = WorkloadPattern {
            corpus_size,
            query_size,
        };

        let cutoff = 30;
        println!(
            "Running benchmarks for {}% of data (corpus: {}, query: {})",
            percentage, corpus_size, query_size
        );

        for algorithm in [
            "vp_tree",
            "trie",
            "linear_scan",
            "unwise_vp_tree",
            "unwise_vp_tree_original",
        ] {
            (0..NUM_SAMPLES)
                .collect::<Vec<_>>()
                .into_par_iter()
                .for_each(|sample| {
                    let metrics = match algorithm {
                        "vp_tree" => benchmark_workload_vp_tree(
                            &raw_digests,
                            &pattern,
                            cutoff,
                            &mut rng.clone(),
                        ),
                        "unwise_vp_tree" => benchmark_workload_unwise_vp_tree(
                            &raw_digests,
                            &pattern,
                            cutoff,
                            &mut rng.clone(),
                            430,
                        ),
                        "unwise_vp_tree_original" => benchmark_workload_unwise_vp_tree(
                            &raw_digests,
                            &pattern,
                            cutoff,
                            &mut rng.clone(),
                            20,
                        ),
                        "trie" => benchmark_workload_trie(
                            &raw_digests,
                            &pattern,
                            cutoff,
                            &mut rng.clone(),
                        ),
                        "linear_scan" => benchmark_workload_linear_scan(
                            &raw_digests,
                            &pattern,
                            cutoff,
                            &mut rng.clone(),
                        ),
                        _ => unreachable!(),
                    };
                    println!("Running {} sample {}", algorithm, sample);
                    write_workload_metrics_to_csv(
                        algorithm,
                        &input_name,
                        &pattern,
                        cutoff,
                        sample,
                        &metrics,
                        false,
                        num_digests,
                    );
                });
        }
    }
}

fn run_workload_benchmarks(input: InputSource) {
    let rng = StdRng::seed_from_u64(SEED);
    let raw_digests = compute_raw_digests(&input);
    let num_digests = raw_digests.len();
    let input_name = get_input_name(&input) + "_wl";

    WORKLOAD_PATTERNS.into_par_iter().for_each(|pattern| {
        let mut v = (1..=19)
            .chain(21..=29)
            .chain(31..=39)
            .chain((20..=100).step_by(10))
            .chain((100..=1000).step_by(50))
            .collect::<Vec<_>>();
        v.reverse();
        v.into_par_iter().for_each(|cutoff| {
            println!(
                "Running benchmarks for corpus: {}, query: {}, cutoff: {}",
                pattern.corpus_size, pattern.query_size, cutoff
            );

            for algorithm in [
                "trie",
                "linear_scan",
                "vp_tree",
                "unwise_vp_tree",
                "unwise_vp_tree_original",
            ] {
                for sample in 0..NUM_SAMPLES {
                    let metrics = match algorithm {
                        "vp_tree" => benchmark_workload_vp_tree(
                            &raw_digests,
                            pattern,
                            cutoff,
                            &mut rng.clone(),
                        ),
                        "unwise_vp_tree" => benchmark_workload_unwise_vp_tree(
                            &raw_digests,
                            pattern,
                            cutoff,
                            &mut rng.clone(),
                            430,
                        ),
                        "unwise_vp_tree_original" => benchmark_workload_unwise_vp_tree(
                            &raw_digests,
                            pattern,
                            cutoff,
                            &mut rng.clone(),
                            20,
                        ),
                        "trie" => {
                            benchmark_workload_trie(&raw_digests, pattern, cutoff, &mut rng.clone())
                        }
                        "linear_scan" => benchmark_workload_linear_scan(
                            &raw_digests,
                            pattern,
                            cutoff,
                            &mut rng.clone(),
                        ),
                        _ => unreachable!(),
                    };
                    println!("nd: {}", num_digests);
                    println!("Running {} sample {}", algorithm, sample);
                    write_workload_metrics_to_csv(
                        algorithm,
                        &input_name,
                        pattern,
                        cutoff,
                        sample,
                        &metrics,
                        false,
                        num_digests,
                    );
                }
            }
        });
    });
}

fn main() {
    println!("Running benchmarks");

    rayon::ThreadPoolBuilder::new()
        .num_threads(5)
        .build_global()
        .unwrap();

    run_regular_benchmarks(InputSource::Random {
        num_digests: 1_000_000,
    });

    run_regular_benchmarks(InputSource::RandomFixedSize {
        fixed_size: 1_000,
        num_digests: 1_000_000,
    });

    run_regular_benchmarks(InputSource::File {
        path: DEFAULT_INPUT_LIST_PATH,
    });

    run_regular_benchmarks(InputSource::FileFixedSize {
        fixed_size: 1_000,
        path: DEFAULT_INPUT_LIST_PATH,
    });

    run_workload_benchmarks(InputSource::Random {
        num_digests: 1_000_000,
    });

    run_workload_benchmarks(InputSource::File {
        path: DEFAULT_INPUT_LIST_PATH,
    });
}
