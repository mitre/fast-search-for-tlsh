import io
import random
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import tlsh

from fast_tlsh import *

# Seed RNG
np.random.seed(1337)


# Used for the report.
def find_paper_dependencies_root():
    current_dir = Path(__file__).resolve().parent

    while current_dir != current_dir.parent:
        print(current_dir.parent, current_dir)
        paper_dependencies_path = current_dir / "paper-dependencies"

        if paper_dependencies_path.is_dir():
            virusshare_data_path = (
                paper_dependencies_path
                / "benchmark-dependencies"
                / "virusshare-data.txt"
            )

            if virusshare_data_path.is_file():
                return paper_dependencies_path
            else:
                raise FileNotFoundError(
                    "virusshare-data.txt not found in paper-dependencies/benchmarking/!",
                )

        current_dir = current_dir.parent

    raise FileNotFoundError("Project root with paper-dependencies not found!")


# Generates random TLSH-like digests for testing, randomizing string capitalization.
def generate_random_digest():
    digest = "".join(
        np.random.choice(list("0123456789ABCDEF"), size=TLSH_DIGEST_LENGTH),
    )

    if np.random.choice([True, False]):
        return digest.lower()
    else:
        return digest.upper()


# Emulates the numpy-accelerated file-loader in fast_tlsh.py
def generate_random_digest_tlsh_fast():
    digest = "".join(
        np.random.choice(list(b"0123456789ABCDEF"), size=TLSH_DIGEST_LENGTH),
    )

    if np.random.choice([True, False]):
        return digest.lower()
    else:
        return digest.upper()


def test_hex_lut() -> None:
    control_lut = [0] * 128

    for i in range(10):
        control_lut[ord("0") + i] = i

    for i in range(6):
        control_lut[ord("a") + i] = 10 + i
        control_lut[ord("A") + i] = 10 + i

    for i, v in enumerate(control_lut):
        assert v == HEX_LUT[i], "HEX_LUT is incorrect!"


def test_decompose_hashes() -> None:
    NUMBER_OF_TESTS = 10000
    MAX_TEST_SIZE = 8

    for _ in range(NUMBER_OF_TESTS):
        num_hashes = np.random.randint(1, MAX_TEST_SIZE + 1)
        hashes = [generate_random_digest() for _ in range(num_hashes)]
        decomposed = decompose_hashes(hashes)

        assert decomposed.shape == (
            num_hashes,
            TLSH_DIGEST_LENGTH,
        ), "Unexpected shape for decomposed hash."


def test_decompose_and_recompose_hash() -> None:
    original_hash = generate_random_digest()
    decomposed = decompose_hash(original_hash)
    recomposed = recompose_hash(decomposed)

    assert (
        original_hash.lower() == recomposed.lower()
    ), "Recomposed hash does not match the original"


def test_save_and_load_decomposed_hashes(mocker) -> None:
    NUMBER_OF_TESTS = 1000
    MAX_TEST_SIZE = 8

    mocker.patch("numpy.save")

    for _ in range(NUMBER_OF_TESTS):
        num_hashes = np.random.randint(1, MAX_TEST_SIZE + 1)
        hashes = [generate_random_digest() for _ in range(num_hashes)]
        decomposed_hashes = decompose_hashes(hashes)

        mocker.patch("numpy.load", return_value=decomposed_hashes)

        save_decomposed_hashes(decomposed_hashes, "dummy_path.npy")
        loaded_hashes = load_decomposed_hashes("dummy_path.npy")

        assert np.array_equal(
            loaded_hashes,
            decomposed_hashes,
        ), "Loaded hashes differ from saved hashes."


def test_get_distances() -> None:
    corpus_hashes = [generate_random_digest() for _ in range(10)]
    query_hashes = corpus_hashes[:5] + [generate_random_digest() for _ in range(5)]
    decomposed_corpus = decompose_hashes(corpus_hashes)
    decomposed_queries = decompose_hashes(query_hashes)
    corpus_tensor = tf.constant(decomposed_corpus, dtype=tf.int32)

    for i, query_hash in enumerate(query_hashes):
        query_tensor = tf.constant(decomposed_queries[i], dtype=tf.int32)
        distances = get_distances(corpus_tensor, query_tensor).numpy()

        for j, corpus_hash in enumerate(corpus_hashes):
            expected_distance = tlsh.diff(corpus_hash, query_hash)
            assert (
                distances[j] == expected_distance
            ), f"Distance for query hash {i} and corpus hash {j} does not match expected value"


def test_batch_get_distances() -> None:
    def run_batch_get_distances_test(
        corpus_size,
        query_size,
        reuse_corpus=False,
    ) -> None:
        corpus_hashes = [generate_random_digest() for _ in range(corpus_size)]

        if reuse_corpus:
            if query_size == 1:
                query_hashes = [np.random.choice(corpus_hashes)]
            else:
                query_hashes = corpus_hashes[: query_size // 2] + [
                    generate_random_digest() for _ in range(query_size // 2)
                ]
        else:
            query_hashes = [generate_random_digest() for _ in range(query_size)]

        decomposed_corpus = decompose_hashes(corpus_hashes)
        decomposed_queries = decompose_hashes(query_hashes)

        corpus_tensor = tf.constant(decomposed_corpus, dtype=tf.int32)
        queries_tensor = tf.constant(decomposed_queries, dtype=tf.int32)

        distances = batch_get_distances(corpus_tensor, queries_tensor).numpy()

        for i, query_hash in enumerate(query_hashes):
            for j, corpus_hash in enumerate(corpus_hashes):
                expected_distance = tlsh.diff(corpus_hash, query_hash)
                assert (
                    distances[i][j] == expected_distance
                ), f"Distance for query {i} and corpus hash {j}, {distances[i][j]}, differs from expected value {expected_distance}"

    run_batch_get_distances_test(10, 10, reuse_corpus=True)
    run_batch_get_distances_test(100, 50, reuse_corpus=True)
    run_batch_get_distances_test(1000, 100, reuse_corpus=True)
    run_batch_get_distances_test(1, 1, reuse_corpus=True)
    run_batch_get_distances_test(1, 50, reuse_corpus=True)
    run_batch_get_distances_test(50, 1, reuse_corpus=True)
    run_batch_get_distances_test(10, 10, reuse_corpus=False)
    run_batch_get_distances_test(100, 50, reuse_corpus=False)
    run_batch_get_distances_test(1000, 100, reuse_corpus=False)
    run_batch_get_distances_test(1, 1, reuse_corpus=False)
    run_batch_get_distances_test(1, 50, reuse_corpus=False)
    run_batch_get_distances_test(50, 1, reuse_corpus=False)


def test_get_indices_within_distance() -> None:
    distances = np.array([100, 200, 300, 400])
    threshold = 250
    indices = get_indices_within_distance(distances, threshold)
    assert np.array_equal(indices, [0, 1]), "Unexpected indices within distance."


def test_batch_get_indices_within_distance() -> None:
    distances = np.array([[100, 200, 300], [400, 500, 600]])
    threshold = 450
    indices = batch_get_indices_within_distance(distances, threshold)
    assert len(indices) == 2, "Unexpected number of index arrays."
    assert np.array_equal(
        indices[0],
        [0, 1, 2],
    ), "Unexpected indices within distance for the first query."
    assert np.array_equal(
        indices[1],
        [0],
    ), "Unexpected indices within distance for the second query."


def test_indices_to_results() -> None:
    query_index = 0
    corpus_indices = np.array([0, 1])
    distances = np.array([100, 200])
    results = indices_to_results(query_index, corpus_indices, distances)
    assert len(results) == 2, "Unexpected number of results."
    assert results[0].corpus_index == 0, "Unexpected corpus index in result."
    assert results[0].query_index == 0, "Unexpected query index in result."
    assert results[0].distance == 100, "Unexpected distance in result."


def test_batch_indices_to_results() -> None:
    indices_of_matches_per_query = [np.array([0, 1]), np.array([0])]
    distances = np.array([[100, 200], [300, 400]])
    results = batch_indices_to_results(indices_of_matches_per_query, distances)
    assert len(results) == 2, "Unexpected number of result lists."
    assert len(results[0]) == 2, "Unexpected number of results for the first query."
    assert len(results[1]) == 1, "Unexpected number of results for the second query."


def test_distance_tlsh() -> None:
    NUMBER_OF_TESTS = 10000

    for _ in range(NUMBER_OF_TESTS):
        hash1 = generate_random_digest()
        hash2 = generate_random_digest()
        decomposed_hash1 = decompose_hash(hash1)
        decomposed_hash2 = decompose_hash(hash2)
        corpus_tensor = tf.constant([decomposed_hash1], dtype=tf.int32)
        query_tensor = tf.constant(decomposed_hash2, dtype=tf.int32)
        distance_1_to_2 = distance_tlsh(corpus_tensor, query_tensor).numpy()[0]
        distance_2_to_1 = distance_tlsh(
            tf.constant([decomposed_hash2], dtype=tf.int32),
            tf.constant(decomposed_hash1, dtype=tf.int32),
        ).numpy()[0]

        assert distance_1_to_2 == distance_2_to_1, "Commutativity violated"

        tlsh_distance = tlsh.diff(hash1, hash2)
        assert (
            distance_1_to_2 == tlsh_distance
        ), "Distance does not match TLSH library's distance"


def test_search() -> None:
    def run_search_test(corpus_size, distance_threshold, reuse_corpus=False):
        corpus_hashes = [generate_random_digest() for _ in range(corpus_size)]

        if reuse_corpus:
            query_hash = np.random.choice(corpus_hashes)
        else:
            query_hash = generate_random_digest()

        decomposed_corpus = decompose_hashes(corpus_hashes)
        decomposed_query = decompose_hash(query_hash)
        corpus_tensor = tf.constant(decomposed_corpus, dtype=tf.int32)
        query_tensor = tf.constant(decomposed_query, dtype=tf.int32)
        results = search(corpus_tensor, query_tensor, distance_threshold)

        for result in results:
            corpus_index = result.corpus_index
            distance = result.distance
            expected_distance = tlsh.diff(corpus_hashes[corpus_index], query_hash)
            assert (
                distance == expected_distance
            ), f"Distance for query and corpus hash {corpus_index} does not match expected value"
            assert (
                distance <= distance_threshold
            ), f"Distance {distance} exceeds threshold {distance_threshold}"

        return len(results)

    run_search_test(1000, 300, reuse_corpus=True)
    run_search_test(500, 300, reuse_corpus=True)
    run_search_test(100, 300, reuse_corpus=True)
    run_search_test(1, 300, reuse_corpus=True)
    run_search_test(10, 0, reuse_corpus=True)
    run_search_test(10, -1, reuse_corpus=True)
    run_search_test(1000, 300, reuse_corpus=False)
    run_search_test(500, 300, reuse_corpus=False)
    run_search_test(100, 300, reuse_corpus=False)
    run_search_test(1, 300, reuse_corpus=False)
    run_search_test(10, 0, reuse_corpus=False)
    run_search_test(10, -1, reuse_corpus=False)

    assert (
        run_search_test(10, -1, reuse_corpus=False) == 0
    ), "Search found matches, when there should've been none."
    assert (
        run_search_test(10, -1, reuse_corpus=True) == 0
    ), "Search found matches, when there should've been none."
    assert (
        run_search_test(10, np.iinfo(np.int32).max, reuse_corpus=True) == 10
    ), "Search found fewer than expected number of matches."
    assert (
        run_search_test(10, np.iinfo(np.int32).max, reuse_corpus=False) == 10
    ), "Search found fewer than expected number of matches."


def test_batch_search() -> None:
    def run_batch_search_test(
        corpus_size,
        query_size,
        distance_threshold,
        reuse_corpus=False,
    ):
        corpus_hashes = [generate_random_digest() for _ in range(corpus_size)]

        if reuse_corpus:
            if query_size == 1:
                query_hashes = [np.random.choice(corpus_hashes)]
            else:
                query_hashes = corpus_hashes[: query_size // 2] + [
                    generate_random_digest() for _ in range(query_size // 2)
                ]
        else:
            query_hashes = [generate_random_digest() for _ in range(query_size)]

        decomposed_corpus = decompose_hashes(corpus_hashes)
        decomposed_queries = decompose_hashes(query_hashes)

        corpus_tensor = tf.constant(decomposed_corpus, dtype=tf.int32)
        queries_tensor = tf.constant(decomposed_queries, dtype=tf.int32)

        results = batch_search(corpus_tensor, queries_tensor, distance_threshold)

        assert (
            len(results) == len(query_hashes)
        ), f"Results length does not match the number of queries for corpus size {corpus_size} and query size {query_size}"

        num_results = 0

        for _, query_results in enumerate(results):
            num_results += len(query_results)
            for result in query_results:
                corpus_index = result.corpus_index
                query_index = result.query_index
                distance = result.distance

                expected_distance = tlsh.diff(
                    corpus_hashes[corpus_index],
                    query_hashes[query_index],
                )
                assert (
                    distance == expected_distance
                ), f"Distance for query {query_index} and corpus hash {corpus_index} does not match expected value"
                assert (
                    distance <= distance_threshold
                ), f"Distance {distance} exceeds threshold {distance_threshold}"

        return num_results

    run_batch_search_test(1000, 100, 300, reuse_corpus=True)
    run_batch_search_test(500, 50, 300, reuse_corpus=True)
    run_batch_search_test(100, 10, 300, reuse_corpus=True)
    run_batch_search_test(1, 1, 300, reuse_corpus=True)
    run_batch_search_test(10, 5, 0, reuse_corpus=True)
    run_batch_search_test(10, 5, -1, reuse_corpus=True)
    run_batch_search_test(1000, 100, 300, reuse_corpus=False)
    run_batch_search_test(500, 50, 300, reuse_corpus=False)
    run_batch_search_test(100, 10, 300, reuse_corpus=False)
    run_batch_search_test(1, 1, 300, reuse_corpus=False)
    run_batch_search_test(10, 5, 0, reuse_corpus=False)
    run_batch_search_test(10, 5, -1, reuse_corpus=False)

    assert (
        run_batch_search_test(10, 5, -1, reuse_corpus=False) == 0
    ), "Batch search found matches, when there should've been none."
    assert (
        run_batch_search_test(10, 5, -1, reuse_corpus=True) == 0
    ), "Batch search found matches, when there should've been none."
    assert (
        run_batch_search_test(10, 10, 0, reuse_corpus=True) <= 5
    ), "Batch search found more matches than expected."
    assert (
        run_batch_search_test(10, 10, 0, reuse_corpus=True) >= 5
    ), "Batch search found fewer matches than expected."
    assert (
        run_batch_search_test(10, 4, np.iinfo(np.int32).max, reuse_corpus=True) == 40
    ), "Batch search found unexpected number of matches."
    assert (
        run_batch_search_test(10, 4, np.iinfo(np.int32).max, reuse_corpus=False) == 40
    ), "Batch search found unexpected number of matches."


def test_batch_get_distances_with_invalid_batch_size() -> None:
    corpus = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
    queries = tf.constant([[1, 2, 3]], dtype=tf.int32)
    with pytest.raises(ValueError):
        batch_get_distances(corpus, queries)


# Benchmark code


def generate_benchmark_dataset(corpus_size, queries_size):
    corpus_hashes = [generate_random_digest() for _ in range(corpus_size)]
    query_hashes = [generate_random_digest() for _ in range(queries_size)]
    return corpus_hashes, query_hashes


def generate_huge_dataset_tlsh(size):
    return "\n".join(generate_random_digest() for _ in range(size))


def generate_huge_dataset_tlsh_fast(size):
    return str.encode("\n".join(generate_random_digest() for _ in range(size)))


large_dataset_tlsh = generate_huge_dataset_tlsh(1_010_000)
large_dataset_tlsh_fast = generate_huge_dataset_tlsh_fast(1_010_000)

dataset_buffer_tlsh = io.StringIO(large_dataset_tlsh)
dataset_buffer_tlsh_fast = io.BytesIO(large_dataset_tlsh_fast)


def benchmark_tlsh(corpus_size, queries_size, use_fast_tlsh) -> None:
    if use_fast_tlsh:
        dataset_buffer_tlsh_fast.seek(0)

        tlsh_fast_hashes = read_hashes_from_file(dataset_buffer_tlsh_fast)
        corpus_tensor = tlsh_fast_hashes[:corpus_size]
        queries_tensor = tlsh_fast_hashes[corpus_size : corpus_size + queries_size]

        batch_get_distances(corpus_tensor, queries_tensor)
    else:
        dataset_buffer_tlsh.seek(0)
        tlsh_hashes = dataset_buffer_tlsh.read().splitlines()
        corpus_hashes = tlsh_hashes[:corpus_size]
        query_hashes = tlsh_hashes[corpus_size : corpus_size + queries_size]
        for query in query_hashes:
            for corpus in corpus_hashes:
                tlsh.diff(corpus, query)


def test_benchmark_tlsh_correctness() -> None:
    np.random.seed(1337)
    large_dataset_tlsh = generate_huge_dataset_tlsh(10_000)
    dataset_buffer_tlsh = io.StringIO(large_dataset_tlsh)

    np.random.seed(1337)
    large_dataset_tlsh_fast = generate_huge_dataset_tlsh_fast(10_000)
    dataset_buffer_tlsh_fast = io.BytesIO(large_dataset_tlsh_fast)

    corpus_size, queries_size = 1000, 1000

    dataset_buffer_tlsh.seek(0)
    tlsh_hashes = dataset_buffer_tlsh.read().splitlines()
    corpus_hashes = tlsh_hashes[:corpus_size]
    query_hashes = tlsh_hashes[corpus_size : corpus_size + queries_size]
    tlsh_results = []

    for query in query_hashes:
        for corpus in corpus_hashes:
            tlsh_results.append(tlsh.diff(corpus, query))

    dataset_buffer_tlsh_fast.seek(0)
    fast_tlsh_hashes = read_hashes_from_file(dataset_buffer_tlsh_fast)
    corpus_tensor = fast_tlsh_hashes[:corpus_size]
    queries_tensor = fast_tlsh_hashes[corpus_size : corpus_size + queries_size]
    fast_tlsh_results = (
        batch_get_distances(corpus_tensor, queries_tensor).numpy().flatten().tolist()
    )

    assert (
        tlsh_results == fast_tlsh_results
    ), "Unexpected difference between fast and regular TLSH :("


def test_performance_fast_tlsh_corpus_heavy(benchmark) -> None:
    corpus_hashes, query_hashes = (1_000_000, 1)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, True),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_tlsh_corpus_heavy(benchmark) -> None:
    corpus_hashes, query_hashes = (1_000_000, 1)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, False),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_fast_tlsh_query_heavy(benchmark) -> None:
    corpus_hashes, query_hashes = (1, 1_000_000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, True),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_tlsh_query_heavy(benchmark) -> None:
    corpus_hashes, query_hashes = (1, 1_000_000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, False),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_fast_tlsh_mixed_small(benchmark) -> None:
    corpus_hashes, query_hashes = (5000, 5000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, True),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_tlsh_mixed_small(benchmark) -> None:
    corpus_hashes, query_hashes = (5000, 5000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, False),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_fast_tlsh_mixed_large(benchmark) -> None:
    corpus_hashes, query_hashes = (10000, 10000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, True),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_tlsh_mixed_large(benchmark) -> None:
    corpus_hashes, query_hashes = (10000, 10000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, False),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_fast_tlsh_fixed_large(benchmark) -> None:
    corpus_hashes, query_hashes = (1_000_000, 1000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, True),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_tlsh_fixed_large(benchmark) -> None:
    corpus_hashes, query_hashes = (1_000_000, 1000)
    benchmark.pedantic(
        benchmark_tlsh,
        args=(corpus_hashes, query_hashes, False),
        iterations=1,
        rounds=11,
        warmup_rounds=1,
    )


def test_performance_tlsh_fast_realistic(benchmark) -> None:
    virusshare_data_path = (
        find_paper_dependencies_root()
        / "benchmark-dependencies"
        / "virusshare-data.txt"
    )
    corpus_tensor = read_hashes_from_file(str(virusshare_data_path.resolve()))

    corpus_tensor = tf.random.shuffle(corpus_tensor)[:1_000_000]
    queries_tensor = tf.random.shuffle(corpus_tensor)[:1_000]

    CORPUS_BATCH_SIZE = len(corpus_tensor)
    corpus_dataset = tf.data.Dataset.from_tensor_slices(corpus_tensor).batch(
        CORPUS_BATCH_SIZE,
    )
    QUERIES_BATCH_SIZE = len(queries_tensor)
    queries_dataset = tf.data.Dataset.from_tensor_slices(queries_tensor).batch(
        QUERIES_BATCH_SIZE,
    )

    # If you don't have enough RAM for the entire dataset, consider running something like this.
    # num_rows = tf.shape(corpus_tensor)[0]
    # queries_tensor = tf.random.shuffle(corpus_tensor)[:num_rows // 10]
    # CORPUS_BATCH_SIZE = 2 ** 22
    # corpus_dataset = tf.data.Dataset.from_tensor_slices(corpus_tensor).batch(CORPUS_BATCH_SIZE)
    # QUERIES_BATCH_SIZE = 2 ** 10
    # queries_dataset = tf.data.Dataset.from_tensor_slices(queries_tensor).batch(QUERIES_BATCH_SIZE)

    @benchmark
    def run_queries():
        for queries_batch in queries_dataset:
            for corpus_batch in corpus_dataset:
                batch_search(corpus_batch, queries_batch, 30)


def test_performance_tlsh_realistic(benchmark) -> None:
    virusshare_data_path = (
        find_paper_dependencies_root()
        / "benchmark-dependencies"
        / "virusshare-data.txt"
    )

    with virusshare_data_path.open() as file:
        corpus_hashes = file.read().splitlines()

    random.shuffle(corpus_hashes)
    corpus_hashes = corpus_hashes[:1_000_000]
    random.shuffle(corpus_hashes)
    query_hashes = corpus_hashes[:1_000]
    random.shuffle(corpus_hashes)

    @benchmark
    def run_query():
        for query in query_hashes:
            for corpus in corpus_hashes:
                tlsh.diff(corpus, query)
