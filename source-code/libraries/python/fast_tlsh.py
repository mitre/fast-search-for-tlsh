from __future__ import annotations

import argparse
import hashlib
import io
import os
from typing import Annotated, NamedTuple

import numpy as np
import tensorflow as tf
from numba import njit
from numba.typed import List
from numpy.typing import NDArray

# The default cutoff to use for matches
DEFAULT_DISTANCE_CUTOFF = 200

# The length of a TLSH digest
TLSH_DIGEST_LENGTH = 70

# Type definitions
Corpus = Annotated[tf.Tensor, (None, TLSH_DIGEST_LENGTH)]
Query = Annotated[tf.Tensor, (TLSH_DIGEST_LENGTH,)]
Distances = Annotated[tf.Tensor, (None,)]
BatchDistances = Annotated[tf.Tensor, (None, None)]
Digest = Annotated[NDArray[np.uint8], (TLSH_DIGEST_LENGTH,)]
DecomposedHashArray = Annotated[NDArray[np.uint8], (None, TLSH_DIGEST_LENGTH)]


class SearchResult(NamedTuple):
    corpus_index: int
    query_index: int
    distance: int


# Maps ordinal versions of hex digits to their decimal equivalents.
# E.g., 'F' corresponds to 70 as a number in the ASCII table.
# The 70th element in hex_lookup_table is 15, which maps to the value
# of 0xF in hex as a number.
HEX_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0,
                    0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0],
                    dtype=np.int32)  # fmt: skip


# Disables GPU usage.
tf.config.set_visible_devices([], "GPU")


@tf.function(jit_compile=True)
def mod_diff(x: Distances, y: Distances, n: int) -> Distances:
    """Computes the modular difference / circular distance between pairwise
    members of tensors x and y modulo n.

    Derived from TLSH's `mod_diff` function.

    Parameters
    ----------
    x : tf.Tensor
        The first tensor, of shape (N,).
    y : tf.Tensor
        The second tensor, of shape (N,).
    n : int
        The modulus.

    Returns
    -------
    tf.Tensor
        A tensor, T, of shape (N,) where T[i] is the modular difference, or
        circular distance, between x[i] and y[i], modulo n.

    """
    delta = tf.abs(x - y)
    return tf.minimum(delta, n - delta)


# The TLSH distance function.
# Takes in a vector of decomposed hashes `corpus` and a decomposed TLSH hash `query`.
# Returns a vector `v` where `v[i] = D(corpus[i], query)`, where `D(X, Y)` is the TLSH distance between X and Y.
# Follows the structure of the TLSH codebase.
@tf.function(jit_compile=True)
def distance_tlsh(corpus: Corpus, query: Query) -> Distances:
    """Computes the TLSH distance between all members of a corpus, `corpus`, of
    decomposed hashes, and a single decomposed TLSH hash, `query`.

    Parameters
    ----------
    corpus : tf.Tensor
        A tensor of shape (N, M) where N is the number of hashes and M is the length of each decomposed hash.
    query : tf.Tensor
        A tensor of shape (M,) representing a single decomposed TLSH hash.

    Returns
    -------
    tf.Tensor
        A tensor, T, of shape (N,) where each element T[i] represents the TLSH distance between corpus[i]
        and the query hash.

    """
    # Header distance

    diff = tf.zeros(len(corpus), dtype=tf.int32)

    # Checksums
    diff += tf.where((corpus[:, 0] != query[0]) | (corpus[:, 1] != query[1]), 1, 0)

    # L values
    lvalue_tX = corpus[:, 2] + corpus[:, 3] * 16
    lvalue_tY = query[2] + query[3] * 16

    ldiff = mod_diff(lvalue_tX, lvalue_tY, 256)
    diff += tf.where(ldiff == 0, 0, tf.where(ldiff == 1, 1, ldiff * 12))

    # Q values
    q1_tX = corpus[:, 5]
    q1_tY = query[5]
    q2_tX = corpus[:, 4]
    q2_tY = query[4]

    q1diff = mod_diff(q1_tX, q1_tY, 16)
    diff += tf.where(q1diff <= 1, q1diff, (q1diff - 1) * 12)

    q2diff = mod_diff(q2_tX, q2_tY, 16)
    diff += tf.where(q2diff <= 1, q2diff, (q2diff - 1) * 12)

    # Body distance

    BODY_INDEX = 6  # Index at which point the TLSH "body" starts

    # The main loop, from TLSH, for body distance computation.
    for i in range(64):
        x = corpus[:, BODY_INDEX + i]
        y = query[BODY_INDEX + i]

        x1 = tf.bitwise.right_shift(x, 2)
        x2 = x & 0b11
        y1 = tf.bitwise.right_shift(y, 2)
        y2 = y & 0b11
        d1 = tf.abs(x1 - y1)
        d2 = tf.abs(x2 - y2)

        diff += tf.where(d1 == 3, 6, d1)
        diff += tf.where(d2 == 3, 6, d2)

    return diff


def recompose_hash(decomposed_hash: Digest) -> str:
    """Recompose a decomposed hash back into a string equivalent.

    Parameters
    ----------
    decomposed_hash : Digest
        An array of bytes representing the decomposed hash.

    Returns
    -------
    str
        The hex string representation of the hash.

    >>> [15, 15, 15]
    "fff"

    """
    return "".join(f"{x:x}" for x in decomposed_hash)


@njit
def decompose_hash(plaintext_hash: str) -> Digest:
    """Decomposes a hex string representation of a hash into a numpy array.

    Parameters
    ----------
    plaintext_hash : str
        A hex string representing the hash.

    Returns
    -------
    Digest
        A numpy array representing the decomposed hash.

    """
    byte_array = np.zeros(TLSH_DIGEST_LENGTH, dtype=np.uint8)

    for i in range(TLSH_DIGEST_LENGTH):
        byte_array[i] = HEX_LUT[ord(plaintext_hash[i])]

    return byte_array


def decompose_hashes(hashes: list[str]) -> np.ndarray:
    """Decomposes a list of hex string hashes into an array of Digests.

    Parameters.
    ----------
    hashes : np.ndarray
        A numpy array of hexadecimal strings representing the hashes.

    Returns
    -------
    np.ndarray
        An array of Digests representing the hashes.

    """
    assert len(hashes) != 0
    assert len(hashes[0]) == TLSH_DIGEST_LENGTH

    @njit(cache=True)
    def _decompose_hashes(hashes: List[str], decomposed_hashes: np.ndarray) -> None:
        for i in range(len(hashes)):
            plaintext_hash = hashes[i]
            for j in range(TLSH_DIGEST_LENGTH):
                decomposed_hashes[i, j] = HEX_LUT[ord(plaintext_hash[j])]

    decomposed_hashes = np.zeros((len(hashes), TLSH_DIGEST_LENGTH), dtype=np.uint8)
    _decompose_hashes(List(hashes), decomposed_hashes)

    return decomposed_hashes


def save_decomposed_hashes(hashes: DecomposedHashArray, output_path: str) -> None:
    """Saves decomposed hashes to a file.
    Used for "indexing" queries and corpuses to speed-up future searches, by
    allowing users to skip pre-processing the next time they use either a query
    list or a corpus. `output_path` is overwritten if it already exists.

    Parameters
    ----------
    hashes : DecomposedHashArray
        An array of decomposed hashes.
    output_path : str
        The path to write the decomposed hashes to.

    Returns
    -------
    None

    """
    np.save(output_path, hashes)


def load_decomposed_hashes(input_path: str) -> DecomposedHashArray:
    """Loads decomposed hashes from a file.

    Used for "indexing" queries and corpuses to speed-up future searches, by
    allowing users to skip pre-processing the next time they use either a query
    list or a corpus.

    Parameters
    ----------
    input_path : str
        The path to the decomposed hashes.

    Returns
    -------
    DecomposedHashArray
        An array of decomposed hashes from the file specified by input_path.

    """
    return np.load(input_path)


# PUBLIC API HERE:


@tf.function(jit_compile=True)
def get_distances(corpus: Corpus, query: Query) -> Distances:
    """Computes the TLSH distances between all members of a corpus and a query.

    Parameters
    ----------
    corpus : Corpus
        A tensor of shape (N, TLSH_DIGEST_LENGTH) where N is the number of hashes.
    query : Query
        A tensor of shape (TLSH_DIGEST_LENGTH,) representing a single decomposed TLSH hash.

    Returns
    -------
    tf.Tensor
        A tensor of shape (N,) where each element represents the TLSH distance between the query and
        each hash in the corpus.

    """
    return distance_tlsh(corpus, query)


@tf.function(jit_compile=True)
def batch_get_distances(
    corpus: Corpus,
    queries: Annotated[tf.Tensor, (None, TLSH_DIGEST_LENGTH)],
) -> BatchDistances:
    """Computes the TLSH distances between multiple queries and all members of a corpus.

    Parameters
    ----------
    corpus : Corpus
        A tensor of shape (N, TLSH_DIGEST_LENGTH) where N is the number of hashes.
    queries : Annotated[tf.Tensor, (None, TLSH_DIGEST_LENGTH)]
        A tensor of shape (M, TLSH_DIGEST_LENGTH) where M is the number of queries.

    Returns
    -------
    BatchDistance
        A tensor of shape (M, N) where each element represents the TLSH distance between each query and
        each hash in the corpus.

    """
    return tf.vectorized_map(lambda query: get_distances(corpus, query), queries)


def get_indices_within_distance(
    distances: NDArray[np.int32],
    distance_cutoff: int,
) -> NDArray[np.int32]:
    """Computes indices of hashes in the corpus within distance_cutoff of a query.

    Parameters
    ----------
    distances : NDArray[np.int32]
        An array of distances, where distances[i] is the TLSH distance between
        the i-th member of the corpus and the query.
    distance_cutoff : int
        The maximum distance beyond which results should be pruned.

    Returns
    -------
    NDArray[np.int32]
        An array of indices where the distances are within the specified distance.

    """
    return (distances <= distance_cutoff).nonzero()[0]


def batch_get_indices_within_distance(
    distances_arrays: NDArray[np.int32],
    distance_cutoff: int,
) -> list[NDArray[np.int32]]:
    """Computes indices of hashes in the corpus within a certain distance of a query, for each query.

    Parameters
    ----------
    distances_arrays : NDArray[np.int32]
        An array, A, of distances_arrays for each query, where A[i][v] is the distance between
        queries[i] and the v-th member of the corpus.
    distance_cutoff : int
        The maximum distance beyond which results should be pruned.

    Returns
    -------
    List[NDArray[np.int32]]
        A list, L, of arrays, where L[i] is a list of indices of members of the corpus
        within distance_cutoff of the i-th query.

    """
    return [
        get_indices_within_distance(distance, distance_cutoff)
        for distance in distances_arrays
    ]


def indices_to_results(
    query_index: int,
    corpus_indices: NDArray[np.int32],
    distances: NDArray[np.int32],
) -> list[SearchResult]:
    """Computes a SearchResult list using the index of a query within a query list and a
    list of the indices of the members of the corpus that it matched, as well as the
    associated distances between the queried hash and those members of the corpus.

    Parameters
    ----------
    query_index : int
        Index of the query within a query list.
    corpus_indices : NDArray[np.int32]
        The indices of hashes in the corpus that were close-enough matches.
    distances : NDArray[np.int32]
        An array, A, of distances for each query, where A[i] is the distance between
        queries[i] and the i-th member of the corpus.

    Returns
    -------
    List[SearchResult]
        A list of SearchResult named tuples.

    """
    results = []

    for i in corpus_indices:
        results.append(SearchResult(i, query_index, distances[i]))

    return results


def batch_indices_to_results(
    indices_of_matches_per_query: list[NDArray[np.int32]],
    distances_arrays: NDArray[np.int32],
) -> list[list[SearchResult]]:
    """Computes a list, `L`, of SearchResult lists, where `L[i]` corresponds to the SearchResult
    list for the `i`-th query against the corpus using distances from `distances_arrays[i]` and
    indices of matches from `indices_of_matches_per_query[i]`. See `indices_to_results` for
    more details.

    Parameters
    ----------
    indices_of_matches_per_query : List[NDArray[np.int32]]
        A list of arrays, each containing the indices for each query.
    distances_arrays : NDArray[np.int32]
        An array, A, of distances for each query, where A[i] is the distance between
        queries[i] and the i-th member of the corpus.

    Returns
    -------
    List[List[SearchResult]]
        A list, L, of SearchResult lists, where L[i] contains the list of
        SearchResults for the i-th query against corpus.

    """
    return [
        indices_to_results(query_index, corpus_indices, distances_arrays[query_index])
        for query_index, corpus_indices in enumerate(indices_of_matches_per_query)
    ]


def search(corpus: Corpus, query: Query, distance_cutoff: int) -> list[SearchResult]:
    """Performs a search across a corpus for matches (within distance_cutoff) of
    a single query.

    Provided as a convenience function abstracting over get_distances.
    Accordingly, performance will likely be worse than writing specialized
    code based on get_distances.

    Parameters
    ----------
    corpus : Corpus
        A tensor of shape (N, TLSH_DIGEST_LENGTH) where N is the number of hashes in the corpus.
    query : Query
        A tensor of shape (TLSH_DIGEST_LENGTH,) representing a single TLSH hash being queried against.
    distance_cutoff : int
        The maximum distance beyond which results should be pruned.

    Returns
    -------
    List[SearchResult]
        A list of SearchResult named tuples.

    """
    distances = get_distances(corpus, query).numpy()
    indices = get_indices_within_distance(distances, distance_cutoff)
    return indices_to_results(0, indices, distances)


def batch_search(
    corpus: Corpus,
    queries: Annotated[tf.Tensor, (None, TLSH_DIGEST_LENGTH)],
    distance_cutoff: int,
) -> list[list[SearchResult]]:
    """Performs a search across a corpus for matches (within distance_cutoff) of
    a list of queries.

    Provided as a convenience function abstracting over batch_get_distances.
    Accordingly, performance will likely be worse than writing specialized
    code based on batch_get_distances.

    Parameters
    ----------
    corpus : Corpus
        A tensor of shape (N, TLSH_DIGEST_LENGTH) where N is the number of hashes in the corpus.
    queries : tf.Tensor
        A tensor of shape (N, TLSH_DIGEST_LENGTH) where N is the number of queries.
    distance_cutoff : int
        The maximum distance beyond which results should be pruned.

    Returns
    -------
    List[List[SearchResult]]
        A list of lists of SearchResult named tuples where every SearchResult is within distance_cutoff
        of an associated query.

    """
    distances = batch_get_distances(corpus, queries).numpy()
    indices_of_matches_per_query = batch_get_indices_within_distance(
        distances,
        distance_cutoff,
    )
    return batch_indices_to_results(indices_of_matches_per_query, distances)


@tf.function(jit_compile=True)
def transform_hashes(byte_array: np.ndarray) -> tf.Tensor:
    """Private API for optimized batch hex digit -> ordinal conversion."""
    hashes = byte_array[:, :TLSH_DIGEST_LENGTH]

    hashes_tensor = tf.convert_to_tensor(hashes, dtype=tf.int32)
    lut_tensor = tf.convert_to_tensor(HEX_LUT, dtype=tf.int32)

    return tf.gather(lut_tensor, hashes_tensor)


def read_hashes_from_file(input_source: str | io.BytesIO) -> tf.Tensor:
    """Reads line-separated hashes from a file and returns a tensor.

    Parameters
    ----------
    input_source : str | io.BytesIO
        The input_source to the file containing the hashes, or an IO object for
        dependency injection.

    Returns
    -------
    tf.Tensor
        A tensor of decomposed hashes.

    """
    if isinstance(input_source, str):
        with open(input_source, "rb") as file:
            data = file.read()
    elif isinstance(input_source, io.BytesIO):
        data = input_source.read() + b"\n"
    else:
        msg = "Invalid type given to read_hashes_from_file!"
        raise
        raise Exception(msg)

    byte_array = (
        np.frombuffer(data, dtype=np.uint8)
        .astype(np.int32)
        .reshape(-1, TLSH_DIGEST_LENGTH + 1)
    )

    return transform_hashes(byte_array)


def get_file_content_hash(path: str) -> str:
    """Generates a hash of the input file's contents.

    Used to generate names for indices.

    Parameters
    ----------
    path : str
        The path of the file.

    Returns
    -------
    str
        The file's hash.

    """
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        while chunk := file.read(65536):
            hasher.update(chunk)

    return hasher.hexdigest()


def cli_run_query(
    corpus_file: str,
    queries_file: str,
    distance_cutoff: int,
    index_prefix: str,
    use_indices: bool,
    use_csv: bool,
) -> None:
    """Runs queries when using the command-line mode.

    Parameters
    ----------
    corpus_file: str
        Path to the corpus file containing hashes.
    queries_file: str
        Path to the queries file containing hashes.
    distance_cutoff: int
        Distance cutoff for matches.
    index_prefix: str
        Path and filename prefix for saving/loading decomposed hashes.
    use_indices: bool
        Flag to use stored decomposed hashes if they exist, or to save them if they don't.
    use_csv: bool
        Flag to use print results as comma-separated values.

    Returns
    -------
    None

    """
    # Large datasets can be processed in batches to preserve memory.
    BATCH_SIZE = 2**22

    corpus_hash_suffix = get_file_content_hash(corpus_file)
    queries_hash_suffix = get_file_content_hash(queries_file)
    corpus_index_file = f"{index_prefix}_corpus_{corpus_hash_suffix}"
    queries_index_file = f"{index_prefix}_queries_{queries_hash_suffix}"

    if use_csv:
        print("corpus_hash,query_hash,distance")

    if use_indices and os.path.isfile(corpus_index_file):
        corpus_tensor = tf.constant(
            load_decomposed_hashes(corpus_index_file),
            dtype=tf.int32,
        )
    else:
        corpus_tensor = read_hashes_from_file(corpus_file)
        if use_indices:
            save_decomposed_hashes(corpus_tensor.numpy(), corpus_index_file)

    if use_indices and os.path.isfile(queries_index_file):
        queries_tensor = tf.constant(
            load_decomposed_hashes(queries_index_file),
            dtype=tf.int32,
        )
    else:
        queries_tensor = read_hashes_from_file(queries_file)
        if use_indices:
            save_decomposed_hashes(queries_tensor.numpy(), queries_index_file)

    results = batch_search(corpus_tensor, queries_tensor, distance_cutoff)

    with open(corpus_file) as file:
        corpus_hashes = file.read().splitlines()

    with open(queries_file) as file:
        query_hashes = file.read().splitlines()

    dataset = tf.data.Dataset.from_tensor_slices(queries_tensor)
    dataset = dataset.batch(BATCH_SIZE)

    for batch_index, queries_batch in enumerate(dataset):
        results = batch_search(corpus_tensor, queries_batch, distance_cutoff)

        for i, query_results in enumerate(results):
            for corpus_index, query_index, distance in query_results:
                assert query_index == i
                actual_query_index = batch_index * BATCH_SIZE + query_index

                if use_csv:
                    print(
                        f"{corpus_hashes[corpus_index]},{query_hashes[actual_query_index]},{distance}",
                    )
                else:
                    print(
                        f"Corpus Match: {corpus_hashes[corpus_index]}, "
                        f"Query: {query_hashes[actual_query_index]}, "
                        f"Distance: {distance}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus_file",
        type=lambda x: (
            x if os.path.isfile(x) else parser.error(f"File {x} does not exist.")
        ),
        required=True,
        help="Path to the corpus file containing hashes.",
    )
    parser.add_argument(
        "--queries_file",
        type=lambda x: (
            x if os.path.isfile(x) else parser.error(f"File {x} does not exist.")
        ),
        required=True,
        help="Path to the queries file containing hashes.",
    )
    parser.add_argument(
        "--distance_cutoff",
        type=int,
        default=DEFAULT_DISTANCE_CUTOFF,
        help="Distance cutoff for matches.",
    )
    parser.add_argument(
        "--index_prefix",
        type=str,
        help="Path and filename prefix for saving/loading decomposed hashes.",
    )
    parser.add_argument(
        "--use_indices",
        action="store_true",
        help="Flag to use stored decomposed hashes if they exist, or to save them if they don't.",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="Flag to use print results as comma-separated values.",
    )

    args = parser.parse_args()

    if args.use_indices and not args.index_prefix:
        parser.error("--index_prefix is required when --use_indices is specified.")

    cli_run_query(
        corpus_file=args.corpus_file,
        queries_file=args.queries_file,
        distance_cutoff=args.distance_cutoff,
        index_prefix=args.index_prefix,
        use_indices=args.use_indices,
        use_csv=args.use_csv,
    )


if __name__ == "__main__":
    main()
