import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from data_visualization_infrastructure import (
    find_project_data_visualization_dependencies_folder,
    find_project_graphics_folder,
    NS_TO_S_RATIO,
)

OUTPUT_FOLDER = find_project_graphics_folder()

COLUMN_NAMES = [
    "sample",
    "timestamp_ns",
    "preprocessing_ns",
    "schema_learning_ns",
    "clone_ns",
    "build_ns",
    "query_ns",
    "total_ns",
    "num_matches",
    "query_to_corpus_ratio",
    "max_cpu_frequency",
    "median_cpu_temp",
]

BenchmarkParameters = namedtuple(
    "BenchmarkParameters",
    [
        "input_source",
        "num_digests",
        "workload",
        "fixed_size",
        "corpus_size",
        "query_size",
        "percent",
        "cutoff",
        "path",
    ],
)


plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{CharisSIL}",  # elsevier cas-dc font
        "font.family": "serif",
        "font.serif": [],
        "pgf.rcfonts": False,
        "figure.constrained_layout.use": True,
    }
)


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def parse_folder_name(name):
    parsed = {}
    parsed["is_fixed_workload"] = "wl" in name
    pattern = r"src(?P<input_source>[^_\d]+)(?:_?(?P<num_digests>\d+))?(?P<workload>_wl)?(?:_fixed(?P<fixed_size>\d+))?_corpus(?P<corpus_size>\d+)_query(?P<query_size>\d+)_percent(?P<percent>\d+)_cutoff(?P<cutoff>\d+)"
    match = re.match(pattern, name)
    if match:
        parameters = match.groupdict()
        return BenchmarkParameters(
            input_source=parameters["input_source"],
            num_digests=int(parameters["num_digests"])
            if parameters["num_digests"]
            else None,
            workload=bool(parameters["workload"]),
            fixed_size=int(parameters["fixed_size"])
            if parameters["fixed_size"]
            else None,
            corpus_size=int(parameters["corpus_size"]),
            query_size=int(parameters["query_size"]),
            percent=int(parameters["percent"]),
            cutoff=int(parameters["cutoff"]),
            path=name,
        )
    else:
        ignorable_names = {
            "rust-bench",
            "trie",
            "linear_scan",
            "vp_tree",
            "unwise_vp_tree_original",
            "unwise_vp_tree",
        }
        if name not in ignorable_names:
            print(f"Error on {name}")
    return None


def read_csv_files(path):
    data = {}
    for root, dirs, _ in os.walk(path):
        name = os.path.basename(root)
        metadata = parse_folder_name(name)
        if metadata:
            for dir in dirs:
                dirpath = os.path.join(root, dir)
                for file in os.listdir(dirpath):
                    if file.endswith(".csv"):
                        algorithm = dir

                        df = pd.read_csv(
                            os.path.join(dirpath, file),
                            names=COLUMN_NAMES,
                        )

                        try:
                            if df["sample"][0] == "sample":
                                df = df.drop(0)
                        except:
                            pass

                        for column in COLUMN_NAMES:
                            df[column] = pd.to_numeric(df[column], errors="coerce")

                        key = (
                            metadata,
                            algorithm,
                        )

                        data.setdefault(key, []).append(df)
    return data


def process_data(data):
    processed_data = {}

    for key, dfs in data.items():
        combined_df = pd.concat(dfs)
        combined_df.dropna(inplace=True)
        processed_data[key] = combined_df

    return processed_data


def plot_fits(xs, ys, xlabel, ylabel, output_path, log_x=False, log_y=False):
    linestyles = {
        "vp_tree": "-",
        "trie": "--",
        "trie_noschema": "-.",
        "linear_scan": ":",
        "unwise_vp_tree_original": (0, (3, 5, 1, 5, 1, 5)),
        "unwise_vp_tree": (0, (3, 1, 1, 1, 1, 1)),
    }

    plt.figure(layout="constrained")
    for algorithm, y in ys.items():
        x, y = zip(*sorted(zip(xs[algorithm], y)))

        unique_x = np.unique(x)

        median_y = [
            np.median([y_val for x_val, y_val in zip(x, y) if x_val == ux])
            for ux in unique_x
        ]

        # Intended for use in slides, which can be in color.
        # if algorithm == 'trie':
        #     plt.scatter(unique_x, list(map(lambda x: x / NS_TO_S_RATIO, median_y)), alpha=0.5, color="blue", s=10, label=None)
        # elif algorithm == 'vp_tree':
        #     plt.scatter(unique_x, list(map(lambda x: x / NS_TO_S_RATIO, median_y)), alpha=0.5, color="red", s=10, label=None)

        plt.scatter(
            unique_x,
            list(map(lambda x: x / NS_TO_S_RATIO, median_y)),
            alpha=0.15,
            color="grey",
            s=10,
            label=None,
        )

        unique_x_reshaped = np.array(unique_x).reshape(-1, 1)

        labels = {
            "vp_tree": "VP tree",
            "trie": "Trie",
            "trie_noschema": "Trie (excluding schema-learning)",
            "unwise_vp_tree": "Trend Micro (Corrected)",
            "unwise_vp_tree_original": "Trend Micro (Uncorrected)",
            "linear_scan": "Linear scan",
            "schema_learning_ns": "Schema learning time (s)",
            "build_ns": "Build time (s)",
            "query_ns": "Querying time (s)",
        }

        try:
            model = make_pipeline(
                PolynomialFeatures(degree=2),
                RANSACRegressor(residual_threshold=None),
            )
            model.fit(unique_x_reshaped, median_y)
            fit = model.predict(unique_x_reshaped) / NS_TO_S_RATIO
            plt.plot(
                unique_x,
                fit,
                linestyle=linestyles.get(algorithm, "-"),
                color="gray",
                label=f"{labels[algorithm]} (RANSAC, quadratic fit)",
            )
        except Exception as e:
            print(f"ransac fit, {algorithm}, {e}")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if log_x:
        plt.xscale("log")

    if log_y:
        plt.yscale("log")

    plt.legend()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def get_corpus_size(metadata):
    return metadata.corpus_size


def generate_line_plots(processed_data):
    line_plot_dir = os.path.join(OUTPUT_FOLDER, "line_plots")
    os.makedirs(line_plot_dir, exist_ok=True)

    for input_source in ["file", "random"]:
        log_options = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]

        for log_x, log_y in log_options:
            suffix = f"{'logx_' if log_x else ''}{'logy_' if log_y else ''}".strip("_")

            # average build (build + schema learning) time vs corpora size
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                algorithm = key[-1]
                if metadata.input_source == input_source and metadata.cutoff == 30:
                    xs[algorithm].extend([corpus_size] * len(df))
                    ys[algorithm].extend(df["build_ns"] + df["schema_learning_ns"])
                    if algorithm == "trie":
                        xs["trie_noschema"].extend([corpus_size] * len(df))
                        ys["trie_noschema"].extend(df["build_ns"])

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "average_build_and_schema_time",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Corpus Size (hashes)",
                "Average Build + Schema-Learning Time (s)",
                os.path.join(
                    line_plot_subdir,
                    f"average_build_and_schema_time_{suffix}.pdf",
                ),
                log_x=log_x,
                log_y=log_y,
            )

            # tree average time-per-query relative to corpus size at cutoff=30
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                algorithm = key[-1]
                query_size = metadata.query_size
                if metadata.input_source == input_source and metadata.cutoff == 30:
                    xs[algorithm].extend([corpus_size] * len(df))
                    ys[algorithm].extend(df["query_ns"] / query_size)

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "average_time_per_query",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Corpus Size (number of hashes)",
                "Average Time Per Query (s)",
                os.path.join(line_plot_subdir, f"average_time_per_query_{suffix}.pdf"),
                log_x=log_x,
                log_y=log_y,
            )

            # Fixed, query time vs. corpus size
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                algorithm = key[-1]
                # A query size of 1,000 is what was decided on when we ran the tests--there's not really another option
                if (
                    metadata.input_source == input_source
                    and metadata.cutoff == 30
                    and metadata.query_size == 1000
                ):
                    xs[algorithm].extend([corpus_size] * len(df))
                    ys[algorithm].extend(df["query_ns"])

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "query_time_fixed",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Corpus Size (number of hashes)",
                "Query Time (s)",
                os.path.join(line_plot_subdir, f"query_time_fixed_{suffix}.pdf"),
                log_x=log_x,
                log_y=log_y,
            )

            # Time to query 10% of the dataset
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                if metadata.input_source == input_source and metadata.cutoff == 30:
                    query_size = corpus_size // 10
                    if metadata.query_size == corpus_size // 10:
                        algorithm = key[-1]
                        xs[algorithm].extend([corpus_size] * len(df))
                        ys[algorithm].extend(df["query_ns"])

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "time_to_query_10_percent",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Corpus Size (number of hashes)",
                "Time to Query 10\\% (s)",
                os.path.join(
                    line_plot_subdir,
                    f"time_to_query_10_percent_{suffix}.pdf",
                ),
                log_x=log_x,
                log_y=log_y,
            )

            # Total time to query 10% of the dataset
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                algorithm = key[-1]
                if (
                    metadata.input_source == input_source
                    and metadata.cutoff == 30
                    and metadata.query_size == corpus_size // 10
                ):
                    xs[algorithm].extend([corpus_size] * len(df))
                    ys[algorithm].extend(
                        df["query_ns"] + df["build_ns"] + df["schema_learning_ns"],
                    )
                    if algorithm == "trie":
                        xs["trie_noschema"].extend([corpus_size] * len(df))
                        ys["trie_noschema"].extend(df["query_ns"] + df["build_ns"])

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "time_to_query_10_percent",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Corpus Size (number of hashes)",
                "Time to Query 10\\% (s)",
                os.path.join(
                    line_plot_subdir,
                    f"time_to_total_10_percent_{suffix}.pdf",
                ),
                log_x=log_x,
                log_y=log_y,
            )

            # Query time as a function of cutoff
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                cutoff = metadata.cutoff
                algorithm = key[-1]
                if (
                    metadata.input_source == input_source
                    and corpus_size == 1000000
                    and metadata.query_size == 1000
                ):
                    xs[algorithm].extend([cutoff] * len(df))
                    ys[algorithm].extend(df["query_ns"])

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "total_time_vs_cutoff",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Cutoff",
                "Query Time (s)",
                os.path.join(line_plot_subdir, f"query_time_vs_cutoff_{suffix}.pdf"),
                log_x=log_x,
                log_y=log_y,
            )

            # Total time as a function of cutoff
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                cutoff = metadata.cutoff
                algorithm = key[-1]
                if (
                    metadata.input_source == input_source
                    and corpus_size == 1000000
                    and metadata.query_size == 1000
                ):
                    xs[algorithm].extend([cutoff] * len(df))
                    ys[algorithm].extend(
                        df["build_ns"] + df["query_ns"] + df["schema_learning_ns"],
                    )
                    if algorithm == "trie":
                        xs["trie_noschema"].extend([cutoff] * len(df))
                        ys["trie_noschema"].extend(df["build_ns"] + df["query_ns"])

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "total_time_vs_cutoff",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Cutoff",
                "Total Time (s)",
                os.path.join(line_plot_subdir, f"total_time_vs_cutoff_{suffix}.pdf"),
                log_x=log_x,
                log_y=log_y,
            )

            # Build + schema learning time as a function of cutoff
            xs = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }
            ys = {
                "unwise_vp_tree": [],
                "unwise_vp_tree_original": [],
                "vp_tree": [],
                "trie": [],
                "linear_scan": [],
                "trie_noschema": [],
            }

            for key, df in processed_data.items():
                metadata = key[0]
                corpus_size = get_corpus_size(metadata)
                cutoff = metadata.cutoff
                algorithm = key[-1]
                if metadata.input_source == input_source and corpus_size == 1000000:
                    xs[algorithm].extend([cutoff] * len(df))
                    ys[algorithm].extend(df["build_ns"] + df["schema_learning_ns"])
                    if algorithm == "trie":
                        xs["trie_noschema"].extend([cutoff] * len(df))
                        ys["trie_noschema"].extend(df["build_ns"])

            line_plot_subdir = os.path.join(
                line_plot_dir,
                input_source,
                "build_and_schema_time_vs_cutoff",
            )
            os.makedirs(line_plot_subdir, exist_ok=True)
            plot_fits(
                xs,
                ys,
                "Cutoff",
                "Build + Schema-Learning Time (s)",
                os.path.join(
                    line_plot_subdir,
                    f"build_and_schema_time_vs_cutoff_{suffix}.pdf",
                ),
                log_x=log_x,
                log_y=log_y,
            )


data = read_csv_files(
    find_project_data_visualization_dependencies_folder() / "rust-bench",
)
processed_data = process_data(data)
generate_line_plots(processed_data)
