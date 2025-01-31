import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from data_visualization_infrastructure import (
    NS_TO_S_RATIO,
    find_project_data_visualization_dependencies_folder,
    find_project_graphics_folder,
)

OUTPUT_FOLDER = find_project_graphics_folder() / "consolidated_bar_charts"

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

METRIC_STYLES = {
    "schema_learning_ns": {"shade": 0.7, "hatch": "o"},
    "build_ns": {"shade": 0.7, "hatch": "+"},
    "query_ns": {"shade": 0.7, "hatch": "x"},
}


plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": "\\usepackage{CharisSIL}",  # elsevier cas-dc font
        "font.family": "serif",
        "font.serif": [],
        "pgf.rcfonts": False,
        "figure.constrained_layout.use": True,
    },
)


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def parse_folder_name(name):
    # We only care about the very specific workloads.
    if "wl" not in name:
        return None
    pattern = r"src(?P<input_source>[^_\d]+)(?:_?(?P<num_digests>\d+))?(?P<workload>_wl)?(?:_fixed(?P<fixed_size>\d+))?_corpus(?P<corpus_size>\d+)_query(?P<query_size>\d+)_percent(?P<percent>\d+)_cutoff(?P<cutoff>\d+)"
    match = re.match(pattern, name)
    if match:
        return match.groupdict()
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
                            metadata["input_source"],
                            metadata["corpus_size"],
                            metadata["query_size"],
                            metadata["cutoff"],
                            algorithm,
                        )

                        data.setdefault(key, []).append(df)
    return data


def process_data(data):
    processed_data = {}

    for k, v in data.items():
        concatted = pd.concat(v)

        concatted.dropna(inplace=True)

        for column in COLUMN_NAMES[1:8]:
            # Yes, I Googled the conversion ratio.
            concatted[column] /= NS_TO_S_RATIO

        medians = concatted.median()
        std_errors = concatted.sem()
        processed_data[k] = (medians, std_errors)

    return processed_data


def plot_consolidated_bar_chart(
    processed_data,
    corpus_size,
    query_size,
    cutoff,
    output_path,
):
    algorithms = ["vp_tree", "trie", "linear_scan"]
    metrics = ["schema_learning_ns", "build_ns", "query_ns"]
    input_sources = ["random", "file"]

    input_colors = {
        "random": "darkgray",
        "file": "dimgray",
    }

    _, ax = plt.subplots(layout="constrained")
    index = np.arange(len(algorithms))
    width = 0.35 / len(input_sources)
    space = 0.05

    for i, algorithm in enumerate(algorithms):
        for j, input_source in enumerate(input_sources):
            key = (input_source, corpus_size, query_size, cutoff, algorithm)
            medians, _ = processed_data.get(key, (None, None))

            if medians is not None:
                bottom = 0
                for metric in metrics:
                    style = METRIC_STYLES[metric]
                    ax.bar(
                        index[i] + j * (width + space),
                        medians[metric],
                        width,
                        bottom=bottom,
                        color=input_colors[input_source],
                        hatch=style["hatch"],
                        alpha=1,
                    )
                    bottom += medians[metric]

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Time (s)")
    ax.set_xticks(index + (width + space) * (len(input_sources) - 1) / 2)
    ax.set_xticklabels([labels[alg] for alg in algorithms])
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    legend_elements = [
        Patch(facecolor=color, edgecolor="black", label=labels[input_source])
        for input_source, color in input_colors.items()
    ]

    hatch_legend_elements = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=METRIC_STYLES[metric]["hatch"],
            label=labels[metric],
        )
        for metric in metrics
    ]

    ax.legend(
        handles=legend_elements + hatch_legend_elements,
        title="Legend",
        title_fontsize=14,
        prop={"size": 14},
        loc="upper left",
    )

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


labels = {
    "vp_tree": "VP tree",
    "trie": "Trie",
    "linear_scan": "Linear scan",
    "schema_learning_ns": "Schema-learning time (s)",
    "random": "Randomly generated data",
    "file": "VirusShare-derived data",
    "build_ns": "Build time (s)",
    "query_ns": "Querying time (s)",
}


def main():
    root = find_project_data_visualization_dependencies_folder() / "rust-bench"
    data = read_csv_files(root)
    processed_data = process_data(data)

    WORKLOAD_PATTERNS = [
        ("1000000", "10"),
        ("10", "1000000"),
        ("10000", "10000"),
        ("5000", "5000"),
        ("1000000", "1000"),
    ]

    for key in tqdm(processed_data.keys(), desc="Plotting"):
        _, corpus_size, query_size, cutoff, _ = key

        if cutoff != "30":
            continue

        if (corpus_size, query_size) in WORKLOAD_PATTERNS:
            export_dir = os.path.join(
                OUTPUT_FOLDER,
                f"corpus_{corpus_size}",
                f"query_{query_size}",
                f"cutoff_{cutoff}",
            )

            consolidated_dir = os.path.join(export_dir, "consolidated")
            os.makedirs(consolidated_dir, exist_ok=True)

            plot_consolidated_bar_chart(
                processed_data,
                corpus_size,
                query_size,
                cutoff,
                os.path.join(
                    consolidated_dir,
                    f"consolidated_bar_chart_query{query_size}_corpus{corpus_size}.pdf",
                ),
            )


if __name__ == "__main__":
    main()
