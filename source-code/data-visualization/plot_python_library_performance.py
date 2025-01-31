import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_visualization_infrastructure import (
    NS_TO_S_RATIO,
    find_project_data_visualization_dependencies_folder,
    find_project_graphics_folder,
)

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


def find_rust_csv_file(search_string, query_number, corpus_number):
    benchmarks = find_project_data_visualization_dependencies_folder() / "rust-bench"

    for name in os.listdir(benchmarks):
        if name == ".DS_Store":
            continue
        if query_number != 1000 and "fixed1000" in name:
            continue
        if search_string not in name:
            continue
        if f"query{query_number}_" not in name:
            continue
        if f"corpus{corpus_number}_" not in name:
            continue
        if name[-8:] != "cutoff30":
            continue

        path = benchmarks / name / "linear_scan" / "report.csv"

        return path


x_labels = [
    "Random,\n1,000,000,\n1",
    "Random,\n1,\n1,000,000",
    "Random,\n10,000,\n10,000",
    "Random,\n5,000,\n5,000",
    "Random,\n1,000,000,\n1,000",
    "VirusShare,\n1,000,000,\n1,000",
]
num_custom_bars_added = 0
new_bar_positions = []
bars_per_category = [2] * len(x_labels)


def add_custom_bar(
    ax,
    x,
    category_index,
    value,
    width,
    label,
    color,
    hatch=None,
    capsize=5,
):
    global x_labels
    global num_custom_bars_added
    global bars_per_category

    current_bars = bars_per_category[category_index]
    new_bar_position = x[category_index] + (current_bars - 1) * width / 2

    ax.bar(
        new_bar_position + (width),
        value,
        width,
        label=label if num_custom_bars_added == 0 else "",
        color=color,
        hatch=hatch,
        capsize=capsize,
        alpha=1,
    )

    bars_per_category[category_index] += 1

    tick_positions = [
        x[i] + (bars_per_category[i] - 1) * width / 2 - (width / 2)
        for i in range(len(x))
    ]

    ax.set_xticks(tick_positions)

    ax.set_xticklabels(x_labels)

    num_custom_bars_added += 1


def get_median_from_csv(csv_file_path):
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

    df = pd.read_csv(csv_file_path, names=COLUMN_NAMES, header=None)
    df["query_ns"] = pd.to_numeric(df["query_ns"], errors="coerce")

    return df["query_ns"].median() / NS_TO_S_RATIO


with (
    find_project_data_visualization_dependencies_folder() / "python-benchmark-data.txt"
).open() as f:
    lines = list(
        map(
            str.strip,
            filter(lambda s: s.startswith("test_performance"), f.readlines()),
        ),
    )

parsed_data = {}

for line in lines:
    parts = line.split()
    name = parts[0]
    median = float(parts[9].replace(",", "")) / 1000
    parsed_data[name] = median

tlsh_fast_medians = [
    parsed_data["test_performance_fast_tlsh_corpus_heavy"],
    parsed_data["test_performance_fast_tlsh_query_heavy"],
    parsed_data["test_performance_fast_tlsh_mixed_small"],
    parsed_data["test_performance_fast_tlsh_mixed_large"],
    parsed_data["test_performance_fast_tlsh_fixed_large"],
    parsed_data["test_performance_tlsh_fast_realistic"],
]
py_tlsh_medians = [
    parsed_data["test_performance_tlsh_corpus_heavy"],
    parsed_data["test_performance_tlsh_query_heavy"],
    parsed_data["test_performance_tlsh_mixed_small"],
    parsed_data["test_performance_tlsh_mixed_large"],
    parsed_data["test_performance_tlsh_fixed_large"],
    parsed_data["test_performance_tlsh_realistic"],
]

width = 0.2
x = list(np.arange(len(x_labels)))

fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
ax.bar(
    np.array(x) - width / 2,
    tlsh_fast_medians,
    width,
    label="tlsh-fast",
    color="dimgray",
    capsize=5,
    alpha=1,
)
ax.bar(
    np.array(x) + width / 2,
    py_tlsh_medians,
    width,
    label="py-tlsh",
    color="darkgray",
    capsize=5,
    alpha=1,
)

# The /10 is because of it does 10x as many queries.
add_custom_bar(
    ax,
    x,
    0,
    get_median_from_csv(find_rust_csv_file("file", 10, 1000000)) / 10,
    width,
    "rust-linear-scan",
    "lightgray",
    hatch="x",
)
add_custom_bar(
    ax,
    x,
    1,
    get_median_from_csv(
        find_rust_csv_file(
            "file",
            1000000,
            10,
        ),
    )
    / 10,
    width,
    "rust-linear-scan",
    "lightgray",
    hatch="x",
)
add_custom_bar(
    ax,
    x,
    2,
    get_median_from_csv(find_rust_csv_file("random", 10000, 10000)),
    width,
    "rust-linear-scan",
    "lightgray",
    hatch="x",
)
add_custom_bar(
    ax,
    x,
    3,
    get_median_from_csv(find_rust_csv_file("random", 5000, 5000)),
    width,
    "rust-linear-scan",
    "lightgray",
    hatch="x",
)
add_custom_bar(
    ax,
    x,
    4,
    get_median_from_csv(find_rust_csv_file("random", 1000, 1000000)),
    width,
    "rust-linear-scan",
    "lightgray",
    hatch="x",
)
add_custom_bar(
    ax,
    x,
    5,
    get_median_from_csv(find_rust_csv_file("file", 1000, 1000000)),
    width,
    "rust-linear-scan",
    "lightgray",
    hatch="x",
)

ax.set_xticklabels(x_labels)
ax.set_xlabel("Source, corpus size, number of queries")
ax.set_ylabel("Time (s)")
ax.legend(title="Legend", title_fontsize=14, prop={"size": 14})

plt.yscale("log")
plt.savefig(
    find_project_graphics_folder() / "python_comparative_chart.pdf", bbox_inches="tight",
)
