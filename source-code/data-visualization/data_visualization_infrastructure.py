from pathlib import Path

NS_TO_S_RATIO = 1e9


def find_project_root():
    current_dir = Path(__file__).resolve().parent

    while current_dir != current_dir.parent:
        paper_dependencies_path = current_dir / "paper-dependencies"

        if paper_dependencies_path.is_dir():
            return current_dir

        current_dir = current_dir.parent

    raise FileNotFoundError("Project root with paper-dependencies not found!")


def find_project_graphics_folder():
    return find_project_root() / "paper-dependencies" / "latex-dependencies" / "plots"


def find_project_data_visualization_dependencies_folder():
    return (
        find_project_root() / "paper-dependencies" / "data-visualization-dependencies"
    )
