"""Shared publication-grade figure export helpers."""

from pathlib import Path


def save_static_figure(figure, path: Path) -> None:
    """Save a static figure as a tightly cropped 300-DPI image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")


def save_interactive_figure(figure, path: Path) -> None:
    """Save a self-contained Plotly HTML document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True)
