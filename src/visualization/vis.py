from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt


def plot_series(x: Sequence[float], y: Sequence[float], title: str, xlabel: str, ylabel: str, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_dual_series(
    x1: Sequence[float], y1: Sequence[float], label1: str,
    x2: Sequence[float], y2: Sequence[float], label2: str,
    title: str, xlabel: str, ylabel: str, out_path: str | Path
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()