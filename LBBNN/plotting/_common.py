from __future__ import annotations
from pathlib import Path

def ensure_parent(save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

def get_matplotlib():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.colors import TwoSlopeNorm
    return plt, mcolors, TwoSlopeNorm

def get_graphviz_digraph():
    try:
        from graphviz import Digraph
        return Digraph
    except Exception as exc:
        raise ImportError("Graph plotting requires the optional dependency 'graphviz'. Install with: python -m pip install -e '.[plot]'") from exc
