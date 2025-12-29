from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Optional
import re
import math
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Iterable
from .config import Params
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

ColorLike = Union[str, List[float], Tuple[float, float, float]]

HEX_RE = re.compile(r'^#([0-9A-Fa-f]{6})$')
RGB_TXT_RE = re.compile(r'^\s*rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*$')
CSV_RE = re.compile(r'^\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*$')


# ---------------- internal helpers ----------------
def _to_hex_from_triplet(tri: Sequence[float]) -> str:
    if len(tri) != 3:
        raise ValueError("RGB triplet must have length 3.")
    tri = list(map(float, tri))
    if any(v > 1.0 for v in tri):
        r, g, b = [int(round(max(0, min(255, v)))) for v in tri]
    else:
        r, g, b = [int(round(max(0, min(1.0, v)) * 255)) for v in tri]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def _normalize_color(c: Optional[ColorLike], fallback: str) -> str:
    if c is None:
        return fallback
    if isinstance(c, (list, tuple)):
        return _to_hex_from_triplet(c)
    if isinstance(c, str):
        s = c.strip()
        if HEX_RE.match(s):
            return s
        m = RGB_TXT_RE.match(s) or CSV_RE.match(s)
        if m:
            tri = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
            return _to_hex_from_triplet(tri)
    return fallback


def _validate_lines(x_lines: Sequence[Sequence[float]],
                    y_lines: Sequence[Sequence[float]],
                    z_lines: Sequence[Sequence[float]]):
    if not (len(x_lines) == len(y_lines) == len(z_lines)):
        raise ValueError("Mismatched number of lines among x/y/z.")
    for i, (x, y, z) in enumerate(zip(x_lines, y_lines, z_lines)):
        if not (len(x) == len(y) == len(z)):
            raise ValueError(f"Line {i}: x/y/z length mismatch ({len(x)}/{len(y)}/{len(z)}).")
        if len(x) == 0:
            raise ValueError(f"Line {i} is empty.")


def plot3d_lines_to_html(
    x_lines,
    y_lines,
    z_lines,
    *,
    line_colors: Optional[Sequence[Optional[ColorLike]]] = None,
    line_labels: Optional[Sequence[str]] = None,
    out_html: Optional[Union[str, Path]] = None,
    title: str | None = None,
    axis_labels: dict[str, str] | None = None,
    segment_per_line: Optional[Sequence[Tuple[int, int, ColorLike]]] = None,
    segment_labels: Optional[Sequence[str]] = None,
    marker_count: int = 25,
    marker_size: int = 2,
    start_marker_size: int = 5,
    end_marker_size: int = 5,
    line_width: int = 4,
):
    """
    Plot 3D polylines to an HTML file. Supports *shared segmentation* coloring
    (the same set of segments is applied to every line).

    Parameters
    ----------
    x_lines, y_lines, z_lines : Sequence[Sequence[float]]
        Each item is one polyline (x/y/z must have the same 1D length per line).
    line_colors : Optional[Sequence[Optional[ColorLike]]]
        Base color for each line, used when no segments are given or for uncovered parts.
    line_labels : Optional[Sequence[str]]
        Display name for each line.
    out_html : Optional[Union[str, Path]]
        Output HTML path. If None, nothing is written to disk.
    title : Optional[str]
        Figure title.
    axis_labels : Optional[Sequence[str]]
        Axis labels (x, y, z).
    segment_per_line : Optional[Sequence[Tuple[int, int, ColorLike]]]
        Shared segments defined as (start, end, color); `end` is exclusive.
        If provided, the same segments are drawn for **every** line and each
        segment gets its own legend entry.
    segment_labels : Optional[Sequence[str]]
        Labels for segments, same length as `segment_per_line`, used in the legend.
    marker_count / marker_size : int
        Number and size of sampled white markers along each line (for reference only;
        not shown in the legend).
    start_marker_size / end_marker_size : int
        Marker sizes for the start/end points. Start = green diamond;
        End = black square.
    line_width : int
        Polyline width.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The created figure. If `out_html` is provided, the figure is also written to disk.
    """
    _validate_lines(x_lines, y_lines, z_lines)

    n_lines = len(x_lines)
    if line_labels is None:
        line_labels = [f"Line {i+1}" for i in range(n_lines)]
    if line_colors is None:
        line_colors = [None] * n_lines

    if not (len(line_labels) == len(line_colors) == n_lines):
        raise ValueError("line_labels and line_colors must have the same length as x_lines/y_lines/z_lines.")

    if segment_per_line is not None and segment_labels is not None:
        if len(segment_labels) != len(segment_per_line):
            raise ValueError("segment_labels and segment_per_line must have the same length.")

    default_color = "#808080"
    fig = go.Figure()

    for i, (x_, y_, z_, base_c, base_label) in enumerate(
        zip(x_lines, y_lines, z_lines, line_colors, line_labels)
    ):
        x = np.asarray(x_, dtype=float)
        y = np.asarray(y_, dtype=float)
        z = np.asarray(z_, dtype=float)
        T = x.shape[0]

        base_hex = _normalize_color(base_c, default_color)

        group_id = f"group_{i}"
        
        fig.update_layout(legend=dict(groupclick="togglegroup"))

        # ---------- Lines ----------
        if segment_per_line:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=line_width, color=base_hex),
                name=base_label,
                legendgroup=group_id,
                showlegend=True,
            ))
            for j, (s, e, c) in enumerate(segment_per_line):
                s = max(0, int(s)); e = min(T, int(e))
                if e <= s:
                    continue
                seg_hex = _normalize_color(c, base_hex)
                fig.add_trace(go.Scatter3d(
                    x=x[s:e], y=y[s:e], z=z[s:e],
                    mode="lines",
                    line=dict(width=line_width, color=seg_hex),
                    name=base_label,
                    legendgroup=group_id,
                    showlegend=False,
                ))
        else:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(width=line_width, color=base_hex),
                name=base_label,
                legendgroup=group_id,
                showlegend=True,
            ))

        mk = np.unique(np.round(np.linspace(0, T - 1, int(min(T, max(marker_count, 1)))))).astype(int)
        fig.add_trace(go.Scatter3d(
            x=x[mk], y=y[mk], z=z[mk],
            mode="markers",
            marker=dict(size=marker_size, color="white", line=dict(color=base_hex, width=1)),
            name=f"{base_label} marks",
            legendgroup=group_id,
            showlegend=False,
        ))

        # ---------- Start point ----------
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode="markers",
            marker=dict(
                size=start_marker_size,
                color="red",
                symbol="x",
                line=dict(color=base_hex, width=1),
            ),
            name=f"{base_label} start",
            legendgroup=group_id,
            showlegend=False,
        ))

        # ---------- End point ----------
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode="markers",
            marker=dict(
                size=end_marker_size,
                color="black",
                symbol="square",
                line=dict(color=base_hex, width=1),
            ),
            name=f"{base_label} end",
            legendgroup=group_id,
            showlegend=False,
        ))

    fig.update_layout(
        title= "Interactive 3D Trajectories" if title is None else title,
        scene= dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3") if axis_labels is None else axis_labels,
        template="plotly_white",
        margin=dict(l=0, r=0, t=48, b=0),
        legend=dict(itemsizing="trace"),
    )

    if out_html is not None:
        out_path = Path(out_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)

    return fig



def _validate_lines_2d(x_lines, y_lines):
    """Validate that x_lines and y_lines have compatible shapes."""
    if len(x_lines) != len(y_lines):
        raise ValueError("x_lines and y_lines must have the same number of lines.")
    for i, (x, y) in enumerate(zip(x_lines, y_lines)):
        if len(x) != len(y):
            raise ValueError(
                f"Line {i} has mismatched lengths: "
                f"len(x)={len(x)}, len(y)={len(y)}."
            )


def plot2d_lines_to_html(
    x_lines,
    y_lines,
    *,
    line_colors: Optional[Sequence[Optional["ColorLike"]]] = None,
    line_labels: Optional[Sequence[str]] = None,
    out_html: Optional[Union[str, Path]] = None,
    title: str | None = None,
    axis_labels: dict[str, str] | None = None,
    segment_per_line: Optional[Sequence[Tuple[int, int, "ColorLike"]]] = None,
    segment_labels: Optional[Sequence[str]] = None,
    marker_count: int = 25,
    marker_size: int = 2,
    start_marker_size: int = 5,
    end_marker_size: int = 5,
    line_width: int = 4,
):
    """
    Plot 2D polylines to an HTML file. This is the 2D analogue of
    plot3d_lines_to_html: each "line" is a polyline defined by x/y
    coordinates over time.

    Supports *shared segmentation* coloring (the same set of segments is
    applied to every line).

    Parameters
    ----------
    x_lines, y_lines : Sequence[Sequence[float]]
        Each item is one polyline (x/y must have the same 1D length per line).
    line_colors : Optional[Sequence[Optional[ColorLike]]]
        Base color for each line, used when no segments are given or for
        line parts not covered by segments.
    line_labels : Optional[Sequence[str]]
        Display name for each line.
    out_html : Optional[Union[str, Path]]
        Output HTML path. If None, nothing is written to disk.
    title : Optional[str]
        Figure title.
    axis_labels : Optional[dict[str, str]]
        Axis labels, e.g. {"x": "PC1", "y": "PC2"}. 
    segment_per_line : Optional[Sequence[Tuple[int, int, ColorLike]]]
        Shared segments defined as (start, end, color); `end` is exclusive.
        If provided, the same segments are drawn for **every** line.
    segment_labels : Optional[Sequence[str]]
        Labels for segments, same length as `segment_per_line`, used for
        bookkeeping; currently not attached to legend entries in this 2D
        version (matches the 3D function style).
    marker_count / marker_size : int
        Number and size of sampled white markers along each line (for
        reference only; not shown in the legend).
    start_marker_size / end_marker_size : int
        Marker sizes for the start/end points.
        Start = red "x"; End = black square.
    line_width : int
        Polyline width.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The created figure. If `out_html` is provided, the figure is also
        written to disk.
    """
    _validate_lines_2d(x_lines, y_lines)

    n_lines = len(x_lines)
    if line_labels is None:
        line_labels = [f"Line {i+1}" for i in range(n_lines)]
    if line_colors is None:
        line_colors = [None] * n_lines

    if not (len(line_labels) == len(line_colors) == n_lines):
        raise ValueError(
            "line_labels and line_colors must have the same length as "
            "x_lines and y_lines."
        )

    if segment_per_line is not None and segment_labels is not None:
        if len(segment_labels) != len(segment_per_line):
            raise ValueError(
                "segment_labels and segment_per_line must have the same length."
            )

    default_color = "#808080"
    fig = go.Figure()

    # Enable grouped legend clicking: clicking a line label toggles all
    # traces with the same legendgroup.
    fig.update_layout(legend=dict(groupclick="togglegroup"))

    for i, (x_, y_, base_c, base_label) in enumerate(
        zip(x_lines, y_lines, line_colors, line_labels)
    ):
        x = np.asarray(x_, dtype=float)
        y = np.asarray(y_, dtype=float)
        T = x.shape[0]

        if y.shape[0] != T:
            raise ValueError(f"Line {i} has mismatched x/y lengths.")

        base_hex = _normalize_color(base_c, default_color)
        group_id = f"group_{i}"

        # ---------- Base line and segments ----------
        if segment_per_line:
            # First draw the full line in the base color.
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=line_width, color=base_hex),
                    name=base_label,
                    legendgroup=group_id,
                    showlegend=True,
                )
            )
            # Then overlay each segment with its own color.
            for j, (s, e, c) in enumerate(segment_per_line):
                s = max(0, int(s))
                e = min(T, int(e))
                if e <= s:
                    continue
                seg_hex = _normalize_color(c, base_hex)
                fig.add_trace(
                    go.Scatter(
                        x=x[s:e],
                        y=y[s:e],
                        mode="lines",
                        line=dict(width=line_width, color=seg_hex),
                        name=base_label,
                        legendgroup=group_id,
                        showlegend=False,  # follow 3D version: segments don't get separate legend entries
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(width=line_width, color=base_hex),
                    name=base_label,
                    legendgroup=group_id,
                    showlegend=True,
                )
            )

        # ---------- Sampled markers along the line ----------
        mk = np.unique(
            np.round(
                np.linspace(0, T - 1, int(min(T, max(marker_count, 1))))
            )
        ).astype(int)

        fig.add_trace(
            go.Scatter(
                x=x[mk],
                y=y[mk],
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="white",
                    line=dict(color=base_hex, width=1),
                ),
                name=f"{base_label} marks",
                legendgroup=group_id,
                showlegend=False,
            )
        )

        # ---------- Start point ----------
        fig.add_trace(
            go.Scatter(
                x=[x[0]],
                y=[y[0]],
                mode="markers",
                marker=dict(
                    size=start_marker_size,
                    color="red",
                    symbol="x",
                    line=dict(color=base_hex, width=1),
                ),
                name=f"{base_label} start",
                legendgroup=group_id,
                showlegend=False,
            )
        )

        # ---------- End point ----------
        fig.add_trace(
            go.Scatter(
                x=[x[-1]],
                y=[y[-1]],
                mode="markers",
                marker=dict(
                    size=end_marker_size,
                    color="black",
                    symbol="square",
                    line=dict(color=base_hex, width=1),
                ),
                name=f"{base_label} end",
                legendgroup=group_id,
                showlegend=False,
            )
        )

    if axis_labels is None:
        axis_labels = {"x": "PC1", "y": "PC2"}

    fig.update_layout(
        title="Interactive 2D Trajectories" if title is None else title,
        xaxis_title=axis_labels.get("x", "X"),
        yaxis_title=axis_labels.get("y", "Y"),
        template="plotly_white",
        margin=dict(l=0, r=0, t=48, b=0), 
        legend=dict(itemsizing="trace"),
    )

    if out_html is not None:
        out_path = Path(out_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)

    return fig


def plot_neuron_id(
    data: Dict[str, Any],
    params: Params, 
    ana_tt: List[str],
    ana_bt: List[str],
    id: int,
):

    zones = data["aligned_zones_id"]

    indices: Iterable[int] = params.ana_index_grid(ana_tt, ana_bt)

    tracks: List[np.ndarray] = []
    labels: List[str] = []
    kept_indices: List[int] = []

    for idx in indices:
        fr = data["aligned_firing"][idx]
        if fr is None:
            continue

        neuron_fr = fr[id, :]

        tracks.append(neuron_fr)
        kept_indices.append(idx)

        label = f"{params.tt[idx[0]]} {params.bt[idx[1]]}"
        labels.append(label)

    n_plots = len(tracks)
    if n_plots == 0:
        return None
    
    cols = min(3, n_plots)
    rows = int(math.ceil(n_plots / cols))

    fig = plt.figure(figsize=(3 * cols + 1, rows * 2))
    for i, track in enumerate(tracks, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.plot(track)
        ax.set_title(labels[i - 1], fontname="Arial", fontsize=12)
        ax.set_xlabel("Position", fontname="Arial", fontsize=10)
        ax.set_ylabel("Firing rate", fontname="Arial", fontsize=10)
        ax.grid(True)

        for row in zones:
            if len(row) < 2:
                continue
            start_x, end_x = row[0], row[1]
            ax.axvline(float(start_x), linestyle="-", color="k", linewidth=0.8)
            ax.axvline(float(end_x), linestyle="-", color="k", linewidth=0.8)

    plt.tight_layout()
    plt.show()
