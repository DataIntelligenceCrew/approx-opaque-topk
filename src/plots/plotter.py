import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Mapping, Optional, Sequence, Union, Tuple
import re
from pathlib import Path


# A mapping from method name to its label, color, and line-style.
# These mappings are kept consistent across experiments by setting this as a global variable.
METHODS = {
    "ScanBestOrder": {
        "name": "Scan (Best Case)",
        "color": "#BBBBBB",  # black
        "style": "solid",
        "index": None
    },
    "ScanWorstOrder": {
        "name": "Scan (Worst Case)",
        "color": "#000000",  # white
        "style": "solid",
        "index": None
    },
    "UniformSample": {
        "name": "UniformSample",
        "color": "#EE3377",  # magenta
        "style": "solid",
        "index": None
    },
    "SortedIndexScan": {
        "name": "SortedIndexScan",
        "color": "#8c3756",  # orange
        "style": "solid",
        "index": "sorted"
    },
    "UCB": {
        "name": "UCB",
        "color": "#EE7733",  # magenta
        "style": "solid",
        "index": "dendrogram"
    },
    "UniformExploration": {
        "name": "ExplorationOnly",
        "color": "#CC3311",  # red
        "style": "solid",
        "index": "dendrogram"
    },
    "EpsGreedy": {
        "name": "Ours",
        "color": "#0077BB",  # blue
        "style": "solid",
        "index": "dendrogram"
    },
    "EpsGreedy (No Rebinning)": {
        "name": "Ours (No Re-binning)",
        "color": "#332288",  # indigo
        "style": "dashed",
        "index": "dendrogram"
    },
    "EpsGreedy (No Subtraction)": {
        "name": "Ours (No Subtraction)",
        "color": "#8877EE",  # cyan
        "style": "dashed",
        "index": "dendrogram"
    },
    "EpsGreedy (No Fallback)": {
        "name": "Ours (No Fallback)",
        "color": "#44AA99",  # teal
        "style": "dashed",
        "index": "dendrogram"
    },
    "EpsGreedy (B=4)": {
        "name": "Ours (B=4)",
        "color": "#FEDA8B", # yellow
        "style": "dotted",
        "index": "dendrogram"
    },
    "EpsGreedy (B=16)": {
        "name": "Ours (B=16)",
        "color": "#FDB366",  # orange
        "style": "dotted",
        "index": "dendrogram"
    },
    "EpsGreedy (B=32)": {
        "name": "Ours (B=32)",
        "color": "#F67E4B",  # dark orange
        "style": "dotted",
        "index": "dendrogram"
    },
    "EpsGreedy (F=5%)": {
        "name": "F=0.05",
        "color": "#E7D4E8",  # light purple
        "style": "dashdot",
        "index": "dendrogram"
    },
    "EpsGreedy (F=10%)": {
        "name": "F=0.1",
        "color": "#C2A5CF",  # purple
        "style": "dashdot",
        "index": "dendrogram"
    },
    "EpsGreedy (F=25%)": {
        "name": "F=0.25",
        "color": "#9970AB",  # dark purple
        "style": "dashdot",
        "index": "dendrogram"
    },
    "EpsGreedy (25%)": {
        "name": "F=0.25",
        "color": "#9970AB",  # dark purple
        "style": "dashdot",
        "index": "dendrogram"
    },
    "EpsGreedy (Batch=25)": {
        "name": "Batch=25",
        "color": "#364B94",
        "style": "dotted",
        "index": "dendrogram"
    },
    "EpsGreedy (Batch=50)": {
        "name": "Batch=50",
        "color": "#4A7BB7",
        "style": "dotted",
        "index": "dendrogram"
    },
    "EpsGreedy (Batch=100)": {
        "name": "Batch=100",
        "color": "#6EA6CD",
        "style": "dotted",
        "index": "dendrogram"
    },
    "EpsGreedy (Batch=200)": {
        "name": "Batch=200",
        "color": "#98CAE1",
        "style": "dotted",
        "index": "dendrogram"
    },
    "EpsGreedy (Batch=800)": {
        "name": "Batch=800",
        "color": "#022963",
        "style": "dotted",
        "index": "dendrogram"
    }
}


def plot_metric_per_time_or_iter(result_stats: Dict,
                                 order: Union[str, List[str], None],
                                 metric: str,
                                 x_axis: str,
                                 y_log: bool = False,
                                 x_log: bool = False,
                                 ylabel: Union[None, str] = None,
                                 xlabel: Union[None, str] = None,
                                 title: Union[None, str] = None,
                                 filename: Union[None, str] = None,
                                 xrange: Union[None, Tuple[float, float]] = None,
                                 yrange: Union[None, Tuple[float, float]] = None,
                                 linewidth: float = 1,
                                 place_legend_outside: bool = False,
                                 legend_loc: str = "best",
                                 fontsize: float = 16):
    """
    Plots specified metric(s) over time or iterations given result statistics.

    Parameters
    result_stats: A dictionary mapping from config name to the result statistics for the config.
    metric: The metric to use for the y-axis. One of 'STK', 'KLS', 'Precision@K', 'Recall@K', 'AvgRank' or 'WorstRank'.
    x_axis: The metric to use for the x-axis. Either 'sec', 'hour', or 'iteration'.
    y_log: Whether the y-axis should be logarithmically scaled.
    x_log: Whether the x-axis should be logarithmically scaled.
    ylabel: The label for the y-axis.
    xlabel: The label for the x-axis.
    title: The title of the plot.
    filename: The file that this plot should be saved to.
    xrange: The range to limit the x-axis to.
    yrange: The range to limit the y-axis to.
    linewidth: The width of the lines in the plot.
    """
    plt.figure(figsize=(16, 10))

    if order is None:
        order = global_order
    if isinstance(order, str):
        order = method_collections[order]

    # Plot each config specified in order
    for config_name in order:
        # Obtain the stats for this config
        if config_name in result_stats and config_name in METHODS:
            stats: Dict = result_stats[config_name]

            # Obtain the x values
            if x_axis == 'sec' or x_axis == 'hour':
                x_values: List[float] = stats['time']
                if x_axis == 'hour':  # The 'time' field logs time incurred in seconds
                    x_values = np.array(x_values) / 3600.0
            elif x_axis == 'iteration':
                x_values = list(range(1, len(stats[f'{metric}']) + 1))
            else:
                raise ValueError

            # Obtain the y values
            y_values = [stats[f'{metric}'][i] for i in range(0, len(stats[f'{metric}']))]

            # Create plot
            plt.plot(x_values, y_values,
                     label=METHODS[config_name]['name'],
                     color=METHODS[config_name]['color'],
                     linestyle=METHODS[config_name]['style'],
                     linewidth=linewidth)

    # Label the x-axis
    if x_axis == 'sec':
        xlabel = 'Time (s)'
    elif x_axis == 'hour':
        xlabel = 'Time (hr)'
    elif x_axis == 'iteration':
        xlabel = 'Iteration'
    else:
        raise ValueError
    plt.xlabel(xlabel, fontsize=fontsize)

    # Label the y-axis
    if not ylabel:
        plt.ylabel(f'{metric} Value', fontsize=fontsize)
    else:
        plt.ylabel(ylabel, fontsize=fontsize)

    # Limit axis values optionally
    if xrange is not None:
        plt.xlim(xrange)
    if yrange is not None:
        plt.ylim(yrange)

    # Title, legend, grid
    if title is not None:
        plt.title(title)
    if place_legend_outside:
        plt.legend(loc=legend_loc, fontsize=fontsize, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    else:
        plt.legend(loc=legend_loc, fontsize=14)
    plt.grid(visible=True, color="#f0f0f0")

    # Log scale the axis optionally
    if y_log:
        plt.yscale('log')
    if x_log:
        plt.xscale('log')

    # Save plot to file optionally
    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")

    plt.show()


TIME_UNITS = {
    "s": 1,
    "ms": 1e3,
    "μs": 1e6,
    "ns": 1e9,
    "hr": 1 / 3600
}


def _safe_float_from_txt(path: Path) -> float:
    """
    Read a single-line TXT file and parse a float value (seconds).
    If file missing or malformed, returns 0.0 and prints a warning.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
        return float(text)
    except FileNotFoundError:
        print(f"[warn] Index-time TXT not found: {path!s} (using 0)")
    except ValueError:
        print(f"[warn] Could not parse float from TXT: {path!s} (using 0)")
    except Exception as e:
        print(f"[warn] Unexpected error reading TXT {path!s}: {e!r} (using 0)")
    return 0.0


def _safe_index_time_from_gt_json(path: Path) -> float:
    """
    Read a JSON file and return the 'index_time' field (seconds).
    If file missing or field absent, returns 0.0 with a warning.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        val = data.get("index_time", 0.0)
        if not isinstance(val, (int, float)):
            print(f"[warn] 'index_time' not numeric in {path!s} (using 0)")
            return 0.0
        return float(val)
    except FileNotFoundError:
        print(f"[warn] GT JSON not found: {path!s} (using 0)")
    except json.JSONDecodeError:
        print(f"[warn] Malformed JSON in {path!s} (using 0)")
    except Exception as e:
        print(f"[warn] Unexpected error reading JSON {path!s}: {e!r} (using 0)")
    return 0.0


def compute_index_build_times(
    algorithms: Sequence[str],
    *,
    gt_json_path: Optional[Union[str, Path]] = None,
    dendrogram_index_time_txt_path: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Compute per-algorithm index build time in SECONDS (base unit).

    Rules (per your spec):
      - SortedIndexScan: read from GT JSON under field 'index_time'
      - EpsGreedy / UCB / UniformExploration: read single float (seconds) from given TXT
      - UniformSample: 0
      - All other algorithms: 0 (unless you extend later)

    Parameters
    ----------
    algorithms : Sequence[str]
        Algorithm keys present in `result_stats` / `METHODS`.
    gt_json_path : Optional[path-like]
        Path to the GT JSON used by SortedIndexScan.
    dendrogram_index_time_txt_path : Optional[path-like]
        Path to TXT with a single float for EpsGreedy/UCB/UniformExploration.

    Returns
    -------
    Dict[str, float]
        Mapping from algorithm key → index build time in seconds.
    """
    # Pre-read once
    sorted_index_time = 0.0
    if gt_json_path is not None:
        sorted_index_time = _safe_index_time_from_gt_json(Path(gt_json_path))

    dendro_time = 0.0
    if dendrogram_index_time_txt_path is not None:
        dendro_time = _safe_float_from_txt(Path(dendrogram_index_time_txt_path))

    build_times: Dict[str, float] = {}
    for alg in algorithms:
        if alg == "SortedIndexScan":
            build_times[alg] = sorted_index_time
        elif alg in {"EpsGreedy", "UCB", "UniformExploration"}:
            build_times[alg] = dendro_time
        elif alg == "UniformSample":
            build_times[alg] = 0.0
        else:
            # Default to 0 unless specified otherwise
            build_times[alg] = 0.0
    return build_times


# --------------------------------------------------------------------------------------
# Plot 1: TOTAL LATENCY (stacked: index build + algorithm runtime)
# --------------------------------------------------------------------------------------

def plot_total_latency_stacked(
    result_stats: Mapping[str, Mapping[str, List[float]]],
    order: Union[str, List[str], None],
    *,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    time_unit: str = "s",
    include_index_build_time: bool = False,
    xtick_rotation: int = 20,
    fontsize: float = 18,
    # New, explicit inputs for index build time sources:
    gt_json_path: Optional[Union[str, Path]] = None,
    dendrogram_index_time_txt_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot an end-to-end latency stacked bar chart for the provided algorithms.

    Assumptions
    -----------
    - `result_stats[alg]["time"]` is a running total in SECONDS; the last element is end-to-end runtime (excluding index build).
    - Units are converted for display using TIME_UNITS[time_unit].

    Parameters
    ----------
    result_stats : Mapping[str, Mapping[str, List[float]]]
        Per-algorithm stats. Must include `["time"]` array in seconds.
    order : Union[str, List[str], None]
        If None, uses `global_order`. If str, uses `method_collections[order]`.
        Otherwise, a concrete ordered list of algorithm keys.
    ylabel, xlabel, title : Optional[str]
        Axis labels and title. Defaults applied if None.
    filename : Optional[str]
        If provided, the plot is saved to this path.
    time_unit : str
        One of TIME_UNITS keys: {"s","ms","μs","ns","hr"}.
    include_index_build_time : bool
        If True, stack index build time below algorithm runtime.
    xtick_rotation : int
        Rotation angle for x-tick labels.
    fontsize : float
        Base font size for labels/legend.
    gt_json_path : Optional[path-like]
        Source for SortedIndexScan build time ('index_time' field).
    dendrogram_index_time_txt_path : Optional[path-like]
        Source for EpsGreedy/UCB/UniformExploration build time (single float).

    Returns
    -------
    None
        Shows and/or saves a matplotlib figure.
    """
    if time_unit not in TIME_UNITS:
        raise ValueError(f"Unknown time_unit {time_unit!r}. Options: {list(TIME_UNITS)}")
    conversion = TIME_UNITS[time_unit]

    # Resolve order using globals if needed
    if order is None:
        order = global_order  # type: ignore[name-defined]
    if isinstance(order, str):
        order = method_collections[order]  # type: ignore[name-defined]
    algorithms: List[str] = [alg for alg in order if alg in result_stats]

    # Gather runtimes (last element of cumulative 'time' array)
    alg_runtimes_sec: List[float] = [float(result_stats[alg]["time"][-1]) for alg in algorithms]

    # Compute build times in seconds (per spec)
    if include_index_build_time:
        build_times_sec_map = compute_index_build_times(
            algorithms,
            gt_json_path=gt_json_path,
            dendrogram_index_time_txt_path=dendrogram_index_time_txt_path,
        )
        build_times_sec = [build_times_sec_map[alg] for alg in algorithms]
    else:
        build_times_sec = [0.0] * len(algorithms)

    # Convert to requested display units
    alg_runtimes_disp = [t * conversion for t in alg_runtimes_sec]
    build_times_disp = [t * conversion for t in build_times_sec]

    # Plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.6

    for i, alg in enumerate(algorithms):
        # Index build time (outlined)
        if build_times_disp[i] > 0:
            plt.bar(
                i,
                build_times_disp[i],
                bar_width,
                color="none",
                edgecolor=METHODS[alg]["color"],
                linewidth=2.0,
                label="Index build time" if i == 0 else None,
            )
        # Algorithm runtime (filled)
        plt.bar(
            i,
            alg_runtimes_disp[i],
            bar_width,
            facecolor=METHODS[alg]["color"],
            alpha=0.5,
            edgecolor=METHODS[alg]["color"],
            linewidth=2.0,
            bottom=build_times_disp[i],
            label="Algorithm runtime" if i == 0 else None,
        )

    # Labels/legend
    labels = [METHODS[alg]["name"] for alg in algorithms]
    plt.xticks(range(len(algorithms)), labels, rotation=xtick_rotation, fontsize=fontsize, ha="right")
    plt.ylabel(ylabel if ylabel else f"End-to-End Latency ({time_unit})", fontsize=fontsize)
    plt.xlabel(xlabel if xlabel else "", fontsize=fontsize)
    plt.title(title if title else "", fontsize=fontsize)

    # Stable legend order (two handles max)
    handles, leg_labels = plt.gca().get_legend_handles_labels()
    if handles:
        # Ensure both labels appear even if one is absent because all zeros
        legend_order = []
        if "Index build time" in leg_labels:
            legend_order.append(("Index build time", "outline"))
        if "Algorithm runtime" in leg_labels:
            legend_order.append(("Algorithm runtime", "filled"))

        # Build custom handles for visual clarity
        custom_handles = []
        for text, kind in legend_order:
            if kind == "outline":
                custom_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=2.0, label=text))
            else:
                custom_handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="gray", alpha=0.3, edgecolor="black", linewidth=2.0, label=text))
        plt.legend(handles=custom_handles, fontsize=fontsize, loc="center right")

    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------------------
# Plot 2: PER-ITERATION LATENCY (stacked breakdown)
# --------------------------------------------------------------------------------------

def plot_iter_latency_stacked(
    result_stats: Mapping[str, Mapping[str, float]],
    order: Union[str, List[str], None],
    *,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    time_unit: str = "s",
    xtick_rotation: int = 20,
    iter_colors: Optional[List[str]] = None,
    include_scoring_fn: bool = False,
    fontsize: float = 18,
) -> None:
    """
    Plot a stacked per-iteration latency breakdown.

    Assumptions
    -----------
    - `result_stats[alg][<category>]` is the per-iteration latency in SECONDS for each category.
      Categories considered:
        * overhead_algo
        * overhead_pq
        * overhead_other
        * overhead_scorer (only if include_scoring_fn=True)

    Parameters
    ----------
    result_stats : Mapping[str, Mapping[str, float]]
        Per-algorithm stats with per-iteration time per category (in seconds).
    order : Union[str, List[str], None]
        If None, uses `global_order`. If str, uses `method_collections[order]`.
        Otherwise, a concrete ordered list of algorithm keys.
    ylabel, xlabel, title : Optional[str]
        Axis labels and title. Defaults applied if None.
    filename : Optional[str]
        If provided, the plot is saved to this path.
    time_unit : str
        One of TIME_UNITS keys: {"s","ms","μs","ns","hr"}.
    xtick_rotation : int
        Rotation angle for x-tick labels.
    iter_colors : Optional[List[str]]
        Custom colors for each stacked category, in order.
    include_scoring_fn : bool
        If True, adds the 'overhead_scorer' category to the stack.
    fontsize : float
        Base font size for labels/legend.

    Returns
    -------
    None
        Shows and/or saves a matplotlib figure.
    """
    if time_unit not in TIME_UNITS:
        raise ValueError(f"Unknown time_unit {time_unit!r}. Options: {list(TIME_UNITS)}")
    conversion = TIME_UNITS[time_unit]

    # Resolve order using globals if needed
    if order is None:
        order = global_order  # type: ignore[name-defined]
    if isinstance(order, str):
        order = method_collections[order]  # type: ignore[name-defined]
    algorithms: List[str] = [alg for alg in order if alg in result_stats]

    # Categories + labels
    categories: List[str]
    labels: List[str]
    if include_scoring_fn:
        categories = ["overhead_algo", "overhead_pq", "overhead_other", "overhead_scorer"]
        labels = ["Algorithm logic", "Priority queue", "Other", "Scoring function"]
    else:
        categories = ["overhead_algo", "overhead_pq", "overhead_other"]
        labels = ["Algorithm logic", "Priority queue", "Other"]

    # Colors
    if iter_colors is None:
        iter_colors = ["#DDAA33", "#BB5566", "#000000"] if not include_scoring_fn else ["#DDAA33", "#BB5566", "#000000", "#004488"]

    # Build matrix of heights in display units
    breakdown_disp: Dict[str, List[float]] = {
        alg: [float(result_stats[alg][cat]) * conversion for cat in categories]
        for alg in algorithms
    }

    # Plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.6
    bottoms = np.zeros(len(algorithms), dtype=float)

    for i, (label, cat) in enumerate(zip(labels, categories)):
        heights = [breakdown_disp[alg][i] for alg in algorithms]
        plt.bar(
            range(len(algorithms)),
            heights,
            bar_width,
            bottom=bottoms,
            label=label,
            color=iter_colors[i],
            alpha=0.7,
            linewidth=1.0,
        )
        bottoms += np.array(heights, dtype=float)

    # Labels/legend
    xticklabels = [METHODS[alg]["name"] for alg in algorithms]
    plt.xticks(range(len(algorithms)), xticklabels, rotation=xtick_rotation, fontsize=fontsize, ha="right")
    plt.ylabel(ylabel if ylabel else f"Latency Per Iteration ({time_unit})", fontsize=fontsize)
    plt.xlabel(xlabel if xlabel else "", fontsize=fontsize)
    plt.title(title if title else "", fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    if filename:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
