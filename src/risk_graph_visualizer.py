import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D


def save_risk_graph_plot(risk_graph, output_dir, task_id=None):
    """Render a task-level risk graph without requiring a display server."""
    if not output_dir:
        return ""

    graph = risk_graph.to_dict() if hasattr(risk_graph, "to_dict") else dict(risk_graph)
    if task_id is None:
        task_id = graph.get("task_id", 0)
    task_id = int(task_id)

    new_types = list(graph.get("new_types", []))
    old_types = list(graph.get("old_types", []))
    edges = list(graph.get("edges", []))
    figure_height = max(4.0, 1.2 * max(len(new_types), len(old_types), 1) + 1.5)
    fig, ax = plt.subplots(figsize=(10, figure_height))
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.6, max(len(new_types), len(old_types), 1) - 0.4)
    ax.axis("off")

    new_positions = _node_positions(new_types, 0.0)
    old_positions = _node_positions(old_types, 1.0)
    _draw_nodes(ax, new_positions, "#e76f51", "New types")
    _draw_nodes(ax, old_positions, "#4c78a8", "Old types")
    _draw_edges(ax, edges, new_positions, old_positions)

    title = "Semantic Risk Graph - Task %d" % task_id
    domain_name = graph.get("domain_name", "")
    if domain_name:
        title += " (%s)" % domain_name
    ax.set_title(title, fontsize=14, pad=18)
    ax.text(0.0, -0.45, "New entity types", ha="center", va="top", fontsize=11)
    ax.text(1.0, -0.45, "Old entity types", ha="center", va="top", fontsize=11)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "risk_graph_task_%d.png" % task_id)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_cumulative_risk_graph_plot(graph_history, output_dir, task_id, is_final=False):
    """Render all risk edges observed up to the current continual-learning task."""
    if not output_dir:
        return ""

    nodes, node_task_ids, edges = _collect_cumulative_graph(graph_history)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.axis("off")
    if not nodes:
        ax.set_title("Cumulative Semantic Risk Graph - Task %d" % int(task_id), fontsize=14, pad=18)
    else:
        positions = _circular_positions(nodes)
        _draw_cumulative_nodes(ax, positions, node_task_ids, task_id)
        _draw_cumulative_edges(ax, edges, positions)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal")
        ax.set_title("Cumulative Semantic Risk Graph - Up to Task %d" % int(task_id), fontsize=15, pad=26)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label="Current task", markerfacecolor="#e76f51", markeredgecolor="#222222", markersize=10),
            Line2D([0], [0], marker="o", color="w", label="Previous task", markerfacecolor="#4c78a8", markeredgecolor="#222222", markersize=10)
        ]
        ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False)

    os.makedirs(output_dir, exist_ok=True)
    filename = "risk_graph_final.png" if is_final else "risk_graph_cumulative_task_%d.png" % int(task_id)
    output_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _node_positions(node_names, x_position):
    if not node_names:
        return {}
    max_y = len(node_names) - 1
    return {
        name: (x_position, max_y - index)
        for index, name in enumerate(node_names)
    }


def _draw_nodes(ax, positions, color, label):
    is_first = True
    for name, (x_position, y_position) in positions.items():
        ax.scatter(
            [x_position], [y_position], s=1800, c=color, edgecolors="#222222",
            linewidths=1.0, zorder=3, label=label if is_first else None
        )
        ax.text(x_position, y_position, name, ha="center", va="center", fontsize=10, zorder=4)
        is_first = False


def _draw_edges(ax, edges, new_positions, old_positions):
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in new_positions or target not in old_positions:
            continue

        risk = min(max(float(edge.get("risk", 0.0)), 0.0), 1.0)
        start = new_positions[source]
        end = old_positions[target]
        arrow = FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=16,
            linewidth=1.0 + 5.0 * risk, color=plt.cm.Reds(0.35 + 0.6 * risk),
            alpha=0.85, shrinkA=30, shrinkB=30, zorder=2
        )
        ax.add_patch(arrow)
        label_x = (start[0] + end[0]) / 2.0
        label_y = (start[1] + end[1]) / 2.0
        ax.text(label_x, label_y + 0.08, "%.2f" % risk, ha="center", va="bottom", fontsize=9)


def _collect_cumulative_graph(graph_history):
    nodes = []
    node_task_ids = {}
    edges = []
    for graph in graph_history:
        task_id = int(graph.get("task_id", 0))
        for node_name in graph.get("new_types", []):
            if node_name not in node_task_ids:
                nodes.append(node_name)
                node_task_ids[node_name] = task_id
        edges.extend(graph.get("edges", []))
    return nodes, node_task_ids, edges


def _circular_positions(nodes):
    node_count = len(nodes)
    angles = [math.pi / 2.0 - 2.0 * math.pi * index / node_count for index in range(node_count)]
    return {
        node_name: (1.35 * math.cos(angle), 1.35 * math.sin(angle))
        for node_name, angle in zip(nodes, angles)
    }


def _draw_cumulative_nodes(ax, positions, node_task_ids, current_task_id):
    for node_name, (x_position, y_position) in positions.items():
        is_current = node_task_ids[node_name] == int(current_task_id)
        color = "#e76f51" if is_current else "#4c78a8"
        ax.scatter(
            [x_position], [y_position], s=1650, c=color, edgecolors="#222222",
            linewidths=1.2, zorder=3
        )
        ax.text(x_position, y_position, node_name, ha="center", va="center", fontsize=10, zorder=4)


def _draw_cumulative_edges(ax, edges, positions):
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source not in positions or target not in positions:
            continue
        risk = min(max(float(edge.get("risk", 0.0)), 0.0), 1.0)
        is_background = "background_risk" in edge.get("risk_type", [])
        edge_color = "#9aa5b1" if is_background else plt.cm.Reds(0.35 + 0.6 * risk)
        arrow = FancyArrowPatch(
            positions[source], positions[target], arrowstyle="->", mutation_scale=15,
            linewidth=0.9 + 4.2 * risk, color=edge_color,
            alpha=0.5 if is_background else 0.85, shrinkA=28, shrinkB=28,
            connectionstyle="arc3,rad=0.14", zorder=2
        )
        ax.add_patch(arrow)
        start = positions[source]
        end = positions[target]
        ax.text(
            (start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0,
            "%.2f" % risk, ha="center", va="center", fontsize=8,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.75}
        )
