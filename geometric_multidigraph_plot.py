"""Matplotlib drawing for geometric multidigraphs.

Import :func:`plot_geometric_rgg` from here (or via ``random_geometric_graph`` re-export)
so graph generation code does not require matplotlib unless you plot.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

from geometric_multidigraph_config import GeometricMultidigraphConfig

# Per-protocol panels: profile contains this protocol vs not.
_NODE_HAS_PROTOCOL = "#2a9d8f"
_NODE_LACKS_PROTOCOL = "#e63946"
_NODE_SIZE = 100

# Default layout: one subplot per catalog protocol (edges optional; nodes colored by profile).
PER_PROTOCOL_FIGURE = True


def _unit_normal(lo, hi, pos: dict) -> tuple[float, float]:
    x0, y0 = pos[lo]
    x1, y1 = pos[hi]
    dx, dy = x1 - x0, y1 - y0
    L = (dx * dx + dy * dy) ** 0.5 or 1e-12
    dx, dy = dx / L, dy / L
    return (-dy, dx)


def _shift_sign(G: nx.DiGraph, u, v) -> float:
    lo, hi = min(u, v), max(u, v)
    if G.has_edge(lo, hi) and G.has_edge(hi, lo):
        return 1.0 if (u == lo and v == hi) else -1.0
    return 0.0


def _offset_endpoints(G: nx.DiGraph, u, v, pos: dict, sep: float):
    lo, hi = min(u, v), max(u, v)
    nx_n, ny_n = _unit_normal(lo, hi, pos)
    s = _shift_sign(G, u, v) * sep
    return (
        (pos[u][0] + s * nx_n, pos[u][1] + s * ny_n),
        (pos[v][0] + s * nx_n, pos[v][1] + s * ny_n),
    )


def _draw_offset_edges(
    ax,
    G_full: nx.DiGraph,
    pos: dict,
    *,
    sep: float,
    edgelist: list[tuple] | None,
    draw_arrows: bool,
) -> None:
    for u, v in (edgelist if edgelist is not None else list(G_full.edges())):
        (x0, y0), (x1, y1) = _offset_endpoints(G_full, u, v, pos, sep)
        if draw_arrows:
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color="0.35",
                    linewidth=1.0,
                    shrinkA=8,
                    shrinkB=8,
                    zorder=1,
                )
            )
        else:
            dx, dy = x1 - x0, y1 - y0
            L = (dx * dx + dy * dy) ** 0.5 or 1e-12
            m = 0.012
            sx, sy = dx / L * m, dy / L * m
            ax.plot(
                [x0 + sx, x1 - sx],
                [y0 + sy, y1 - sy],
                color="0.35",
                linewidth=1.2,
                solid_capstyle="round",
                zorder=1,
            )


def _draw_offset_labels(ax, G_full: nx.DiGraph, pos: dict, labels: dict, *, sep: float) -> None:
    for (u, v), text in labels.items():
        (x0, y0), (x1, y1) = _offset_endpoints(G_full, u, v, pos, sep)
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        ang = math.degrees(math.atan2(dy, dx))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        ax.text(
            xm,
            ym,
            str(text),
            fontsize=6,
            ha="center",
            va="center",
            rotation=ang,
            rotation_mode="anchor",
            zorder=3,
        )


def _position_limits(
    pos: dict, *, pad: float = 0.04
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Default unit square [0, 1]²; widen if any node lies outside."""
    if not pos:
        return (0.0, 1.0), (0.0, 1.0)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    lo_x = 0.0 if min(xs) >= 0.0 else min(xs) - pad
    hi_x = 1.0 if max(xs) <= 1.0 else max(xs) + pad
    lo_y = 0.0 if min(ys) >= 0.0 else min(ys) - pad
    hi_y = 1.0 if max(ys) <= 1.0 else max(ys) + pad
    return (lo_x, hi_x), (lo_y, hi_y)


def _apply_position_ticks(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Tick marks at 0, ¼, ½, ¾, 1 when that span is visible."""
    quarter = [0.0, 0.25, 0.5, 0.75, 1.0]
    if xlim[0] <= 0.0 and xlim[1] >= 1.0:
        ax.set_xticks(quarter)
        ax.set_xticklabels([f"{t:g}" for t in quarter])
    if ylim[0] <= 0.0 and ylim[1] >= 1.0:
        ax.set_yticks(quarter)
        ax.set_yticklabels([f"{t:g}" for t in quarter])


def _style_axes(ax, *, xlabel: bool = False, ylabel: bool = False) -> None:
    ax.set_facecolor("#f2f2f2")
    ax.tick_params(
        axis="both",
        which="major",
        bottom=True,
        left=True,
        labelbottom=True,
        labelleft=True,
        labelsize=8,
        colors="#3d3d3d",
    )
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#3d3d3d")
        spine.set_linewidth(1.4)
    if xlabel:
        ax.set_xlabel("x", fontsize=9)
    if ylabel:
        ax.set_ylabel("y", fontsize=9)


def _protocols_for_panels(
    G: nx.DiGraph, proto_order: Sequence[str] | None
) -> list[str]:
    """Protocols to show as subplots (one panel each, even if no edges).

    If ``proto_order`` is set, every catalog entry gets a panel. Otherwise use all
    tags seen on node profiles or edges, sorted alphabetically.
    """
    if proto_order:
        return list(proto_order)
    seen: set[str] = set()
    for _nid, data in G.nodes(data=True):
        seen.update(str(p) for p in (data.get("profile") or []))
    for _u, _v, data in G.edges(data=True):
        p = data.get("protocol")
        if p is not None:
            seen.add(str(p))
    return sorted(seen, key=str)


def format_config_for_plot(cfg: GeometricMultidigraphConfig) -> str:
    """One-line run summary for figure suptitles."""
    pos = cfg.positions
    prof = cfg.profiles
    pk = pos.sampler_kwargs()
    pos_parts = [f"dist={pos.distribution}"]
    for key in sorted(pk.keys()):
        val = pk[key]
        if isinstance(val, float):
            pos_parts.append(f"{key}={val:g}")
        else:
            pos_parts.append(f"{key}={val}")
    pos_str = " ".join(pos_parts)
    return (
        f"n={cfg.n}  conn_r={cfg.radius:g}  pos_seed={cfg.position_seed}  "
        f"prof_seed={cfg.profile_seed}  {pos_str}  "
        f"ao={prof.alpha_outer:g}  ai={prof.alpha_inner:g}  mt={prof.mean_tags:g}"
    )


def _node_has_protocol(G: nx.DiGraph, node, protocol: str) -> bool:
    prof = G.nodes[node].get("profile") or []
    return protocol in prof


def _node_style_for_protocol(
    G: nx.DiGraph,
    nodes: list,
    protocol: str,
) -> tuple[list[float], list[str]]:
    sizes = [_NODE_SIZE] * len(nodes)
    colors = [
        _NODE_HAS_PROTOCOL
        if _node_has_protocol(G, v, protocol)
        else _NODE_LACKS_PROTOCOL
        for v in nodes
    ]
    return sizes, colors


def _subplot_grid(n: int) -> tuple[int, int]:
    """Rows × cols for ``n`` protocol panels; extra slots are hidden."""
    if n <= 0:
        return 1, 1
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n == 3:
        return 1, 3
    ncols = 2
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def plot_geometric_rgg(
    G: nx.DiGraph,
    *,
    per_protocol: bool = PER_PROTOCOL_FIGURE,
    sep: float = 0.018,
    save_path: str | Path | None = None,
    dpi: float = 150,
    show: Optional[bool] = None,
    proto_order: Sequence[str] | None = None,
    config: GeometricMultidigraphConfig | None = None,
) -> None:
    """Draw offset edges/labels.

    Per-protocol panels (``per_protocol=True``): highlight nodes whose profile
    contains that protocol; combined view uses out-degree coloring.

    Pass ``config`` to show run parameters in the figure suptitle.
    """
    pos = nx.get_node_attributes(G, "pos")
    labels_all = {(u, v): d.get("protocol") for u, v, d in G.edges(data=True)}
    deg = dict(G.out_degree())
    nodes = list(G.nodes())
    default_sizes = [100 + deg[v] * 10 for v in nodes]
    default_colors = [deg[v] for v in nodes]

    draw_arrows = bool(G.graph.get("bidirectional", True))
    xlim, ylim = _position_limits(pos)

    figure_title = (
        format_config_for_plot(config) if config is not None else "Geometric multidigraph"
    )

    def one_ax(
        ax,
        edgelist: list[tuple],
        edge_labels: dict,
        *,
        highlight_protocol: str | None = None,
        xlabel: bool = False,
        ylabel: bool = False,
    ) -> None:
        _draw_offset_edges(ax, G, pos, sep=sep, edgelist=edgelist, draw_arrows=draw_arrows)
        if highlight_protocol is not None:
            sizes, colors = _node_style_for_protocol(G, nodes, highlight_protocol)
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=nodes,
                node_size=sizes,
                node_color=colors,
            )
        else:
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                nodelist=nodes,
                node_size=default_sizes,
                node_color=default_colors,
                cmap=plt.cm.viridis,
            )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        _draw_offset_labels(ax, G, pos, edge_labels, sep=sep)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        _apply_position_ticks(ax, xlim, ylim)
        _style_axes(ax, xlabel=xlabel, ylabel=ylabel)

    if per_protocol:
        protocols = _protocols_for_panels(G, proto_order)
        if not protocols:
            fig, ax = plt.subplots(figsize=(8, 8))
            one_ax(ax, list(G.edges()), labels_all, xlabel=True, ylabel=True)
            fig.suptitle(f"{figure_title}\n(no protocols in catalog)", fontsize=9)
            plt.tight_layout()
        else:
            n = len(protocols)
            nrows, ncols = _subplot_grid(n)
            fw, fh = 5.5 * ncols, 5.2 * nrows
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(fw, fh), squeeze=False
            )
            fig.patch.set_facecolor("#cfcfcf")
            axes_flat = axes.ravel().tolist()

            for i, proto in enumerate(protocols):
                ax = axes_flat[i]
                edgelist = [
                    (u, v)
                    for u, v, d in G.edges(data=True)
                    if d.get("protocol") == proto
                ]
                edge_labels = {(u, v): proto for u, v in edgelist}
                row, col = divmod(i, ncols)
                n_has = sum(1 for v in nodes if _node_has_protocol(G, v, proto))
                one_ax(
                    ax,
                    edgelist,
                    edge_labels,
                    highlight_protocol=proto,
                    xlabel=row == nrows - 1,
                    ylabel=col == 0,
                )
                edge_note = f"{len(edgelist)} edges" if edgelist else "no edges"
                ax.set_title(
                    f"{proto}  ({n_has}/{len(nodes)} nodes, {edge_note})",
                    pad=10,
                    fontsize=11,
                    fontweight="medium",
                )

            for j in range(n, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(figure_title, fontsize=9, y=1.0)
            fig.subplots_adjust(
                wspace=0.28, hspace=0.38, left=0.12, right=0.97, top=0.82, bottom=0.12
            )
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        one_ax(ax, list(G.edges()), labels_all, xlabel=True, ylabel=True)
        fig.suptitle(figure_title, fontsize=9)
        plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=dpi)
    if show is None:
        show = save_path is None
    if show:
        plt.show()
    else:
        plt.close(fig)
