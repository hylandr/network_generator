"""Minimal directed RGG + parallel offset arrows and labels on each offset segment."""

import math
import random
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Rectangle

from crp_sampler import CRPSampler
from graph_export import (
    ensure_export_subdirectory_for_metadata,
    graph_from_export,
    graph_stats,
    load_graph_export,
    print_graph_stats_bundle,
    protocol_edge_counts,
    protocol_profile_node_counts,
    subgraph_by_protocol,
    write_graph_export,
)







# -----------------------------------------------------------------------------
# Shared between plotting and generation (protocol names on edges / figure mode)
# -----------------------------------------------------------------------------

# _PROTOCOLS = ("ssh", "ftp")
PER_PROTOCOL_FIGURE = True

_PROTOCOLS = ("protocol_1", "protocol_2", "protocol_3","protocol_4","protocol_5","protocol_6")


# =============================================================================
# PLOTTING
#
#   Matplotlib / NetworkX drawing: parallel offset geometry, optional arrows vs
#   plain lines (from G.graph["bidirectional"]), node styling, multi-panel layout.
# =============================================================================


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


def _style_protocol_panel(ax) -> None:
    ax.set_facecolor("#f2f2f2")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#3d3d3d")
        spine.set_linewidth(1.4)
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            transform=ax.transAxes,
            linewidth=2.0,
            edgecolor="#2a2a2a",
            facecolor="none",
            zorder=1000,
            clip_on=False,
        )
    )


def _protocols_with_edges(G: nx.DiGraph, proto_order: tuple[str, ...]) -> list[str]:
    """Protocols that appear on at least one edge: ``proto_order`` first, then any others."""
    seen_on_edge = {d.get("protocol") for _, _, d in G.edges(data=True)}
    seen_on_edge.discard(None)
    ordered = [p for p in proto_order if p in seen_on_edge]
    rest = sorted(seen_on_edge - set(proto_order), key=str)
    return ordered + rest


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
) -> None:
    """Offset edges/labels, nodes by out-degree; optional one panel per protocol that has edges.

    If ``save_path`` is set, the figure is written there (``.png`` recommended).
    ``show``: if ``None``, the figure is only shown when ``save_path`` is also ``None``."""
    pos = nx.get_node_attributes(G, "pos")
    labels_all = {(u, v): d.get("protocol") for u, v, d in G.edges(data=True)}
    deg = dict(G.out_degree())
    nodes = list(G.nodes())
    sizes = [100 + deg[v] * 10 for v in nodes]
    colors = [deg[v] for v in nodes]

    draw_arrows = bool(G.graph.get("bidirectional", True))

    def one_ax(ax, edgelist: list[tuple], edge_labels: dict, *, framed: bool) -> None:
        _draw_offset_edges(ax, G, pos, sep=sep, edgelist=edgelist, draw_arrows=draw_arrows)
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=nodes,
            node_size=sizes,
            node_color=colors,
            cmap=plt.cm.viridis,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        _draw_offset_labels(ax, G, pos, edge_labels, sep=sep)
        ax.set_aspect("equal", adjustable="datalim")
        if framed:
            _style_protocol_panel(ax)
        else:
            ax.set_axis_off()

    if per_protocol:
        present = _protocols_with_edges(G, _PROTOCOLS)
        if not present:
            fig, ax = plt.subplots(figsize=(8, 8))
            one_ax(ax, list(G.edges()), labels_all, framed=False)
            fig.suptitle("No edges with protocol set", fontsize=11)
            plt.tight_layout()
        else:
            n = len(present)
            nrows, ncols = _subplot_grid(n)
            fw, fh = 5.5 * ncols, 5.2 * nrows
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(fw, fh), squeeze=False
            )
            fig.patch.set_facecolor("#cfcfcf")
            axes_flat = axes.ravel().tolist()

            for i, proto in enumerate(present):
                ax = axes_flat[i]
                edgelist = [
                    (u, v)
                    for u, v, d in G.edges(data=True)
                    if d.get("protocol") == proto
                ]
                edge_labels = {(u, v): proto for u, v in edgelist}
                one_ax(ax, edgelist, edge_labels, framed=True)
                ax.set_title(proto, pad=10, fontsize=11, fontweight="medium")

            for j in range(n, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle("Edges by protocol (offset geometry from full graph)", fontsize=11)
            fig.subplots_adjust(
                wspace=0.22, hspace=0.28, left=0.06, right=0.97, top=0.88, bottom=0.06
            )
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        one_ax(ax, list(G.edges()), labels_all, framed=False)
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


# =============================================================================
# GRAPH GENERATION
#
#   Build a directed graph from an undirected random geometric graph: copy node
#   positions, then orient edges (one or both directions) and attach protocols.
# =============================================================================

POSITION_DISTRIBUTIONS = ("uniform", "gaussian", "beta", "grid")


def sample_positions(
    n: int,
    rnd: random.Random,
    distribution: str = "uniform",
    **kwargs,
) -> dict[int, tuple[float, float]]:
    """Sample node positions on [0, 1]² for use with ``random_geometric_graph``.

    ``distribution`` is one of ``POSITION_DISTRIBUTIONS``. Required kwargs per type
    (raises ``KeyError`` if missing):

    - uniform: ``x0``, ``x1``, ``y0``, ``y1``
    - gaussian: ``mu_x``, ``mu_y``, ``sigma`` (coordinates are not clamped to [0, 1])
    - beta: ``alpha``, ``beta``
    - grid: ``jitter``; optional ``cols`` (auto ``ceil(sqrt(n))`` when omitted)
    """
    match distribution:
        # Flat density: every point in the rectangle [x0, x1] × [y0, y1] is equally likely.
        case "uniform":
            x0 = kwargs["x0"]
            x1 = kwargs["x1"]
            y0 = kwargs["y0"]
            y1 = kwargs["y1"]
            pos = {
                i: (rnd.uniform(x0, x1), rnd.uniform(y0, y1)) for i in range(n)
            }

        # Bell-shaped cluster around (mu_x, mu_y); most nodes near the center, fewer at the edges.
        case "gaussian":
            mu_x = kwargs["mu_x"]
            mu_y = kwargs["mu_y"]
            sigma = kwargs["sigma"]
            pos = {
                i: (rnd.gauss(mu_x, sigma), rnd.gauss(mu_y, sigma)) for i in range(n)
            }

        # Skewed density on [0, 1]²; alpha=beta>1 pulls toward center, <1 toward corners.
        case "beta":
            alpha = kwargs["alpha"]
            beta_shape = kwargs["beta"]
            pos = {
                i: (
                    rnd.betavariate(alpha, beta_shape),
                    rnd.betavariate(alpha, beta_shape),
                )
                for i in range(n)
            }

        # Regular lattice with a small random offset (jitter=0 gives a perfect grid).
        case "grid":
            jitter = kwargs["jitter"]
            cols = kwargs["cols"] if "cols" in kwargs else int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            pos = {
                i: (
                    (i % cols + 0.5) / cols + rnd.uniform(-jitter, jitter),
                    (i // cols + 0.5) / rows + rnd.uniform(-jitter, jitter),
                )
                for i in range(n)
            }

        case _:
            raise ValueError(
                f"unknown distribution {distribution!r}; "
                f"choose from {POSITION_DISTRIBUTIONS}"
            )

    return pos


def geometric_multidigraph_network(
    radius: float,
    pos: dict[int, tuple[float, float]],
    *,
    profiles: Sequence[Sequence[str]],
) -> nx.MultiDiGraph:

    n = len(pos)
    if len(profiles) != n:
        raise ValueError(
            f"profiles has length {len(profiles)} but pos has {n} nodes"
        )
    U = nx.random_geometric_graph(n, radius, pos=pos)
    U = U.to_directed()
    G = nx.MultiDiGraph()
    G.add_nodes_from(U.nodes(data=True))
    for i in G.nodes():
        G.nodes[i]["profile"] = list(profiles[int(i)])

    add_service_edges(G, U)

    G.graph["bidirectional"] = True
    return G


def add_service_edges(G: nx.DiGraph, U: nx.Graph) -> None:
    for u, v, _d in U.edges(data=True):
        prof = G.nodes[u].get("profile", [])
        if prof:
            for protocol in prof:
                G.add_edge(v, u, protocol=protocol)
        
        # if not prof:
        #     protocol = rnd.choice(_PROTOCOLS)
        # elif len(prof) == 1:
        #     protocol = prof[0]
        # else:
        #     protocol = rnd.choice(prof)
        # G.add_edge(v, u, protocol=protocol)
       



# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":


    seeds = 50


    for seed in range(seeds):


        for alpha_outer in [2,4,8]:
            for alpha_inner in [2,4,8]:
                for mean_tags in [2,4,6]:
                    for radius in [0.15,0.25,0.35]:

                        n = 30
                        
                        rnd = random.Random(seed)

                        initial_profiles = [["protocol_1", "protocol_2", "protocol_3"]]

                        # alpha_outer = 4
                        # alpha_inner = 8
                        # mean_tags = 3

                        profiles = CRPSampler.sample_profiles(
                            rnd,
                            n,
                            _PROTOCOLS,
                            alpha_outer=alpha_outer,
                            alpha_inner=alpha_inner,
                            mean_tags=mean_tags,
                            initial_profiles=initial_profiles,
                        )

                        pos_distribution = "uniform"
                        pos_kw = {"x0": 0, "x1": 1, "y0": 0, "y1": 1}

                        # radius = 0.25

                        pos = sample_positions(n, rnd, pos_distribution, **pos_kw)

                        G = geometric_multidigraph_network(radius=radius, pos=pos, profiles=profiles)

                        export_meta: dict = {
                            "model": "geometric_multidigraph",
                            "n": n,
                            "seed": seed,
                            "radius": radius,
                            "initial_profiles": initial_profiles,
                            "profiles_sampler": {
                                "qualified_name": f"{CRPSampler.sample_profiles.__module__}.{CRPSampler.sample_profiles.__qualname__}",
                                "inputs": {
                                    "n": n,
                                    "catalog": list(_PROTOCOLS),
                                    "alpha_outer": alpha_outer,
                                    "alpha_inner": alpha_inner,
                                    "mean_tags": mean_tags,
                                    "initial_profiles": initial_profiles,
                                },
                            },
                            "positions_sampler": {
                                "qualified_name": f"{sample_positions.__module__}.{sample_positions.__qualname__}",
                                "inputs": {
                                    "n": n,
                                    "distribution": pos_distribution,
                                    **pos_kw,
                                },
                            },
                        }
                        catalog = list(_PROTOCOLS)
                        export_meta["protocol_edge_counts"] = protocol_edge_counts(G, protocols=catalog)
                        export_meta["protocol_profile_node_counts"] = protocol_profile_node_counts(
                            G, protocols=catalog
                        )
                        repo_root = Path(__file__).resolve().parent
                        export_dir = ensure_export_subdirectory_for_metadata(repo_root, export_meta)
                        export_path = export_dir / f"rgg_with_stats_{export_meta['seed']}.json"

                        write_graph_export(
                            export_path,
                            G,
                            metadata=export_meta,
                            protocols_for_subgraph_statistics=list(_PROTOCOLS),
                        )
                        bundle = load_graph_export(export_path)
                        G_loaded = graph_from_export(bundle)
                        assert G_loaded.number_of_nodes() == G.number_of_nodes()
                        assert G_loaded.number_of_edges() == G.number_of_edges()
                        print(f"Exported and reloaded graph from {export_path}")

                        figure_dir = repo_root / "figure" / export_dir.name
                        figure_dir.mkdir(parents=True, exist_ok=True)
                        figure_path = figure_dir / f"{export_path.stem}.png"
                        plot_geometric_rgg(G, save_path=figure_path)
                        print(f"Saved figure to {figure_path}")

                        print_graph_stats_bundle(graph_stats(G), heading="Full multigraph (all protocols)")
                        for proto in _PROTOCOLS:
                            print_graph_stats_bundle(
                                graph_stats(subgraph_by_protocol(G, proto)),
                                heading=f"Subgraph: {proto}",
                            )
