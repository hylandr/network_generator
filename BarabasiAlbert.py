"""Barabási–Albert preferential attachment random graphs via NetworkX."""

from typing import Any, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx


def random_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = None,
    initial_graph: Optional[nx.Graph] = None,
    *,
    create_using: Any = None,
) -> nx.Graph:
    """Return a random Barabási–Albert graph on ``n`` nodes.

    Each arriving node attaches ``m`` edges to existing nodes, biased toward
    high degree (preferential attachment). Requires ``1 <= m < n``.

    Note: ``m`` is *edges per new node*, not the total edge count used in
    Erdős–Rényi G(n, m).

    https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html
    """
    return nx.barabasi_albert_graph(
        n, m, seed=seed, initial_graph=initial_graph, create_using=create_using
    )


def print_graph_stats(G: nx.Graph, heading: Optional[str] = None) -> None:
    """Print per-node degree and clustering, then an adjacency list."""
    if heading is not None:
        print(heading)
        print()
    print("node degree clustering")
    for v in nx.nodes(G):
        print(f"{v} {nx.degree(G, v)} {nx.clustering(G, v)}")
    print()
    print("the adjacency list")
    for line in nx.generate_adjlist(G):
        print(line)


def draw_graph_spring(
    G: nx.Graph,
    seed: Optional[int] = None,
    *,
    figsize: tuple[float, float] = (8.0, 8.0),
    with_labels: bool = True,
    node_size_by_degree: bool = True,
    color_by_degree: bool = True,
) -> None:
    """2D spring layout tuned for reading structure: labels, degree → size and color, soft edges.

    For large ``n``, set ``with_labels=False`` to avoid clutter.
    """
    pos = nx.spring_layout(G, seed=seed)
    degrees = dict(G.degree())
    nodes = list(G.nodes())
    degs = [degrees[v] for v in nodes]
    nmax = max(degs, default=1)
    nmin = min(degs, default=0)

    if node_size_by_degree and len(nodes):
        if nmax == nmin:
            sizes = [400] * len(nodes)
        else:
            sizes = [120 + 900 * (d - nmin) / (nmax - nmin) for d in degs]
    else:
        sizes = 400

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.plasma

    if color_by_degree and len(nodes):
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=(0.45, 0.45, 0.45, 0.45), width=1.0
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            nodelist=nodes,
            node_size=sizes,
            node_color=degs,
            cmap=cmap,
            edgecolors="white",
            linewidths=1.2,
        )
        norm = mcolors.Normalize(vmin=nmin, vmax=max(nmax, 1))
        plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            shrink=0.55,
            label="Degree",
        )
    else:
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=(0.45, 0.45, 0.45, 0.45), width=1.0
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_size=sizes,
            node_color="tab:blue",
            edgecolors="white",
            linewidths=1.2,
        )

    if with_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            ax=ax,
            font_size=max(6, min(11, 140 // max(len(nodes), 1))),
            font_weight="bold",
            font_color="0.1",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.9,
                "linewidth": 0.4,
            },
        )

    ax.set_axis_off()
    ax.set_title(
        "Node size & color = degree (hubs stand out in Barabási–Albert)",
        fontsize=11,
        pad=12,
    )
    fig.tight_layout()
    plt.show()


def draw_graph_spring_3d(G: nx.Graph, seed: Optional[int] = None) -> None:
    """Spring layout in 3D; straight edges between node positions."""
    pos = nx.spring_layout(G, dim=3, seed=seed)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for u, v in G.edges():
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            [pos[u][2], pos[v][2]],
            color="gray",
            linewidth=0.8,
            alpha=0.65,
        )
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    ax.scatter(xs, ys, zs, s=55, c="tab:blue", depthshade=True)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    n = 100
    m = 10 # edges each new node adds (must satisfy 1 <= m < n)
    seed = 44

    G = random_barabasi_albert(n, m, seed=seed)

    print_graph_stats(
        G,
        f"Barabási–Albert: n={n}, m={m}, |E|={G.number_of_edges()}",
    )

    draw_graph_spring(G, seed=seed, with_labels=n <= 50)

    # draw_graph_spring_3d(G, seed=seed)
