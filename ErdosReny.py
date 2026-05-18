from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx


def random_gnm(n: int, m: int, seed: Optional[int] = None) -> nx.Graph:
    """Erdős–Rényi G(n, m): n nodes, exactly m edges chosen uniformly at random."""
    return nx.gnm_random_graph(n, m, seed=seed)


def random_gnp(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
    """Erdős–Rényi G(n, p): each of the n choose 2 possible edges appears independently with probability p."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


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


def draw_graph_spring_3d(G: nx.Graph, seed: Optional[int] = None) -> None:
    """Spring layout in 3D (matplotlib); use nx.spring_layout(..., dim=3) and draw edges/nodes manually."""
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
    n = 30
    m = 20  # for random_gnm(n, m, ...)
    p = 0.3
    seed = 44

    gnp = random_gnp(n, p, seed=seed)
    gnm = random_gnm(n, m, seed=seed)

    # print_graph_stats(G, f"G(n,p): n={n}, p={p}, |E|={G.number_of_edges()}")

    pos = nx.spring_layout(gnp, seed=seed)
    nx.draw(gnp, pos=pos)
    plt.show()

    pos = nx.spring_layout(gnm, seed=seed)
    nx.draw(gnm, pos=pos)
    plt.show()


    draw_graph_spring_3d(gnp, seed=seed)
    draw_graph_spring_3d(gnm, seed=seed)


    # draw_graph_spring_3d(G, seed=seed)
