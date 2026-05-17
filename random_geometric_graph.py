"""Minimal directed RGG + parallel offset arrows and labels on each offset segment."""

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import networkx as nx

from crp_sampler import CRPSampler
from geometric_multidigraph_config import (
    BetaPositions,
    GaussianPositions,
    GeometricMultidigraphConfig,
    GridPositions,
    POSITION_DISTRIBUTIONS as _CONFIG_POSITION_DISTRIBUTIONS,
    ProfileSamplerConfig,
    UniformPositions,
)
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
from geometric_multidigraph_plot import PER_PROTOCOL_FIGURE, plot_geometric_rgg

# -----------------------------------------------------------------------------
# Protocol catalog (user-defined per run; this is only the batch/script default)
# -----------------------------------------------------------------------------

DEFAULT_PROTOCOLS: tuple[str, ...] = (
    "protocol_1",
    "protocol_2",
    "protocol_3",
    "protocol_4",
    "protocol_5",
    "protocol_6",
)

# Back-compat alias for scripts that import _PROTOCOLS from this module.
_PROTOCOLS = DEFAULT_PROTOCOLS

# Re-export config + plot API from this module for convenience.
__all__ = [
    "DEFAULT_PROTOCOLS",
    "POSITION_DISTRIBUTIONS",
    "BetaPositions",
    "GaussianPositions",
    "GeometricMultidigraphConfig",
    "GeometricMultidigraphResult",
    "GridPositions",
    "ProfileSamplerConfig",
    "UniformPositions",
    "export_metadata",
    "generate",
    "geometric_multidigraph_network",
    "plot_geometric_rgg",
    "sample_positions",
    "add_service_edges",
    "PER_PROTOCOL_FIGURE",
]


# Plotting lives in geometric_multidigraph_plot (re-exported above as plot_geometric_rgg).

# =============================================================================
# GRAPH GENERATION (low-level; prefer generate() for full runs)
# =============================================================================
#
#   Build a directed graph from an undirected random geometric graph: copy node
#   positions, then orient edges (one or both directions) and attach protocols.
# =============================================================================

# Kept in sync with geometric_multidigraph_config.POSITION_DISTRIBUTIONS.
POSITION_DISTRIBUTIONS = _CONFIG_POSITION_DISTRIBUTIONS


def _uniform_in_triangle(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    rnd: random.Random,
) -> tuple[float, float]:
    u, v = rnd.random(), rnd.random()
    if u + v > 1.0:
        u, v = 1.0 - u, 1.0 - v
    return (
        (1.0 - u - v) * ax + u * bx + v * cx,
        (1.0 - u - v) * ay + u * by + v * cy,
    )


def _uniform_in_regular_polygon(
    cx: float,
    cy: float,
    radius: float,
    sides: int,
    rnd: random.Random,
) -> tuple[float, float]:
    phase = rnd.uniform(0, 2 * math.pi)
    verts = [
        (
            cx + radius * math.cos(phase + 2 * math.pi * i / sides),
            cy + radius * math.sin(phase + 2 * math.pi * i / sides),
        )
        for i in range(sides)
    ]
    areas = [
        abs((x0 - cx) * (y1 - cy) - (x1 - cx) * (y0 - cy)) / 2
        for (x0, y0), (x1, y1) in zip(verts, verts[1:] + verts[:1])
    ]
    total = sum(areas)
    pick = rnd.uniform(0, total)
    acc = 0.0
    for i, area in enumerate(areas):
        acc += area
        if pick <= acc:
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % sides]
            return _uniform_in_triangle(cx, cy, x0, y0, x1, y1, rnd)
    x0, y0 = verts[-1]
    x1, y1 = verts[0]
    return _uniform_in_triangle(cx, cy, x0, y0, x1, y1, rnd)


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
    - disk: ``cx``, ``cy``, ``radius`` (uniform area on the disk)
    - annulus: ``cx``, ``cy``, ``r_inner``, ``r_outer`` (uniform area on the ring)
    - polygon: ``sides``, ``cx``, ``cy``, ``radius`` (regular n-gon, n >= 3)
    - arc: ``cx``, ``cy``, ``radius``, ``angle_start_deg``, ``angle_end_deg``, ``jitter``
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

        case "disk":
            cx = kwargs["cx"]
            cy = kwargs["cy"]
            radius = kwargs["radius"]
            pos = {}
            for i in range(n):
                t = rnd.uniform(0, 2 * math.pi)
                r = radius * math.sqrt(rnd.random())
                pos[i] = (cx + r * math.cos(t), cy + r * math.sin(t))

        case "annulus":
            cx = kwargs["cx"]
            cy = kwargs["cy"]
            r_inner = kwargs["r_inner"]
            r_outer = kwargs["r_outer"]
            pos = {}
            for i in range(n):
                t = rnd.uniform(0, 2 * math.pi)
                r = math.sqrt(
                    rnd.uniform(r_inner * r_inner, r_outer * r_outer)
                )
                pos[i] = (cx + r * math.cos(t), cy + r * math.sin(t))

        case "polygon":
            sides = int(kwargs["sides"])
            cx = kwargs["cx"]
            cy = kwargs["cy"]
            radius = kwargs["radius"]
            if sides < 3:
                raise ValueError(f"polygon sides must be >= 3, got {sides}")
            pos = {
                i: _uniform_in_regular_polygon(cx, cy, radius, sides, rnd)
                for i in range(n)
            }

        case "arc":
            cx = kwargs["cx"]
            cy = kwargs["cy"]
            radius = kwargs["radius"]
            a0 = math.radians(kwargs["angle_start_deg"])
            a1 = math.radians(kwargs["angle_end_deg"])
            jitter = kwargs["jitter"]
            pos = {}
            for i in range(n):
                t = rnd.uniform(a0, a1)
                x = cx + radius * math.cos(t)
                y = cy + radius * math.sin(t)
                if jitter > 0:
                    nx_n = -math.sin(t)
                    ny_n = math.cos(t)
                    off = rnd.uniform(-jitter, jitter)
                    x += off * nx_n
                    y += off * ny_n
                pos[i] = (x, y)

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
# CONFIG-DRIVEN GENERATION
#
#   Use GeometricMultidigraphConfig (in geometric_multidigraph_config) with
#   generate() below instead of calling the low-level steps manually.
# =============================================================================


@dataclass
class GeometricMultidigraphResult:
    """Output of :func:`generate` — graph plus inputs used to build it."""

    graph: nx.MultiDiGraph
    config: GeometricMultidigraphConfig
    pos: dict[int, tuple[float, float]]
    profiles: list[list[str]]
    position_rng: random.Random
    profile_rng: random.Random

    @property
    def metadata(self) -> dict[str, Any]:
        """Export-oriented metadata (same shape as batch ``__main__`` writes)."""
        return export_metadata(self)

    @property
    def catalog(self) -> list[str]:
        return list(self.config.protocols)


def export_metadata(result: GeometricMultidigraphResult) -> dict[str, Any]:
    """Metadata dict for :func:`graph_export.write_graph_export`."""
    cfg = result.config
    pos_cfg = cfg.positions
    prof_cfg = cfg.profiles
    catalog = list(cfg.protocols)
    G = result.graph

    return {
        "model": "geometric_multidigraph",
        "n": cfg.n,
        "position_seed": cfg.position_seed,
        "profile_seed": cfg.profile_seed,
        "radius": cfg.radius,
        "initial_profiles": prof_cfg.initial_profiles,
        "profiles_sampler": {
            "qualified_name": (
                f"{CRPSampler.sample_profiles.__module__}."
                f"{CRPSampler.sample_profiles.__qualname__}"
            ),
            "inputs": {
                "n": cfg.n,
                "catalog": catalog,
                "alpha_outer": prof_cfg.alpha_outer,
                "alpha_inner": prof_cfg.alpha_inner,
                "mean_tags": prof_cfg.mean_tags,
                "initial_profiles": prof_cfg.initial_profiles,
            },
        },
        "positions_sampler": {
            "qualified_name": (
                f"{sample_positions.__module__}.{sample_positions.__qualname__}"
            ),
            "inputs": {
                "n": cfg.n,
                "distribution": pos_cfg.distribution,
                **pos_cfg.sampler_kwargs(),
            },
        },
        "protocol_edge_counts": protocol_edge_counts(G, protocols=catalog),
        "protocol_profile_node_counts": protocol_profile_node_counts(
            G, protocols=catalog
        ),
    }


def generate(
    config: GeometricMultidigraphConfig,
    *,
    rnd: Optional[random.Random] = None,
) -> GeometricMultidigraphResult:
    """Sample profiles and positions, then build the geometric multidigraph.

    Parameters
    ----------
    config :
        Full run specification; ``config.validate()`` is called automatically.
    rnd :
        If given, both position and profile sampling use this RNG (coupled
        behavior). Otherwise ``config.position_seed`` and ``config.profile_seed``
        define independent streams.
    """
    config.validate()
    if rnd is not None:
        position_rng = profile_rng = rnd
    else:
        position_rng = random.Random(config.position_seed)
        profile_rng = random.Random(config.profile_seed)

    prof_cfg = config.profiles
    catalog = list(config.protocols)

    pos_cfg = config.positions
    pos = sample_positions(
        config.n,
        position_rng,
        pos_cfg.distribution,
        **pos_cfg.sampler_kwargs(),
    )

    profiles = CRPSampler.sample_profiles(
        profile_rng,
        config.n,
        catalog,
        alpha_outer=prof_cfg.alpha_outer,
        alpha_inner=prof_cfg.alpha_inner,
        mean_tags=prof_cfg.mean_tags,
        initial_profiles=prof_cfg.initial_profiles,
    )

    graph = geometric_multidigraph_network(
        radius=config.radius,
        pos=pos,
        profiles=profiles,
    )

    return GeometricMultidigraphResult(
        graph=graph,
        config=config,
        pos=pos,
        profiles=profiles,
        position_rng=position_rng,
        profile_rng=profile_rng,
    )


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == "__main__":
    # Batch sweep: build a GeometricMultidigraphConfig per parameter combo, then generate once.
    position_seeds = 20
    profile_seeds = 20
    catalog = list(DEFAULT_PROTOCOLS)

    for pos_seed in range(position_seeds):
        for prof_seed in range(profile_seeds):
            for alpha_outer in [2, 4, 8]:
                for alpha_inner in [2, 4, 8]:
                    for mean_tags in [2, 4, 6]:
                        for radius in [0.15, 0.25, 0.35]:
                            config = GeometricMultidigraphConfig(
                                protocols=catalog,
                                position_seed=pos_seed,
                                profile_seed=prof_seed,
                                n=30,
                                radius=radius,
                                positions=UniformPositions(x0=0, x1=1, y0=0, y1=1),
                                profiles=ProfileSamplerConfig(
                                    alpha_outer=alpha_outer,
                                    alpha_inner=alpha_inner,
                                    mean_tags=mean_tags,
                                    initial_profiles=None,
                                ),
                            )
                            result = generate(config)
                            G = result.graph
                            export_meta = result.metadata

                            repo_root = Path(__file__).resolve().parent
                            export_dir = ensure_export_subdirectory_for_metadata(
                                repo_root, export_meta
                            )
                            export_path = (
                                export_dir
                                / f"rgg_with_stats_pos{export_meta['position_seed']}_prof{export_meta['profile_seed']}.json"
                            )

                            write_graph_export(
                                export_path,
                                G,
                                metadata=export_meta,
                                protocols_for_subgraph_statistics=catalog,
                            )
                            bundle = load_graph_export(export_path)
                            G_loaded = graph_from_export(bundle)
                            assert G_loaded.number_of_nodes() == G.number_of_nodes()
                            assert G_loaded.number_of_edges() == G.number_of_edges()
                            print(f"Exported and reloaded graph from {export_path}")


                            if pos_seed < 20 and prof_seed < 20:
                                figure_dir = repo_root / "figure" / export_dir.name
                                figure_dir.mkdir(parents=True, exist_ok=True)
                                figure_path = figure_dir / f"{export_path.stem}.png"
                                plot_geometric_rgg(
                                    G, save_path=figure_path, proto_order=catalog
                                )
                                print(f"Saved figure to {figure_path}")

                                print_graph_stats_bundle(
                                    graph_stats(G), heading="Full multigraph (all protocols)"
                                )
                                for proto in catalog:
                                    print_graph_stats_bundle(
                                        graph_stats(subgraph_by_protocol(G, proto)),
                                        heading=f"Subgraph: {proto}",
                                    )
