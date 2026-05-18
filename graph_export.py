"""Serialize NetworkX graphs to JSON alongside summary statistics."""

from __future__ import annotations

import json
import math
import numbers
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import networkx as nx
from networkx.readwrite import json_graph


FORMAT_VERSION = 2


def _percentile_linear(xs_sorted: list[float], p: float) -> float:
    """``p`` in [0, 100]; linear interpolation on sorted values (NumPy-style)."""
    n = len(xs_sorted)
    if n == 1:
        return xs_sorted[0]
    if p <= 0:
        return xs_sorted[0]
    if p >= 100:
        return xs_sorted[-1]
    k = (n - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs_sorted[lo]
    return xs_sorted[lo] * (hi - k) + xs_sorted[hi] * (k - lo)


def _tail_mean_fraction(xs_sorted: list[float], frac: float, *, lower: bool) -> float:
    """Mean of the smallest (``lower``) or largest ``ceil(frac * n)`` observations."""
    n = len(xs_sorted)
    k = max(1, math.ceil(n * frac))
    chunk = xs_sorted[:k] if lower else xs_sorted[-k:]
    return statistics.fmean(chunk)


def _node_metric_discrete_stats(metric: dict[Any, float] | None) -> dict[str, Any] | None:
    """Summarize a node → float map (mean, median, spread, tails, CVaR-style means)."""
    if metric is None:
        return None
    values = list(metric.values())
    n = len(values)
    if n == 0:
        return {"count": 0}
    xs = sorted(values)
    out: dict[str, Any] = {
        "count": n,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": xs[0],
        "max": xs[-1],
    }
    out["stdev"] = statistics.stdev(values) if n > 1 else 0.0
    out["p90"] = _percentile_linear(xs, 90.0)
    tail_frac = 0.10
    out["cvar_lower_90"] = _tail_mean_fraction(xs, tail_frac, lower=True)
    out["cvar_upper_90"] = _tail_mean_fraction(xs, tail_frac, lower=False)
    if n >= 4:
        q1, q2, q3 = statistics.quantiles(values, n=4, method="inclusive")
        out["q1"] = q1
        out["q2"] = q2
        out["q3"] = q3
    return out


def graph_stats(G: nx.Graph) -> dict:
    """Return graph scalars, per-node centralities, and discrete summaries of each.

    Keys: ``graph_stats``, ``node_stats``, ``node_stats_discrete``.
    """
    graph_part: dict = {
        "number_of_nodes": G.number_of_nodes(),
        "number_of_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed_acyclic_graph": nx.is_directed_acyclic_graph(G),
        "number_of_strongly_connected_components": (
            nx.number_strongly_connected_components(G)
        ),
        "number_of_weakly_connected_components": (
            nx.number_weakly_connected_components(G)
        ),
    }
    node_part: dict = {
        "degree_centrality": nx.degree_centrality(G),
        "betweenness_centrality": nx.betweenness_centrality(G),
        "closeness_centrality": nx.closeness_centrality(G),
    }

    discrete_part = {
        name: _node_metric_discrete_stats(val) for name, val in node_part.items()
    }
    return {
        "graph_stats": graph_part,
        "node_stats": node_part,
        "node_stats_discrete": discrete_part,
    }


def print_graph_stats_bundle(stats: dict, *, heading: str = "Graph stats") -> None:
    """Print ``graph_stats`` output with clear sections (no huge per-node dict dumps)."""
    bar = "-" * max(24, len(heading) + 8)
    print(f"\n{bar}\n  {heading}\n{bar}")

    print("  graph_stats")
    for key, val in stats["graph_stats"].items():
        print(f"    {key:45} {val}")

    print("  node_stats_discrete")
    for metric, summary in stats["node_stats_discrete"].items():
        print(f"    {metric}")
        if summary is None:
            print("      (none)")
            continue
        for sk, sv in summary.items():
            if isinstance(sv, float):
                print(f"      {sk:12} {sv:.6g}")
            else:
                print(f"      {sk:12} {sv}")

    print("    (per-node maps: stats['node_stats'][...] — omitted)")


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert objects to plain JSON-compatible Python types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {_to_jsonable(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, numbers.Integral) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, numbers.Real):
        out = float(obj)
        return out if math.isfinite(out) else None
    return obj


def graph_to_node_link_dict(G: nx.Graph) -> dict[str, Any]:
    """Standard NetworkX node-link serialization (loads with ``node_link_graph``)."""
    return json_graph.node_link_data(G)


def subgraph_by_protocol(G: nx.Graph, proto: str) -> nx.Graph:
    """Directed (multi) subgraph of edges whose ``protocol`` attribute equals ``proto``.

    Preserves parallel edges when ``G`` is a ``MultiDiGraph``; yields a plain
    ``DiGraph`` when ``G`` has no parallel edges for that protocol.
    """
    if G.is_directed():
        H = nx.MultiDiGraph() if G.is_multigraph() else nx.DiGraph()
    elif G.is_multigraph():
        H = nx.MultiGraph()
    else:
        H = nx.Graph()

    H.add_nodes_from(G.nodes(data=True))
    if G.is_multigraph():
        for u, v, key, d in G.edges(keys=True, data=True):
            if d.get("protocol") == proto:
                H.add_edge(u, v, key=key, **d)
    else:
        for u, v, d in G.edges(data=True):
            if d.get("protocol") == proto:
                H.add_edge(u, v, **d)

    H.graph.update(G.graph)
    return H


def protocol_edge_counts(
    G: nx.Graph,
    *,
    protocols: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """Count directed edges (including parallel) by ``data['protocol']``.

    If ``protocols`` is set, every name appears with a count (possibly 0), in
    that order, followed by any other protocol keys on edges (sorted).
    """
    c: Counter[str] = Counter()
    if G.is_multigraph():
        for _u, _v, _key, d in G.edges(keys=True, data=True):
            p = d.get("protocol")
            if p is not None:
                c[str(p)] += 1
    else:
        for _u, _v, d in G.edges(data=True):
            p = d.get("protocol")
            if p is not None:
                c[str(p)] += 1

    if protocols is None:
        return {p: int(n) for p, n in sorted(c.items(), key=lambda t: t[0])}

    catalog = {str(p) for p in protocols}
    out: dict[str, int] = {}
    for p in protocols:
        ps = str(p)
        out[ps] = int(c[ps])
    for p, n in sorted(c.items(), key=lambda t: t[0]):
        if str(p) not in catalog:
            out[str(p)] = int(n)
    return out


def protocol_profile_node_counts(
    G: nx.Graph,
    *,
    protocols: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """Count nodes whose ``profile`` list contains each protocol tag (unique per node).

    If ``protocols`` is set, every entry appears with a count (possibly 0),
    in that order, then other observed tags sorted.
    """
    tally: Counter[str] = Counter()
    for _nid, attrs in G.nodes(data=True):
        prof = attrs.get("profile") or attrs.get("profiles")
        if not prof:
            continue
        tags = prof if isinstance(prof, list) else list(prof)
        for t in frozenset(str(x) for x in tags):
            tally[t] += 1

    if protocols is None:
        return {p: int(n) for p, n in sorted(tally.items(), key=lambda t: t[0])}

    catalog = {str(p) for p in protocols}
    out: dict[str, int] = {}
    for p in protocols:
        ps = str(p)
        out[ps] = int(tally[ps])
    for p, n in sorted(tally.items(), key=lambda t: t[0]):
        if str(p) not in catalog:
            out[str(p)] = int(n)
    return out


_BAD_PATH_CHARS = '<>:"/\\|?*\n\r\t'


def _path_num_slug_val(v: Any) -> str:
    return str(v).replace(".", "p").replace("-", "m")


def export_run_folder_name(metadata: dict[str, Any]) -> str:
    """Build one ``exports/…`` directory name from export metadata.

    Uses ``model``, ``n``, ``radius``, ``positions_sampler.inputs``,
    ``profiles_sampler.inputs``, and counts labels in ``initial_profiles``.
    Seeds are **not** in the folder name (use per-file names like
    ``rgg_with_stats_pos0_prof1.json`` inside the folder).

    Caller should store full parameters in JSON; this string groups runs that share
    the same config except seeds.
    """
    pos_i = metadata["positions_sampler"]["inputs"]
    prof_i = metadata["profiles_sampler"]["inputs"]
    raw_model = str(metadata["model"]).strip() or "empty"
    model_slug = "".join(
        "_" if c in _BAD_PATH_CHARS else c for c in raw_model
    ).strip(" .")
    dist_raw = str(pos_i["distribution"]).strip() or "empty"
    dist_slug = "".join(
        "_" if c in _BAD_PATH_CHARS else c for c in dist_raw
    ).strip(" .")
    pos_bbox = "pos_default"
    if all(k in pos_i for k in ("x0", "x1", "y0", "y1")):
        pos_bbox = (
            f"x{_path_num_slug_val(pos_i['x0'])}_{_path_num_slug_val(pos_i['x1'])}"
            f"_y{_path_num_slug_val(pos_i['y0'])}_{_path_num_slug_val(pos_i['y1'])}"
        )
    rad = _path_num_slug_val(metadata["radius"])
    ao = _path_num_slug_val(prof_i["alpha_outer"])
    ai = _path_num_slug_val(prof_i["alpha_inner"])
    ip = metadata.get("initial_profiles") or ()
    initial_label_count = sum(len(seq or ()) for seq in ip)
    run_folder = (
        f"{model_slug}_n{metadata['n']}_r{rad}_{dist_slug}_{pos_bbox}"
        f"_prof_{initial_label_count}_ao{ao}_ai{ai}_mt{prof_i['mean_tags']}"
    )
    return run_folder.strip(" .") or "export_run"


def ensure_export_subdirectory_for_metadata(
    repo_root: str | Path,
    metadata: dict[str, Any],
    *,
    exports_dir_name: str = "exports",
    mkdir: bool = True,
) -> Path:
    """``repo_root`` / ``exports_dir_name`` / slug(metadata); optionally ``mkdir``. Returns folder path."""
    root = Path(repo_root)
    out = root / exports_dir_name / export_run_folder_name(metadata)
    if mkdir:
        out.mkdir(parents=True, exist_ok=True)
    return out


def _statistics_for_export(
    raw: dict[str, Any], *, include_node_statistics_maps: bool
) -> dict[str, Any]:
    if include_node_statistics_maps:
        return raw
    return {
        "graph_stats": raw["graph_stats"],
        "node_stats_discrete": raw["node_stats_discrete"],
    }


def build_export_bundle(
    G: nx.Graph,
    *,
    metadata: Optional[dict[str, Any]] = None,
    include_node_statistics_maps: bool = True,
    protocols_for_subgraph_statistics: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Build export dict: ``graph``, ``statistics`` (+ optional ``statistics_by_protocol``).

    When ``include_node_statistics_maps`` is True, each statistics block includes
    ``node_stats`` (per-node centrality maps). When ``protocols_for_subgraph_statistics``
    is set, the same stats are computed on ``subgraph_by_protocol(G, p)`` for each ``p``.
    """
    stats = graph_stats(G)
    bundle: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "graph": graph_to_node_link_dict(G),
        "statistics": _statistics_for_export(
            stats, include_node_statistics_maps=include_node_statistics_maps
        ),
    }
    if protocols_for_subgraph_statistics:
        by_protocol: dict[str, Any] = {}
        for proto in protocols_for_subgraph_statistics:
            H = subgraph_by_protocol(G, proto)
            sub = graph_stats(H)
            by_protocol[proto] = _statistics_for_export(
                sub, include_node_statistics_maps=include_node_statistics_maps
            )
        bundle["statistics_by_protocol"] = by_protocol
    if metadata:
        bundle["metadata"] = metadata
    return bundle


def write_graph_export(
    path: str | Path,
    G: nx.Graph,
    *,
    metadata: Optional[dict[str, Any]] = None,
    include_node_statistics_maps: bool = True,
    protocols_for_subgraph_statistics: Optional[Sequence[str]] = None,
    indent: int = 2,
) -> Path:
    """Write ``build_export_bundle(...)`` JSON to ``path`` (UTF-8). Returns path."""
    p = Path(path)
    bundle = build_export_bundle(
        G,
        metadata=metadata,
        include_node_statistics_maps=include_node_statistics_maps,
        protocols_for_subgraph_statistics=protocols_for_subgraph_statistics,
    )
    p.write_text(
        json.dumps(_to_jsonable(bundle), indent=indent if indent else None),
        encoding="utf-8",
    )
    return p


def load_graph_export(path: str | Path) -> dict[str, Any]:
    """Read a bundle written by ``write_graph_export``."""
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def graph_from_export(bundle: dict[str, Any]) -> nx.Graph:
    """Rehydrate NetworkX graph from bundle['graph']."""
    graph_payload = bundle.get("graph")
    if not isinstance(graph_payload, dict):
        raise ValueError("bundle missing valid 'graph' node-link dictionary")
    return json_graph.node_link_graph(graph_payload)
