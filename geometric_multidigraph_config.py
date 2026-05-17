"""Configuration dataclasses for geometric multidigraph generation.

Build a :class:`GeometricMultidigraphConfig`, then call
:func:`random_geometric_graph.generate` to sample and construct the graph.
Plotting and JSON export are separate modules.

Example::

    from geometric_multidigraph_config import (
        GeometricMultidigraphConfig,
        ProfileSamplerConfig,
        UniformPositions,
    )
    from random_geometric_graph import DEFAULT_PROTOCOLS, generate
    from geometric_multidigraph_plot import plot_geometric_rgg

    catalog = list(DEFAULT_PROTOCOLS)  # or your own names
    config = GeometricMultidigraphConfig(
        protocols=catalog,
        position_seed=0,
        profile_seed=1,
        n=30,
        radius=0.25,
        positions=UniformPositions(x0=0.0, x1=1.0, y0=0.0, y1=1.0),
        profiles=ProfileSamplerConfig(
            alpha_outer=4,
            alpha_inner=8,
            mean_tags=4,
            initial_profiles=None,
        ),
    )
    result = generate(config)
    plot_geometric_rgg(result.graph)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, Union

POSITION_DISTRIBUTIONS: tuple[str, ...] = (
    "uniform",
    "gaussian",
    "beta",
    "grid",
    "disk",
    "annulus",
    "polygon",
    "arc",
)


def _require(body: Mapping[str, Any], key: str) -> Any:
    if key not in body:
        raise ValueError(f"{key!r} is required")
    return body[key]


# -----------------------------------------------------------------------------
# Node position sampling — one dataclass per distribution (no **kwargs)
# -----------------------------------------------------------------------------


@dataclass
class UniformPositions:
    """Flat density on the axis-aligned box [x0, x1] × [y0, y1]."""

    x0: float
    x1: float
    y0: float
    y1: float

    @property
    def distribution(self) -> Literal["uniform"]:
        return "uniform"

    def sampler_kwargs(self) -> dict[str, float]:
        return {"x0": self.x0, "x1": self.x1, "y0": self.y0, "y1": self.y1}


@dataclass
class GaussianPositions:
    """Bell cluster around (mu_x, mu_y). Coordinates are not clamped to [0, 1]."""

    mu_x: float
    mu_y: float
    sigma: float

    @property
    def distribution(self) -> Literal["gaussian"]:
        return "gaussian"

    def sampler_kwargs(self) -> dict[str, float]:
        return {"mu_x": self.mu_x, "mu_y": self.mu_y, "sigma": self.sigma}


@dataclass
class BetaPositions:
    """Beta-distributed coordinates on [0, 1]² (independent per axis)."""

    alpha: float
    beta: float

    @property
    def distribution(self) -> Literal["beta"]:
        return "beta"

    def sampler_kwargs(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta}


@dataclass
class GridPositions:
    """Regular lattice with uniform jitter; jitter=0 gives a perfect grid."""

    jitter: float
    # None → ceil(sqrt(n)) inside sample_positions
    cols: Optional[int]

    @property
    def distribution(self) -> Literal["grid"]:
        return "grid"

    def sampler_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"jitter": self.jitter}
        if self.cols is not None:
            kw["cols"] = self.cols
        return kw


@dataclass
class DiskPositions:
    """Uniform area on a disk centered at (cx, cy)."""

    cx: float
    cy: float
    radius: float

    @property
    def distribution(self) -> Literal["disk"]:
        return "disk"

    def sampler_kwargs(self) -> dict[str, float]:
        return {"cx": self.cx, "cy": self.cy, "radius": self.radius}


@dataclass
class AnnulusPositions:
    """Uniform area on a ring: r_inner <= r <= r_outer around (cx, cy)."""

    cx: float
    cy: float
    r_inner: float
    r_outer: float

    @property
    def distribution(self) -> Literal["annulus"]:
        return "annulus"

    def sampler_kwargs(self) -> dict[str, float]:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "r_inner": self.r_inner,
            "r_outer": self.r_outer,
        }


@dataclass
class PolygonPositions:
    """Uniform area inside a regular n-gon (circumradius ``radius``)."""

    sides: int
    cx: float
    cy: float
    radius: float

    @property
    def distribution(self) -> Literal["polygon"]:
        return "polygon"

    def sampler_kwargs(self) -> dict[str, float | int]:
        return {
            "sides": self.sides,
            "cx": self.cx,
            "cy": self.cy,
            "radius": self.radius,
        }


@dataclass
class ArcPositions:
    """Points on a circular arc; optional normal-direction ``jitter``."""

    cx: float
    cy: float
    radius: float
    angle_start_deg: float
    angle_end_deg: float
    jitter: float

    @property
    def distribution(self) -> Literal["arc"]:
        return "arc"

    def sampler_kwargs(self) -> dict[str, float]:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "radius": self.radius,
            "angle_start_deg": self.angle_start_deg,
            "angle_end_deg": self.angle_end_deg,
            "jitter": self.jitter,
        }


PositionConfig = Union[
    UniformPositions,
    GaussianPositions,
    BetaPositions,
    GridPositions,
    DiskPositions,
    AnnulusPositions,
    PolygonPositions,
    ArcPositions,
]


# -----------------------------------------------------------------------------
# CRP profile sampling
# -----------------------------------------------------------------------------


@dataclass
class ProfileSamplerConfig:
    """Nested CRP parameters for per-node protocol lists (service bundles)."""

    alpha_outer: float
    alpha_inner: float
    mean_tags: float
    # None or [] → no seeded rows; [["protocol_1", ...]] fixes node 0, etc.
    initial_profiles: Optional[Sequence[Sequence[str]]]


# -----------------------------------------------------------------------------
# Full model run
# -----------------------------------------------------------------------------


@dataclass
class GeometricMultidigraphConfig:
    """Everything needed to build one geometric multidigraph instance."""

    # User-supplied protocol names (CRP catalog and edge labels); no default here.
    protocols: Sequence[str]
    position_seed: int
    profile_seed: int
    n: int
    radius: float
    positions: PositionConfig
    profiles: ProfileSamplerConfig

    def validate(self) -> None:
        if not self.protocols:
            raise ValueError("protocols must be a non-empty sequence of names")
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        dist = self.positions.distribution
        if dist not in POSITION_DISTRIBUTIONS:
            raise ValueError(
                f"unknown position distribution {dist!r}; "
                f"choose from {POSITION_DISTRIBUTIONS}"
            )
        pos = self.positions
        if isinstance(pos, DiskPositions) and pos.radius <= 0:
            raise ValueError(f"disk radius must be positive, got {pos.radius}")
        if isinstance(pos, AnnulusPositions):
            if pos.r_inner < 0 or pos.r_outer <= 0:
                raise ValueError("annulus radii must be non-negative with r_outer > 0")
            if pos.r_inner >= pos.r_outer:
                raise ValueError(
                    f"annulus r_inner must be < r_outer, got {pos.r_inner} >= {pos.r_outer}"
                )
        if isinstance(pos, PolygonPositions):
            if pos.sides < 3:
                raise ValueError(f"polygon sides must be >= 3, got {pos.sides}")
            if pos.radius <= 0:
                raise ValueError(f"polygon radius must be positive, got {pos.radius}")
        if isinstance(pos, ArcPositions) and pos.radius <= 0:
            raise ValueError(f"arc radius must be positive, got {pos.radius}")
        ip = self.profiles.initial_profiles
        if ip is not None and len(ip) > self.n:
            raise ValueError(
                f"initial_profiles has length {len(ip)} but n is {self.n}"
            )

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> GeometricMultidigraphConfig:
        """Build config from a flat dict (e.g. web UI JSON or script kwargs).

        Every field must be present; missing keys raise :class:`ValueError`.
        """
        dist = str(_require(body, "pos_distribution"))
        if dist == "uniform":
            positions: PositionConfig = UniformPositions(
                x0=float(_require(body, "x0")),
                x1=float(_require(body, "x1")),
                y0=float(_require(body, "y0")),
                y1=float(_require(body, "y1")),
            )
        elif dist == "gaussian":
            positions = GaussianPositions(
                mu_x=float(_require(body, "mu_x")),
                mu_y=float(_require(body, "mu_y")),
                sigma=float(_require(body, "sigma")),
            )
        elif dist == "beta":
            positions = BetaPositions(
                alpha=float(_require(body, "alpha")),
                beta=float(_require(body, "beta")),
            )
        elif dist == "grid":
            cols_raw = _require(body, "cols")
            positions = GridPositions(
                jitter=float(_require(body, "jitter")),
                cols=None if cols_raw is None else int(cols_raw),
            )
        elif dist == "disk":
            positions = DiskPositions(
                cx=float(_require(body, "cx")),
                cy=float(_require(body, "cy")),
                radius=float(_require(body, "disk_radius")),
            )
        elif dist == "annulus":
            positions = AnnulusPositions(
                cx=float(_require(body, "cx")),
                cy=float(_require(body, "cy")),
                r_inner=float(_require(body, "r_inner")),
                r_outer=float(_require(body, "r_outer")),
            )
        elif dist == "polygon":
            positions = PolygonPositions(
                sides=int(_require(body, "sides")),
                cx=float(_require(body, "cx")),
                cy=float(_require(body, "cy")),
                radius=float(_require(body, "poly_radius")),
            )
        elif dist == "arc":
            positions = ArcPositions(
                cx=float(_require(body, "cx")),
                cy=float(_require(body, "cy")),
                radius=float(_require(body, "arc_radius")),
                angle_start_deg=float(_require(body, "angle_start_deg")),
                angle_end_deg=float(_require(body, "angle_end_deg")),
                jitter=float(_require(body, "jitter")),
            )
        else:
            raise ValueError(
                f"pos_distribution must be one of {POSITION_DISTRIBUTIONS}, got {dist!r}"
            )

        raw_ip = _require(body, "initial_profiles")
        profiles = ProfileSamplerConfig(
            alpha_outer=float(_require(body, "alpha_outer")),
            alpha_inner=float(_require(body, "alpha_inner")),
            mean_tags=float(_require(body, "mean_tags")),
            initial_profiles=None if raw_ip is None else list(raw_ip),
        )

        return cls(
            protocols=list(_require(body, "protocols")),
            position_seed=int(_require(body, "position_seed")),
            profile_seed=int(_require(body, "profile_seed")),
            n=int(_require(body, "n")),
            radius=float(_require(body, "radius")),
            positions=positions,
            profiles=profiles,
        )


def position_config_asdict(cfg: PositionConfig) -> dict[str, Any]:
    """JSON-friendly position config (includes ``distribution`` key)."""
    d = asdict(cfg)
    d["distribution"] = cfg.distribution
    return d
