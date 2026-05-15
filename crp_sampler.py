import math
import random
from typing import Optional, Sequence


class CRPSampler:
    """Namespace for CRP sampling; every stochastic method takes ``rnd`` explicitly.

    Calling code passes a ``random.Random`` (or compatible) so runs are reproducible
    without touching the ``random`` module global state.
    """

    @staticmethod
    def sample_labels(
        rnd: random.Random,
        n: int,
        catalog: Sequence[str],
        alpha: float,
        initial_labels: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Return one label per entity, length ``n``, in entity index order
        ``0 .. n-1``.

        Each label is either drawn from ``catalog`` (new atom) or sampled from
        the history ``tape`` (reuse), with probability depending on ``alpha``
        and ``len(tape)`` — classic rich-get-richer clustering.

        Parameters
        ----------
        rnd :
            RNG for all random draws in this call.
        n :
            Number of entities; output has length ``n``.
        catalog :
            Finite support for "new" tag draws; each label is ``rnd.choice``.
        alpha :
            Concentration: larger values favor new tags from ``catalog`` vs reuse.
        initial_labels :
            Optional prefix; length must be ``<= n``. First ``len(initial_labels)``
            entries are fixed; remaining entities are sampled CRP-style.

        Returns
        -------
        list[str]
            One string per entity index ``0 .. n-1``.
        """
        labels: list[str] = []
        tape: list[str] = []

        if initial_labels is not None:
            if len(initial_labels) > n:
                raise ValueError(
                    f"initial_labels has length {len(initial_labels)} but n is {n}"
                )
            for label in initial_labels:
                labels.append(label)
                tape.append(label)

        for _ in range(len(labels), n):
            tag = CRPSampler._draw_tag(rnd, catalog, tape, alpha)
            tape.append(tag)
            labels.append(tag)

        return labels

    @staticmethod
    def sample_profiles(
        rnd: random.Random,
        n: int,
        catalog: Sequence[str],
        alpha_outer: float,
        alpha_inner: float,
        mean_tags: float,
        initial_profiles: Optional[Sequence[Sequence[str]]] = None,
    ) -> list[list[str]]:
        """Return a multiset (unique tags per list) for each entity — nested CRP.

        **Outer:** with ``m`` profiles built so far, the next profile is built
        from scratch with probability ``alpha_outer / (alpha_outer + m)``;
        otherwise it is an exact copy of a uniformly chosen existing profile.

        **Inner:** for each "new" profile, draw ``Poisson(mean_tags)`` tags.
        Each tag is new from ``catalog`` or reused from the growing ``tape``
        with probability ``alpha_inner / (alpha_inner + len(tape))`` before the
        draw. Duplicates on the tape increase reuse weight; each profile list
        keeps tags unique (order preserved).

        Parameters
        ----------
        rnd :
            RNG for all random draws in this call.
        n :
            Total number of entities; ``len(return) == n``.
        catalog :
            Labels for fresh inner draws; ``rnd.choice`` when sampling "new".
        alpha_outer :
            Outer concentration — how often a wholly new profile is invented
            vs cloning an existing one.
        alpha_inner :
            Inner concentration — how often inner tags are new from ``catalog``
            vs reused from ``tape``.
        mean_tags :
            Poisson mean for how many inner draws run for each *new* profile
            (not used when the outer step copies an existing profile).
        initial_profiles :
            Optional fixed rows; ``len(initial_profiles) <= n``. Row ``i`` seeds
            entity ``i``; its tags extend ``tape``. Remaining entities are
            sampled with the nested rules above.

        Returns
        -------
        list[list[str]]
            One list of unique tags per entity index ``0 .. n-1``.
        """
        profiles: list[list[str]] = []
        tape: list[str] = []

        if initial_profiles is not None:
            if len(initial_profiles) > n:
                raise ValueError(
                    f"initial_profiles has length {len(initial_profiles)} but n is {n}"
                )
            for row in initial_profiles:
                profile = list(dict.fromkeys(row))
                profiles.append(profile)
                tape.extend(profile)

        for _ in range(len(profiles), n):
            m = len(profiles)
            if m == 0 or rnd.random() < alpha_outer / (alpha_outer + m):
                profile = CRPSampler._new_profile(
                    rnd, catalog, tape, alpha_inner, mean_tags
                )
            else:
                profile = list(rnd.choice(profiles))
            profiles.append(profile)

        return profiles

    @staticmethod
    def _new_profile(
        rnd: random.Random,
        catalog: Sequence[str],
        tape: list[str],
        alpha_inner: float,
        mean_tags: float,
    ) -> list[str]:
        profile: list[str] = []
        for _ in range(_poisson(rnd, mean_tags)):
            tag = CRPSampler._draw_tag(rnd, catalog, tape, alpha_inner)
            tape.append(tag)
            if tag not in profile:
                profile.append(tag)
        return profile

    @staticmethod
    def _draw_tag(
        rnd: random.Random, catalog: Sequence[str], tape: list[str], alpha: float
    ) -> str:
        if not tape:
            return rnd.choice(catalog)
        if rnd.random() < alpha / (alpha + len(tape)):
            return rnd.choice(catalog)
        return rnd.choice(tape)


def _poisson(rnd: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while p > L:
        k += 1
        p *= rnd.random()
    return k - 1
