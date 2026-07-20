# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Leaderboard module - Manages candidate solutions organized by idea/approach.

Candidates are grouped into idea clusters assigned by the manager LLM.
Each cluster corresponds to one distinct algorithmic approach explored per generation.

Each candidate may have multiple (metric, complexity) evaluation points
produced by post-process hyperparameter tuning.  The global Pareto front
is computed over *all* such points across every candidate.
"""

import json
import math
import random
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


_VERDICT_YES_RE = re.compile(
    r"^\s*Followed\s+assigned\s+approach\s*:\s*yes\b",
    re.IGNORECASE,
)


def _verdict_is_yes(summary: str) -> bool:
    """True iff the summary starts with the 'yes' verdict prefix."""
    if not summary:
        return False
    return bool(_VERDICT_YES_RE.match(summary))


def _is_better(a: float, b: float, higher_is_better: bool) -> bool:
    return a > b if higher_is_better else a < b


def _best_hp(hp_results: list, higher_is_better: bool):
    """Return the HPResult with the best metric in the given direction."""
    return max(
        hp_results,
        key=lambda hp: hp.metric if higher_is_better else -hp.metric,
    )


@dataclass
class HPResult:
    """A single evaluation point for a specific hyperparameter configuration.

    The ``params`` dict holds the actual hyperparameter values used for
    this point.  An empty dict means "baseline" (coded defaults in
    ``solution.py`` with no overrides).
    """
    hp_index: int
    metric: float
    complexity: float
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HPResult":
        return cls(
            hp_index=int(data["hp_index"]),
            metric=float(data["metric"]),
            complexity=float(data["complexity"]),
            params=dict(data.get("params", {})),
        )


@dataclass
class Candidate:
    """A candidate solution with code and cluster information."""
    workspace_id: str
    generation: int
    code: str = ""
    summary: str = ""
    cluster: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None
    hp_results: list[HPResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["hp_results"] = [hp.to_dict() for hp in self.hp_results]
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Candidate":
        required_fields = {"workspace_id", "generation"}
        missing = required_fields - data.keys()
        if missing:
            raise ValueError(f"Missing required fields for Candidate: {missing}")
        d = dict(data)
        if "cluster" in d and isinstance(d["cluster"], str):
            d["cluster"] = int(d["cluster"])
        raw_hp = d.pop("hp_results", [])
        d.pop("metric", None)
        candidate = cls(**d)
        candidate.hp_results = [HPResult.from_dict(hp) for hp in raw_hp]
        return candidate


@dataclass
class ClusteredLeaderboard:
    """
    A leaderboard that organizes candidates by cluster/approach.

    Clusters are assigned by ideas (from the orchestrator).
    """
    clusters: dict[int, list[Candidate]] = field(default_factory=dict)
    cluster_descriptions: dict[int, str] = field(default_factory=dict)
    query: str = ""
    higher_is_better: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    _next_cluster_id: int = 0

    def get_next_cluster_ids(self, n: int) -> list[int]:
        """
        Reserve the next n cluster ids and return them.

        Call this when generating ideas so cluster ids always increase across generations.
        """
        ids = [self._next_cluster_id + i for i in range(n)]
        self._next_cluster_id += n
        self.last_updated = datetime.now().isoformat()
        return ids

    def add_cluster(self, cluster_id: int, description: str) -> None:
        """
        Register a cluster with the given id and description.

        Creates an empty candidate list and stores the idea description.
        If the cluster already exists, only the description is updated.
        """
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []
        self.cluster_descriptions[cluster_id] = description
        self.last_updated = datetime.now().isoformat()

    def add_candidate(self, candidate: Candidate) -> None:
        """Add a candidate to the cluster given by candidate.cluster."""
        cluster_id = candidate.cluster
        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = []
        self.clusters[cluster_id].append(candidate)
        self.last_updated = datetime.now().isoformat()

    def get_all_candidates(self) -> list[Candidate]:
        """Get all candidates across all clusters."""
        all_candidates = []
        for candidates in self.clusters.values():
            all_candidates.extend(candidates)
        return all_candidates

    def get_successful_candidates(self) -> list[Candidate]:
        """Get all successful candidates across all clusters."""
        return [c for c in self.get_all_candidates() if c.success]

    def get_current_generation_candidates(self, generation: int) -> list[Candidate]:
        """Get all candidates from a specific generation."""
        return [c for c in self.get_all_candidates() if c.generation == generation]

    def pareto_front(self) -> list[dict]:
        """Compute the 2D Pareto front over (metric, complexity).

        Flattens all :class:`HPResult` points from all successful
        candidates, then applies the sorting-based sweep-line algorithm:

        1. Sort by metric (best first according to ``higher_is_better``).
        2. Sweep, tracking the best (lowest) complexity seen so far.
           A point enters the front iff its complexity is strictly lower
           than all previous points in the sweep.

        Returns a list of dicts with keys: ``generation``, ``cluster_id``,
        ``idea_description``, ``summary``, ``metric``, ``complexity``,
        ``workspace_id``, ``hp_index``, ``params``, ``code``.
        """
        points: list[tuple[float, float, Candidate, HPResult]] = []
        for c in self.get_successful_candidates():
            for hp in c.hp_results:
                points.append((hp.metric, hp.complexity, c, hp))

        if not points:
            return []

        points.sort(key=lambda p: -p[0] if self.higher_is_better else p[0])

        front: list[dict] = []
        best_complexity = float('inf')
        for metric, complexity, c, hp in points:
            if complexity < best_complexity:
                front.append({
                    "generation": c.generation,
                    "cluster_id": c.cluster,
                    "idea_description": self.cluster_descriptions.get(c.cluster, ""),
                    "summary": c.summary,
                    "metric": metric,
                    "complexity": complexity,
                    "workspace_id": c.workspace_id,
                    "hp_index": hp.hp_index,
                    "params": dict(hp.params),
                    "code": c.code,
                })
                best_complexity = complexity

        return front

    def sample_off_front(
        self,
        k: int,
        temperature: float,
        rng: Optional[random.Random] = None,
    ) -> list[dict]:
        """Sample up to ``k`` candidates from outside the Pareto front.

        Pool definition (one entry per cluster):
          * For each cluster *not* on the current Pareto front, take the
            best-yes-verdict candidate (best metric across all HPResults),
            and represent it by its best HPResult.
          * Candidates whose summary does not start with
            "Followed assigned approach: yes" are excluded.

        Sampling: stable softmax over the **spread-normalised** metric gap
        to the best entry in the pool, sampled without replacement.

            g_i      = |m_i - m*|             (gap to best)
            S        = max_i g_i              (pool spread)
            ~g_i     = g_i / max(S, eps)      (in [0, 1])
            logit_i  = -~g_i / T
            p_i      = softmax(logit_i)

        The spread normalisation makes ``temperature`` scale-invariant:
        the worst-vs-best probability ratio is exp(-1/T) regardless of
        the metric's units.

        Args:
            k: maximum number of entries to return. The returned list is
                shorter if the pool has fewer than ``k`` candidates.
            temperature: softmax temperature. Must be > 0. Lower → greedier;
                higher → flatter.
            rng: optional ``random.Random`` for reproducibility.

        Returns dicts with the same keys as :meth:`pareto_front`.
        """
        if k <= 0:
            return []
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        rng = rng or random.Random()

        front_clusters = {entry["cluster_id"] for entry in self.pareto_front()}

        pool: list[dict] = []
        for cluster_id, candidates in self.clusters.items():
            if cluster_id in front_clusters:
                continue
            best_entry: Optional[dict] = None
            for c in candidates:
                if not c.success or not c.hp_results:
                    continue
                if not _verdict_is_yes(c.summary):
                    continue
                hp = _best_hp(c.hp_results, self.higher_is_better)
                entry = {
                    "generation": c.generation,
                    "cluster_id": c.cluster,
                    "idea_description": self.cluster_descriptions.get(
                        c.cluster, ""
                    ),
                    "summary": c.summary,
                    "metric": hp.metric,
                    "complexity": hp.complexity,
                    "workspace_id": c.workspace_id,
                    "hp_index": hp.hp_index,
                    "params": dict(hp.params),
                    "code": c.code,
                }
                if best_entry is None or _is_better(
                    entry["metric"], best_entry["metric"], self.higher_is_better
                ):
                    best_entry = entry
            if best_entry is not None:
                pool.append(best_entry)

        if not pool:
            return []

        k = min(k, len(pool))

        metrics = [e["metric"] for e in pool]
        best_metric = (
            max(metrics) if self.higher_is_better else min(metrics)
        )
        gaps = [abs(m - best_metric) for m in metrics]
        spread = max(gaps)
        eps = 1e-12
        norm_gaps = [g / max(spread, eps) for g in gaps]

        selected: list[dict] = []
        remaining_idx = list(range(len(pool)))
        remaining_norm_gaps = list(norm_gaps)
        for _ in range(k):
            logits = [-g / temperature for g in remaining_norm_gaps]
            max_logit = max(logits)
            exps = [math.exp(l - max_logit) for l in logits]
            total = sum(exps)
            if total <= 0.0:
                probs = [1.0 / len(remaining_idx)] * len(remaining_idx)
            else:
                probs = [e / total for e in exps]

            u = rng.random()
            acc = 0.0
            chosen = len(remaining_idx) - 1
            for i, p in enumerate(probs):
                acc += p
                if u <= acc:
                    chosen = i
                    break

            pool_idx = remaining_idx.pop(chosen)
            remaining_norm_gaps.pop(chosen)
            selected.append(pool[pool_idx])

        return selected

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "higher_is_better": self.higher_is_better,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "next_cluster_id": self._next_cluster_id,
            "total_candidates": len(self.get_all_candidates()),
            "successful_candidates": len(self.get_successful_candidates()),
            "num_clusters": len(self.clusters),
            "cluster_descriptions": {str(k): v for k, v in self.cluster_descriptions.items()},
            "clusters": {
                str(cluster): [c.to_dict() for c in candidates]
                for cluster, candidates in self.clusters.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClusteredLeaderboard":
        """Create a ClusteredLeaderboard from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")

        higher_is_better = data.get("higher_is_better", False)

        clusters: dict[int, list[Candidate]] = {}
        if "clusters" in data:
            if not isinstance(data["clusters"], dict):
                raise ValueError("'clusters' field must be a dict")
            for key, candidates_data in data["clusters"].items():
                cluster_id = int(key) if isinstance(key, str) else key
                clusters[cluster_id] = [Candidate.from_dict(c) for c in candidates_data]

        raw_descriptions = data.get("cluster_descriptions", {})
        cluster_descriptions = {
            int(k) if isinstance(k, str) else k: v
            for k, v in raw_descriptions.items()
        }

        next_cluster_id = data.get("next_cluster_id")
        if next_cluster_id is None:
            next_cluster_id = max(clusters.keys(), default=-1) + 1

        lb = cls(
            clusters=clusters,
            cluster_descriptions=cluster_descriptions,
            query=data.get("query", ""),
            higher_is_better=higher_is_better,
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
        )
        lb._next_cluster_id = next_cluster_id
        return lb

    def save(self, path: Path):
        """Save the leaderboard to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ClusteredLeaderboard":
        """Load a leaderboard from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
