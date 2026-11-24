"""Opt-in temporal comparisons at evaluator-aligned semantic decision points."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from math import fsum, isclose, isfinite, log2, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gepa_mindfulness.verification.runtime_governance import TrustedClock

    from .attribution_graphs import AttributionGraph
    from .native_causal_records import CausalCommit
    from .native_neurons import FrozenNativeModel


class Contrast(str, Enum):
    HONESTY = "honest_vs_instructed_misleading"
    UNCERTAINTY = "recognized_vs_overlooked_uncertainty"
    ETHICS = "memorized_vs_generalized_ethics"
    REASONING = "valid_vs_superficial_reasoning"
    CORRECTION = "genuine_vs_superficial_correction"
    OBJECTIVE = "stable_objective_vs_irrelevant_sensitivity"


def _finite(value: object) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("measurement must be finite")
    return float(value)


def _mean(values: list[float]) -> float:
    return fsum(values) / len(values)


def _rms(values: list[float]) -> float:
    return sqrt(_mean([x * x for x in values]))


Feature = tuple[int, str, int | None, str | None]
Node = tuple[str, Feature]
Edge = tuple[Node, Node]


@dataclass(frozen=True)
class SemanticTrace:
    model_digest: str
    input_digest: str
    protocol: str
    method: str
    positions: Mapping[str, int]
    activations: Mapping[Node, float]
    edges: Mapping[Edge, float]


def capture_decisions(
    graph: AttributionGraph,
    positions: Mapping[str, int],
    *,
    model_digest: str,
    input_digest: str,
    protocol: str,
    enabled: bool = False,
) -> SemanticTrace | None:
    """Project graph nodes onto declared semantic points and retain numeric data only."""
    if enabled is not True:
        return None
    for digest in (model_digest, input_digest):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("require exact model and input digests")
    if not protocol.strip() or not graph.method.strip() or not positions:
        raise ValueError("instrumentation and semantic annotation protocol required")
    if any(
        type(k) is not str or not k.strip() or type(v) is not int or v < 0
        for k, v in positions.items()
    ) or len(set(positions.values())) != len(positions):
        raise ValueError("each decision needs a unique in-trace position")
    labels = {v: k for k, v in positions.items()}
    identities: dict[int, Node] = {}
    activations = {}
    for node in graph.nodes:
        if node.token_position in labels:
            feature = (node.layer, node.component_type, node.component_idx, node.hook_name)
            name = (labels[node.token_position], feature)
            if name in activations:
                raise ValueError("duplicate component at a semantic point")
            activations[name] = _finite(node.activation_value)
            identities[id(node)] = name
    edges = {}
    for edge in graph.edges:
        src, dst = identities.get(id(edge.source_node)), identities.get(id(edge.target_node))
        if src is not None and dst is not None:
            if (src, dst) in edges:
                raise ValueError("ambiguous duplicate observed edge")
            edges[src, dst] = _finite(edge.attribution_weight)
    ordered = dict(sorted(positions.items(), key=lambda pair: pair[1]))
    return SemanticTrace(
        model_digest,
        input_digest,
        protocol,
        graph.method,
        MappingProxyType(ordered),
        MappingProxyType(activations),
        MappingProxyType(edges),
    )


def compare_decisions(
    left: SemanticTrace,
    right: SemanticTrace,
    *,
    contrast: Contrast,
    enabled: bool = False,
) -> dict[str, Any] | None:
    """Contrast paired trajectories only on features measured at every matched decision."""
    if enabled is not True:
        return None
    if not isinstance(contrast, Contrast) or (left.model_digest, left.protocol, left.method) != (
        right.model_digest,
        right.protocol,
        right.method,
    ):
        raise ValueError("incomparable model, instrumentation protocol or contrast")
    labels = [key for key in left.positions if key in right.positions]
    ordered = labels == [key for key in right.positions if key in left.positions]
    common: set[Feature] = set()
    missing: list[str] = []
    for index, key in enumerate(labels):
        a = {f for label, f in left.activations if label == key}
        b = {f for label, f in right.activations if label == key}
        if not a.intersection(b):
            missing.append(key)
        common = a & b if index == 0 else common & a & b
    report: dict[str, Any] = dict(
        contrast=contrast.value,
        model_digest=left.model_digest,
        method=left.method,
        protocol=left.protocol,
        left_input=left.input_digest,
        right_input=right.input_digest,
        matched=labels,
        left_only=[k for k in left.positions if k not in right.positions],
        right_only=[k for k in right.positions if k not in left.positions],
        order_agrees=ordered,
        missing_measurements=missing,
        feature_count=len(common),
        pooled_rms=None,
        individual_rms={},
        centered_rms=None,
        ordered_delta_rms=None,
        edge_deltas=[],
        training_eligibility="DEVELOPMENT",
    )
    if not common:
        return report
    offsets = {
        f: [right.activations[k, f] - left.activations[k, f] for k in labels] for f in common
    }
    averages = {f: _mean(seq) for f, seq in offsets.items()}
    report["pooled_rms"] = _rms(list(averages.values()))
    report["individual_rms"] = {
        key: _rms([seq[i] for seq in offsets.values()]) for i, key in enumerate(labels)
    }
    if len(labels) >= 2:
        report["centered_rms"] = _rms([x - averages[f] for f, seq in offsets.items() for x in seq])
        if ordered:
            report["ordered_delta_rms"] = _rms(
                [b - a for seq in offsets.values() for a, b in zip(seq, seq[1:])]
            )
    # A missing edge stays absent in the report. Thresholded extraction cannot establish zero.
    for edge in sorted(set(left.edges) | set(right.edges), key=repr):
        if all(label in labels and feature in common for label, feature in edge):
            first, second = left.edges.get(edge), right.edges.get(edge)
            report["edge_deltas"].append(
                dict(
                    edge=edge,
                    left=first,
                    right=second,
                    delta=None if first is None or second is None else second - first,
                )
            )
    return report


def run_decision_probe(
    trace: SemanticTrace,
    decision: str,
    backend: FrozenNativeModel,
    commit: CausalCommit,
    *,
    clock: TrustedClock,
    enabled: bool = False,
) -> dict[str, Any] | None:
    """Run the prediction-bound probe at its registered decision point."""
    if enabled is not True:
        return None
    from .causal_interventions import execute_causal_plan

    plan = commit.plan
    if (
        trace.model_digest != plan.model_digest
        or trace.input_digest != plan.primary_input.digest
        or decision not in trace.positions
        or plan.source.token_positions != (trace.positions[decision],)
    ):
        raise ValueError("causal plan must match the captured model, input and decision")
    return execute_causal_plan(backend, commit, clock=clock, enabled=True)


def recovery_telemetry(
    *,
    enabled: bool = False,
    train_recovered: tuple[bool, ...] = (),
    test_recovered: tuple[bool, ...] = (),
    displacements: tuple[float, ...] = (),
    jacobian: tuple[tuple[float, ...], ...] = (),
    probabilities: tuple[float, ...] = (),
    energy_samples: tuple[float, ...] = (),
    step: float | None = None,
    energy_definition: str | None = None,
    endpoints: tuple[str, ...] = (),
    known: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """Describe supplied perturbation trials and local derivatives in their measured coordinates."""
    if enabled is not True:
        return None

    def rate(bits: tuple[bool, ...]) -> float | None:
        if any(type(b) is not bool for b in bits):
            raise ValueError("recovery indicators must be observed booleans")
        return sum(bits) / len(bits) if bits else None

    train, test = rate(train_recovered), rate(test_recovered)
    distances = [_finite(x) for x in displacements]
    if any(x < 0 for x in distances):
        raise ValueError("representation distances must be nonnegative")
    spectrum = None
    if jacobian:
        import numpy as np

        if not jacobian[0] or any(len(row) != len(jacobian[0]) for row in jacobian):
            raise ValueError("jacobian must be a nonempty rectangular matrix")
        matrix = [[_finite(x) for x in row] for row in jacobian]
        spectrum = tuple(float(s) for s in np.linalg.svd(matrix, compute_uv=False))
    entropy = None
    if probabilities:
        p = [_finite(x) for x in probabilities]
        if any(x < 0 for x in p) or not isclose(fsum(p), 1.0, abs_tol=1e-9):
            raise ValueError("conditional distribution must sum to one")
        entropy = -fsum(x * log2(x) for x in p if x > 0)
    curvature = None
    if energy_samples or step is not None or energy_definition is not None:
        if (
            len(energy_samples) != 3
            or not isinstance(energy_definition, str)
            or not energy_definition.strip()
            or step is None
            or _finite(step) <= 0
        ):
            raise ValueError("central difference needs a model-defined energy and positive step")
        minus, center, plus = map(_finite, energy_samples)
        curvature = (plus - 2 * center + minus) / (step**2)
    counts = Counter(endpoints)
    return dict(
        train_n=len(train_recovered),
        test_n=len(test_recovered),
        train_recovery=train,
        test_recovery=test,
        train_test_recovery_gap=None if train is None or test is None else train - test,
        mean_displacement=_mean(distances) if distances else None,
        jacobian_spectrum=spectrum,
        energy_curvature=curvature,
        energy_definition=energy_definition,
        conditional_entropy_bits=entropy,
        endpoint_frequency={k: v / len(endpoints) for k, v in counts.items()},
        spurious_attractor_candidates={k: v for k, v in counts.items() if v > 1 and k not in known},
    )
