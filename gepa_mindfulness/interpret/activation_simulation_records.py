"""Experimental frozen simulation controls; inert diagnostics, never training authority."""

# Standard library
from __future__ import annotations

from dataclasses import dataclass

# Local
from gepa_mindfulness.verification.cva_records import (
    CVARecord,
    _read_array,
    _read_records,
    _record,
    array,
    digest,
    text,
)
from gepa_mindfulness.verification.state import parse_rfc3339_datetime

from .explanation_tests import ActivationPredictor
from .native_records import NativeBatch, finite, index, indices


@dataclass(frozen=True)
class ChoiceToken(CVARecord):
    label: str
    token_id: int

    schema_version = "simulation-choice-token-v1"

    def __post_init__(self) -> None:
        text(self.label, "label")
        index(self.token_id)


@dataclass(frozen=True)
class LogitChoiceProbe(CVARecord):
    """Host-declared answer-token mapping, not a certified semantic judge."""

    probe_id: str
    example_id: str
    position: int
    choices: tuple[ChoiceToken, ...]
    expected_label: str | None = None

    schema_version = "simulation-logit-choice-probe-v1"
    restorers = {"choices": lambda v: _read_records(v, ChoiceToken)}

    def __post_init__(self) -> None:
        text(self.probe_id, "probe_id")
        text(self.example_id, "example_id")
        index(self.position, maximum=128)
        choices = tuple(_record(c, ChoiceToken) for c in array(self.choices, 16, required=True))
        if len(choices) < 2 or len({c.label for c in choices}) != len(choices):
            raise ValueError("choice probe requires distinct labels")
        if len({c.token_id for c in choices}) != len(choices):
            raise ValueError("choice tokens must be distinct")
        if self.expected_label is not None and self.expected_label not in {
            c.label for c in choices
        }:
            raise ValueError("expected_label not in declared choices")


@dataclass(frozen=True)
class ComponentControl(CVARecord):
    commit_digest: str
    generation_batch_digest: str
    component_index: int
    mean: float
    std: float
    matched_predictor: ActivationPredictor

    schema_version = "simulation-component-control-v1"
    restorers = {"matched_predictor": ActivationPredictor.from_dict}

    def __post_init__(self) -> None:
        digest(self.commit_digest)
        digest(self.generation_batch_digest)
        index(self.component_index)
        finite(self.mean)
        finite(self.std)
        if self.std < 0:
            raise ValueError("standard deviation cannot be negative")
        _record(self.matched_predictor, ActivationPredictor)


@dataclass(frozen=True)
class SimulationControls(CVARecord):
    """Freeze source identities, construction seed, positions, mappings and controls."""

    model_digest: str
    hook_digest: str
    split_digest: str
    components: tuple[ComponentControl, ...]
    generation_batches: tuple[NativeBatch, ...]
    activation_positions: tuple[int, ...]
    logit_positions: tuple[int, ...]
    label_permutation: tuple[int, ...]
    random_component_indices: tuple[int, ...]
    probes: tuple[LogitChoiceProbe, ...]
    seed: int
    created_at: str
    prediction_input_policy: str = "causal_prefix"

    schema_version = "simulation-controls-v1"
    restorers = {
        "components": lambda v: _read_records(v, ComponentControl),
        "generation_batches": lambda v: _read_records(v, NativeBatch),
        "activation_positions": _read_array,
        "logit_positions": _read_array,
        "label_permutation": _read_array,
        "random_component_indices": _read_array,
        "probes": lambda v: _read_records(v, LogitChoiceProbe),
    }

    def __post_init__(self) -> None:
        if self.prediction_input_policy != "causal_prefix":
            raise ValueError("simulation requires causal_prefix prediction inputs")
        for value in (self.model_digest, self.hook_digest, self.split_digest):
            digest(value)
        components = tuple(
            _record(c, ComponentControl) for c in array(self.components, 16, required=True)
        )
        if len({c.component_index for c in components}) != len(components):
            raise ValueError("duplicate simulation components")
        if len({c.commit_digest for c in components}) != len(components):
            raise ValueError("duplicate simulation commitments")
        batches = tuple(
            _record(b, NativeBatch) for b in array(self.generation_batches, 16, required=True)
        )
        if len(batches) != len(components):
            raise ValueError("one original generation batch is required per component")
        for component, batch in zip(components, batches):
            if (
                component.generation_batch_digest != batch.digest
                or batch.model.digest != self.model_digest
                or batch.hook.digest != self.hook_digest
                or batch.component_indices != (component.component_index,)
            ):
                raise ValueError("generation provenance differs from frozen controls")
        for position in indices(self.activation_positions, 128) + indices(
            self.logit_positions, 128
        ):
            index(position, maximum=128)
        permutation = array(self.label_permutation, 16, required=True)
        if any(type(i) is not int for i in permutation) or sorted(permutation) != list(
            range(len(components))
        ):
            raise ValueError("invalid label permutation")
        if len(components) > 1 and any(i == p for i, p in enumerate(permutation)):
            raise ValueError("shuffled label control requires a derangement")
        random_components = array(self.random_component_indices, 16)
        if random_components:
            indices(random_components, 16)
            if len(random_components) != len(components):
                raise ValueError("random component control must match group size")
            if set(random_components) & {c.component_index for c in components}:
                raise ValueError("random component control overlaps target components")
        probes = tuple(_record(p, LogitChoiceProbe) for p in array(self.probes, 32))
        if len({p.probe_id for p in probes}) != len(probes):
            raise ValueError("duplicate probe IDs")
        if any(p.position not in self.logit_positions for p in probes):
            raise ValueError("probe is outside frozen logit positions")
        index(self.seed, maximum=2**32)
        parse_rfc3339_datetime(self.created_at, "created_at")
