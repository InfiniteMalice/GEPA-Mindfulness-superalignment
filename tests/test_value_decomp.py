from __future__ import annotations

import pytest

from mindful_trace_gepa.configuration import DSPyConfig
from mindful_trace_gepa.dspy_modules.pipeline import GEPAChain
from mindful_trace_gepa.value_decomp.deep_value_spaces import (
    DeepValueVector,
    ShallowPreferenceVector,
)
from mindful_trace_gepa.value_decomp.dvb_eval import DVBExample, compute_dvgr
from mindful_trace_gepa.value_decomp.gepa_decomposition import (
    GepaDecomposition,
    LinearValueProbe,
    decompose_gepa_score,
)
from mindful_trace_gepa.value_decomp.output_value_analyzer import (
    analyze_output_deep_values,
    analyze_output_shallow_features,
)
from mindful_trace_gepa.value_decomp.user_value_parser import (
    parse_user_deep_values,
    parse_user_shallow_prefs,
)


def test_vector_tensor_conversion_roundtrip() -> None:
    vector = DeepValueVector(1.0, 0.5, 0.25, 0.1, 0.2, 0.3, 0.4)
    tensor = vector.to_tensor()
    restored = DeepValueVector.from_tensor(tensor)
    assert restored.reduce_suffering == pytest.approx(1.0)
    assert restored.agency == pytest.approx(0.4)

    shallow = ShallowPreferenceVector(tone_formal=0.5, verbosity=0.3)
    restored_shallow = ShallowPreferenceVector.from_tensor(shallow.to_tensor())
    assert restored_shallow.tone_formal == pytest.approx(0.5)
    assert restored_shallow.verbosity == pytest.approx(0.3)


def test_parse_user_values() -> None:
    prompt = "Please be concise, avoid harm, and offer perspective while being kind."
    deep = parse_user_deep_values(prompt)
    shallow = parse_user_shallow_prefs(prompt)
    assert deep.reduce_suffering > 0
    assert deep.perspective > 0
    assert shallow.verbosity > 0
    assert shallow.deference > 0


def test_analyze_output_features() -> None:
    output = "I'm here to help. Maybe we should consider options carefully!"
    deep_vec = analyze_output_deep_values(
        output, {"Reduce Suffering": 0.8, "Increase Knowledge": 0.6}
    )
    shallow_vec = analyze_output_shallow_features(output)
    assert deep_vec.reduce_suffering >= 0.8
    assert shallow_vec.hedging > 0
    assert shallow_vec.tone_therapeutic > 0


def test_compute_dvgr_metric() -> None:
    examples = [
        DVBExample("p1", "a", "b", deep_label=0, shallow_label=1),
        DVBExample("p2", "a", "b", deep_label=1, shallow_label=0),
        DVBExample("p3", "a", "b", deep_label=0, shallow_label=1),
    ]
    dvgr = compute_dvgr(examples, lambda ex: ex.deep_label)
    assert dvgr == pytest.approx(1.0)


def test_gepa_decomposition_with_probe() -> None:
    deep = DeepValueVector(1.0, 0.5, 0.2, 0.1, 0.1, 0.1, 0.1)
    shallow = ShallowPreferenceVector(verbosity=0.5, hedging=0.2)
    probe = LinearValueProbe.from_sizes(len(deep.ORDER), len(shallow.ORDER), scale=0.5)
    decomp = decompose_gepa_score([0.8, 0.7, 0.9], deep, shallow, probe=probe)
    assert isinstance(decomp, GepaDecomposition)
    assert decomp.deep_contribution > decomp.shallow_contribution


def test_gepa_chain_integration_value_decomp() -> None:
    config = DSPyConfig(enable_value_decomposition=True, enable_dvgr_eval=True)
    chain = GEPAChain(config=config)
    result = chain.run("Provide honest, safe advice", context="formal tone")
    assert result.value_decomposition is not None
    payload = result.value_decomposition
    assert payload["output_deep"]["reduce_suffering"] >= 0
    assert payload["gepa_decomposition"]["deep_contribution"] != 0
    assert payload["dvgr"] is not None

    no_value_chain = GEPAChain(config=DSPyConfig())
    no_value_result = no_value_chain.run("Simple answer", context="")
    assert no_value_result.value_decomposition is None
