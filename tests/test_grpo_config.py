from gepa_mindfulness.training.configs import GRPOConfig


def test_grpo_config_defaults_and_overrides():
    cfg = GRPOConfig.from_mapping(
        {
            "group_size": 4,
            "kl_coef": 0.1,
            "trace_frequency": 0.25,
            "trace_strategy": "extremes",
            "reward_weights": {"alpha": 0.1, "delta": 0.4},
            "hallucination": {"confident_wrong_penalty": -3.0},
        }
    )
    assert cfg.group_size == 4
    assert cfg.kl_coef == 0.1
    assert cfg.circuit_tracer.trace_frequency == 0.25
    assert cfg.circuit_tracer.trace_strategy == "extremes"
    assert cfg.reward_weights.alpha == 0.1
    assert cfg.reward_weights.delta == 0.4
    assert cfg.hallucination.confident_wrong_penalty == -3.0
