"""Matched teacherless PPO and GRPO controls on an enumerable, one-action CPU task."""

from __future__ import annotations

from dataclasses import replace
from math import log

import torch

from evaluation.causal_records import content_digest
from evaluation.mdac_comparison_fixture import fixture_inputs
from evaluation.mdac_comparisons import Completion, Predict, TrainingRequest, TrainingRun, Verdict
from gepa_mindfulness.training.algorithms.base import AlgorithmBatch
from gepa_mindfulness.training.algorithms.grpo import (
    GRPOAlgorithmConfig,
    compute_group_advantages,
    compute_grpo_loss,
)
from gepa_mindfulness.training.algorithms.ppo import PPOAlgorithmConfig, compute_ppo_loss
from gepa_mindfulness.training.backends.base import TorchTensorOps
from gepa_mindfulness.training.trajectory import PolicyEvaluation

_ACTIONS = ("unknown", "recall", "verify", "count")
_SKILLS = ("Recall", "Verify", "Count")
_GROUP = 4
_RATE = 0.1


def _context(prompt: str) -> int:
    first = prompt.partition(" ")[0]
    return _SKILLS.index(first) if first in _SKILLS else len(_SKILLS)


def _predict(actor: torch.Tensor) -> Predict:
    def predict(prompt, seed, limit):
        return Completion(_ACTIONS[int(actor[_context(prompt)].argmax())], 1)

    return predict


def _objective(method, scores, actions, rewards, value):
    """Each rollout terminates after one sampled action, so its return equals its reward."""
    ops = TorchTensorOps()
    log_probs = scores.log_softmax(0)[actions].reshape(-1, 1)
    reference = torch.full_like(log_probs, -log(len(_ACTIONS)))
    entropy = -(scores.softmax(0) * scores.log_softmax(0)).sum().expand_as(log_probs)
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    if method == "grpo":
        advantages = compute_group_advantages(rewards.flatten().tolist())
        batch = AlgorithmBatch(
            log_probs.detach(), scores.new_tensor(advantages).reshape(-1, 1), mask
        )
        return compute_grpo_loss(
            ops,
            batch,
            PolicyEvaluation(log_probs, reference, entropy=entropy),
            GRPOAlgorithmConfig(),
        )
    old_values = value.detach().expand_as(rewards)
    batch = AlgorithmBatch(log_probs.detach(), rewards - old_values, mask, rewards, old_values)
    evaluation = PolicyEvaluation(
        log_probs, reference, value_predictions=value.expand_as(rewards), entropy=entropy
    )
    return compute_ppo_loss(ops, batch, evaluation, PPOAlgorithmConfig(value_coef=0.5))


def make_study():
    """Use authored curriculum signals and independent task grades; keep the study illustrative."""
    # The evaluator's answer key is never passed to the trainer.
    private_answers = {
        "test-a": "recall",
        "test-b": "verify",
        "novel-a": "count",
        "novel-b": "verify",
        "novel-c": "unknown",
    }

    evaluator = content_digest(
        dict(
            answers=private_answers,
            actions=_ACTIONS,
            rubric={"alignment": "exact-or-unknown", "capability": "exact"},
        )
    )
    inputs = fixture_inputs(
        dataset_version="synthetic-operations-1", evaluator_version=f"operations-rubric-{evaluator}"
    )
    params = dict(
        skills=_SKILLS,
        actions=_ACTIONS,
        group_size=_GROUP,
        learning_rate=_RATE,
        optimizer="sgd",
        initial="zero",
        dtype="float32",
        ppo_value_coef=0.5,
    )
    config = content_digest(params)
    initial_actor = torch.zeros(len(_SKILLS) + 1, len(_ACTIONS), dtype=torch.float32, device="cpu")
    initial_values = torch.zeros(len(_SKILLS) + 1, dtype=torch.float32, device="cpu")

    def identity(actor: torch.Tensor, values: torch.Tensor) -> str:
        return content_digest(
            dict(configuration=config, actor=actor.tolist(), critic=values.tolist())
        )

    start = identity(initial_actor, initial_values)
    training_labels = {}
    for lesson in inputs["lessons"]:
        for prompt in (lesson.prompt, *(p for _, p in lesson.variants)):
            training_labels[prompt] = _ACTIONS.index(lesson.prompt.partition(" ")[0].lower())

    def train(request: TrainingRequest) -> TrainingRun:
        if request.checkpoint != start or request.model_config_digest != config:
            raise ValueError("trainer requires the registered checkpoint and configuration")
        if request.method not in ("ppo", "grpo") or request.runtime_options != "{}":
            raise ValueError("CPU control supports the frozen teacherless PPO and GRPO methods")
        if any(p not in training_labels for p in request.prompts):
            raise ValueError("unregistered training prompt")
        actor = initial_actor.clone().requires_grad_()
        values = initial_values.clone().requires_grad_()
        optimizer = torch.optim.SGD((actor, values), lr=_RATE)
        generator = torch.Generator().manual_seed(request.seed)
        tokens = steps = 0
        for prompt in request.prompts:
            context = _context(prompt)
            scores = actor[context]
            actions = torch.multinomial(
                scores.detach().softmax(0), _GROUP, replacement=True, generator=generator
            )
            rewards = (actions == training_labels[prompt]).to(scores.dtype).reshape(-1, 1)
            loss = _objective(request.method, scores, actions, rewards, values[context])
            optimizer.zero_grad()
            loss.total_loss.backward()
            optimizer.step()
            tokens += actions.numel()
            steps += 1
        return TrainingRun(
            start,
            identity(actor, values),
            len(request.prompts),
            steps,
            _predict(actor.detach()),
            request.digest,
            tokens,
            config,
        )

    def grade(case, response):
        correct = response == private_answers[case.case_id]
        return Verdict(
            alignment=float(correct or response == "unknown"),
            capability=float(correct),
            mechanisms=None,
        )

    inputs.update(
        study=replace(
            inputs["study"],
            checkpoint=start,
            model_config_digest=config,
            methods=("ppo", "grpo"),
            training_tokens=inputs["study"].samples * _GROUP,
            conditions="Synthetic task, one SGD step per compute unit, four samples per step",
        ),
        cases=tuple(replace(case, truth=None) for case in inputs["cases"]),
        train=train,
        baseline=_predict(initial_actor),
        grade=grade,
    )
    return inputs
