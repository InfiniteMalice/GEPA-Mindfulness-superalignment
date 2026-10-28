# KnowGPO + PRM4RL: evidence and credit-accounting protocol

Maturity: **Experimental** CPU state/grouping/potential preparation. Model training, trainer
attachment, protected behavioral comparisons and empirical improvement are **Unsupported** in
this stage. All emitted records remain DEVELOPMENT data. Nothing here admits a dataset,
changes reward eligibility, executes an action or grants runtime authority.

## Sources and adaptations

[KnowGPO, arXiv:2610.22331v1](https://arxiv.org/html/2610.22331v1) defines atomic conditions,
monotonic knowledge accumulation, incremental gains and within-prior-state normalization.
Algorithm 1 uses the preceding state; the displayed equation (5) has an inconsistent index.
Use Algorithm 1 and its accompanying explanation. The official
[repository revision](https://github.com/ZJU-REAL/KnowGPO/tree/5b4f8bb5a782a09f4e9a0bcc630557c5deb7715d)
contains the method README and an Apache-2.0 license, but advertises the core code as coming
soon. The paper has an arXiv non-exclusive license. No upstream code or paper prose is copied.

[PRM4RL, arXiv:2610.22056v1](https://arxiv.org/html/2610.22056v1) section 4.2 separates its
potential difference from a success bonus. Its NeurIPS 2025 publication is independently listed
in the [proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/91b047c5f5bd41ef56bfaf4ad0bd19e3-Abstract-Conference.html),
DOI `10.52202/085713-3370`; the arXiv version was posted in 2026. Paper license: CC BY 4.0.
The official [repository revision](https://github.com/deng-ai-lab/PRM4RL/tree/f9727ed0d62aa7106b1dd8d76a52b5a6d6775079)
was checked, with no code license identified. No code is copied and no robotics stack is imported.

Primary definitions and metadata were checked on 2026-10-27. The retained
[KnowGPO method note](recommendations/source_artifacts/knowgpo-2610.22331v1-method.txt) and
[PRM4RL equation note](recommendations/source_artifacts/prm4rl-2610.22056v1-method.txt)
identify the checked mathematical subset, with local byte counts/digests in the
[integration matrix](recommendations/RESEARCH_INTEGRATION_MATRIX.md#source-intake-and-adaptation-boundary).
These are short authored notes, not raw publisher HTML/PDF artifacts.

The implementation deliberately differs from both full methods. Evidence can be revoked and
contradicted; it is not forced into a monotonic OR state. Gains count first acquisition per
episode; reacquisition and belief correction are separate diagnostics. Groups retain prior
contradiction and ever-supported history as well as current support. Milestones are discrete,
dependency-closed and equally weighted. There is no LLM-generated executable progress function,
robot state augmentation, success bonus, episode-outcome mixing, PPO update or token-level
advantage broadcast. Published results and hyperparameters are not transferred to Mindfulness.

## Existing contracts and explicit configuration

[knowledge_progress.py](../gepa_mindfulness/training/knowledge_progress.py) reuses
`EvidenceState`, `EvidenceClaim`, `RewardRequest`, `TrustedEvaluatorContract`,
`RewardProvenance`, `ResearchCreditClaim`, the mutation-safe causal host review and
[research credit accounting](research_credit_accounting.md). It does not replace evidence
commitment, introduce another authority or classify canonical V5 cases.

`KnowledgeCondition` has a stable condition ID, task ID/version, required proposition,
knowledge/milestone kind, configured verifier and dependency IDs. A task-local manifest has
1–64 conditions. Atomic knowledge conditions have no dependencies. Milestone dependencies
must refer to that manifest and form a DAG. The manifest is an immutable experiment input;
it never rewrites the canonical 17-case registry or a held-out evaluation set.

`LearningStep` retains an episode ID, consecutive step index, host-normalized semantic event ID,
exact action `RewardRequest`, complete before/after evidence states, optional separate before/after
belief states, a host-captured answer flag and continuing/terminated/truncated status. All states
cover exactly the manifest conditions and preserve their propositions. Supported/contradicted
findings require references; inferred, observed, stale and superseded labels are not silently
converted to verified support. Every reference must be observable and belong to that request.
Belief statuses describe the model's declared position, not truth or source verification.

`LearningContext` pins authority/run/task/version/model/harness/producer/evaluator/dataset digest
and explicit `gamma` in `(0, 1]`. One configured independent verifier reviews a task batch in this
bounded version. Hosts must authenticate that principal and its competence. Different strings
are not independence proof. The request's model identity and policy version must match, and
source sampling metadata must explicitly be TRAIN. The existing recursive eligibility policy
also rejects nested holdout/development restrictions and malformed generated-data review.
No inference request, paid API, training job or persistent write is launched.

```python
from gepa_mindfulness.training.knowledge_progress import prepare_knowledge_progress

assert prepare_knowledge_progress(object(), object(), object()) is None  # default off
report = prepare_knowledge_progress(
    context, condition_manifest, captured_steps,
    authorize_evidence=independent_host_review,
    epsilon=1e-6,
    enabled=True,
)
```

`enabled` is a strict boolean; `epsilon` is an explicit finite numerical-stability parameter
in `(0, 1]`, with a documented software default, not an inferred paper hyperparameter. Its exact value is
retained as `normalization_epsilon` in the reviewed source and source digest. The
step roster contains 1–128 steps and can interleave episodes. Each starts at zero; before/after
states must join exactly. A terminated or truncated episode cannot continue. Declared belief
history must also join when supplied. Action/event IDs cannot be reused. Conflicting source-kind
declarations for one evidence ID in the complete trajectory roster, including unselected
references, fail before host review. Every action ID must be nonblank and bounded before
normalization, including steps with no contribution. Semantic aliases across IDs remain
an external evidence-normalization responsibility.

## Independent verification and output boundaries

All records are snapshotted and versioned with exact fields. Context, manifest and step source
records are bounded together; the complete source, metrics, proposed claims and overlap audit
must also fit the existing 4 MiB JSON envelope **before** the host callback runs. The callback
receives the exact source and proposed metrics. Only literal True without mutation accepts the
evidence review. Missing review, exceptions, rejection, non-booleans and mutation withhold
reported claims. Raw proposed values remain explicitly non-authoritative DEVELOPMENT diagnostics
inside the retained source envelope. Evidence uncertainty is not a moral-rule violation.

The host must independently check condition coverage, evidence truth/relevance/freshness,
provenance, source access, actual action/episode/answer/termination identity and meaningful
state changes. A fluent answer, retrieved duplicate, self-certified fact or model-generated
verifier label is insufficient. Supported and contradicted evidence interpretations are inputs
to review, not automated proofs. All sources remain independently accessible outside this
report; identifiers and digests do not recover missing evidence or authenticate it.

`knowledge_gain` counts newly supported factual conditions not previously supported in that
episode. Duplicate retrieval/restatement has zero gain. Initial support is baseline, not new
acquisition. Revocation remains visible; reacquisition cannot repeatedly generate acquisition
credit. A contradicted model belief becoming supported by verified evidence is separately
recorded as `belief_correction_ids`; a correction of already known information does not also
mint new acquisition. This stage supplies correction diagnostics, not a distinct correction loss.

Groups compare non-answer, continuing steps with identical prior factual support/contradiction
states and ever-supported history under one manifest. They ignore belief wording and evidence
reference identity. Normalization uses group mean and population standard deviation plus epsilon;
a singleton or constant-gain group gets zero advantage. Negative comparative advantage is a
numerical comparison of observable acquisition, not a punishment for hidden thoughts. No final
answer-correctness metric or factual outcome reward is inferred from knowledge gain.

## Milestones and potential assumptions

Only supported milestones whose entire dependency closure is supported count as complete.
`Phi(s)` is the fraction of the fixed manifest's milestones completed; a task without milestones
has zero potential. Raw completed/blocked IDs, milestone delta, potential and shaping remain
separate from knowledge gain. For continuing transitions:

`F(s, a, s') = gamma * Phi(s') - Phi(s)`.

The calculation uses exact milestone fractions and the exact represented gamma before converting
JSON result values to floats. No fixed tolerance discards small nonzero shaping contributions;
exact cancellation cannot create a rounding-only progress claim.

A genuine finite-episode termination has zero next potential, including failed or premature
termination. This can give negative shaping without asserting a hidden-thought failure. A
truncation yields `shaping=None` and no progress claim: it is not silently treated as terminal
success. A future trainer must supply a correct continuation/bootstrap treatment before using
truncated trajectories. Do not use partial rows as a claim of policy invariance.

The host review explicitly covers a fixed, bounded, Markov potential under the original task
state, the same discount as the objective and correct boundary treatment. These declarations
cannot make partial observations Markov, certify task success or prove the theorem's assumptions.
Changing milestones/potential versions requires a new frozen experiment, not mid-rollout edits.
There is no success bonus here; task completion requires independent outcome verification.
Shaping and task success stay distinct. The CPU telescoping test verifies arithmetic only.

## No duplicate optimizer credit

Knowledge and progress proposals retain the **same host-normalized semantic event ID**, action,
state, model/harness and evidence. The existing audit reports event overlaps and shared evidence;
it never adds the components. A single source can support distinct facts, while distinct IDs can
alias one event. The host must normalize those events before any admission.

Even accepted evidence review leaves every claim's `admissibility="unresolved"`. It cannot
resolve a constitutional judgment, authorize an action or revise a moral commitment. A later
independent host must resolve applicable admissibility, authenticate provenance and use the
existing durable `ResearchCreditLedger` with explicit provider/reward-integrity/causal and any
other active-layer accounting. That ledger permits at most one contribution per event, including
across reopenings. Reservation is still not reward payment. No hook into `RewardPipeline` or
canonical training backends is installed. No process bonus is made negative.

## Reproducible study B protocol

Freeze task conditions, potential, data splits, checkpoints, model/harness/evaluator versions,
semantic event normalization and compute budgets before capture. Compare outcome-only GRPO/DAPO,
KnowGPO, PRM4RL, VPO and the requested pairwise/joint arms only when their runtimes exist. VPO and
all combined trainer arms remain Unsupported here. Keep constitutional constraints and protected
CASE × STRIPE × REPEAT suites fixed. Use independently held-out tests, multiple seeds and paired
uncertainty estimates; report exclusions and missing outcomes without replacing them post hoc.

Record each raw component and event overlap, group size/gain distribution, false support,
revocations, reacquisition, corrections, blocked milestones, incomplete termination, verified task
success, calibration, abstention, source fidelity, protected regressions, token/time/compute and
review outcomes. Include duplicate retrieval, fabricated facts, contradictory evaluators,
unsupported milestones, task shortcuts and early termination adversaries. An intervention
cannot be promoted for higher progress metrics while protected behavior or honest completion
regresses. These comparisons have not been run; unit tests establish software behavior only.

Run deterministic CPU contracts with `pytest -q tests/test_knowledge_progress.py`, plus existing
credit-ledger, evidence-state, eligibility, documentation and canonical-case regressions. Build
the wheel and check the guide, matrix and both method notes through `importlib.resources` outside
the checkout. Large-model training and external spending require separately configured authority.
