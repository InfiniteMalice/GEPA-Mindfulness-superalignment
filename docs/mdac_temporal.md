# Temporal evidence at semantic decisions

Two episodes can have identical pooled activations even when their trajectories through a sequence of decisions differ. CounterPoint tests this problem on synchronized physical sensors by centering each modality over time and comparing aligned transient signals. This adapter explores the corresponding language-agent hypothesis. Semantic event correspondence must be supplied and validated independently for each prompt. Token clocks alone do not identify equivalent decisions. [CounterPoint](https://arxiv.org/html/2611.24249)

Use an instrumented model with frozen weights, input and configuration identities. Collect both AttributionGraphs under the same hook method, component definitions, extraction threshold and analysis protocol. Record the protocol in a versioned identifier. Annotate one position per material semantic decision independently in each trace. The adapter compares only features measured at every matched point. Feature identities retain the full hook name as well as the layer and component. A thresholded graph can have absent features or edges. Missing data remains explicit. Existing dual-path rendering and access controls govern collection. The comparison output includes measurement coordinates and numeric results. Keep private prompts, scratchpads and reasoning in the existing restricted collector.

| Matched contrast | Observable test for selecting the pair |
| --- | --- |
| Honest and instructed misleading | Score the same verifiable proposition under both instruction conditions. |
| Recognized and overlooked uncertainty | Match the decision that resolves or surfaces the same missing fact. |
| Memorized and generalized ethics | Use independently scored unseen variants of the same value conflict. |
| Valid and superficial reasoning | Hold surface answer cues fixed and vary a verifiable inference. |
| Genuine and superficial correction | Supply the same corrective evidence and score incorporation at the next decision. |
| Stable objective and irrelevant sensitivity | Perturb a task-irrelevant feature while fixing the task and authority. |

```python
from gepa_mindfulness.interpret.mdac_temporal import Contrast, capture_decisions, compare_decisions

config = {"enabled": True, "model_digest": frozen_model_digest, "protocol": "hook-set-v1"}
left = capture_decisions(graph_a, {"inspect": 17, "answer": 33}, input_digest=input_a, **config)
right = capture_decisions(graph_b, {"inspect": 24, "answer": 41}, input_digest=input_b, **config)
report = compare_decisions(left, right, contrast=Contrast.UNCERTAINTY, enabled=True)
```

Pooled RMS compares per-feature time averages. Individual RMS compares each matched step. Centered RMS removes each trajectory's feature means. Ordered delta RMS compares successive changes when both traces agree on the semantic order. Graph deltas retain the two measured edge weights. Reordered decisions and unmatched points are reported explicitly. A contrast label describes the tested conditions and supplies no ground truth about intent.

To test a mechanistic hypothesis, register an existing native CausalPlan with a prediction, comparison and controls. Supply its frozen CausalCommit to run_decision_probe. The adapter checks the model, primary intervened-on input and decision position against the capture and invokes the established native executor. Inspect identity, random/directional controls, target effect and evidence status. A measured effect is specific to the intervention, outcome and model. A claim about deceptive intent or a general ethical circuit needs additional independent experiments.

Recovery telemetry takes host-measured perturbation trials. Document the perturbation distribution, radius, representation metric, iterative stopping rule, and train/test split alongside the output. The endpoint frequencies describe the sampled initial states. Repeated non-reference endpoints are candidates for further perturb-and-recover trials. Train and test recovery rates have separate denominators. Small local displacements describe measured stability. Jacobian singular values describe a supplied local derivative. Conditional entropy summarizes a supplied normalized prediction at one context. Central-difference curvature requires a meaningful, documented scalar energy, a symmetric perturbation and its step. These are exploratory adaptations of associative-memory diagnostics from image and text diffusion experiments. Applicability to an autoregressive transformer must be tested per measurement. [Pham](https://arxiv.org/html/2611.26797)

Run `python -m pytest -q tests/test_mdac_temporal.py tests/test_native_causal_interventions.py`. CPU tests cover pooled cancellation, individual and centered differences, constant shifts, unmatched or reordered points, edge changes and synthetic telemetry. The optional Transformers/PyTorch test executes a locally initialized GPT-2 intervention with a real zero-effect control. The synthetic contrasts test the measurements. They do not demonstrate a mechanism in a trained agent. No historical artifact migration is required.
