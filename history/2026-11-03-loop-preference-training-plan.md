# Native DeLoop preference training

The operator requests enabled software features and excludes research studies.
Reuse verified PEO loop/recovery mining, the separate existing DPO objective,
native CPU optimizer/checkpoints (including optional SST), and existing
CoevolutionStore admission. Source data must already be TRAIN; proposal/review
reports remain DEVELOPMENT. No diagnostic relabeling, new case taxonomy, reward
authority or deployment mechanism is introduced.

The previously authorized repo-quality-gate fallback applies because that skill
is unavailable: specification/design, implementation, tests, one independent
review, and evidence. Use Superpowers inline execution and TDD. No beads work.

Train an isolated model on independently reviewed first recoveries at the exact
preceding state of each rejected repeat. The later recovery prompt is never used
as the preference prompt. Require fresh host source and training-use approval,
tokenizer/current-policy/source-file binding, registered correction/PEO equality,
and immutable raw sources. Apply action-sequence DPO against the frozen reference;
never merge its loss with LoCD or ActionOPD. Beta is explicit operator input.

Extract existing LoCD candidate finalization for reuse: new immutable checkpoint,
bounded public-only evaluation, store-issued held-out/protected receipts, existing
admission, unchanged original model. Catalog acceptance never deploys a model.
Keep the path disabled by default and all limits/versioning explicit.

Tests must demonstrate real parameter changes, original-state token alignment,
unchanged reference/source/RNG, rejection on uncertainty/unsupported recovery,
held-out provenance, stale checksums, changed registered events, mutation during
review, missing host authority, protected regression, and checkpoint round-trip
with ordinary and SST optimizers. Tiny authored CPU fixtures establish software
behavior only. Document research rationale/protocol without running it.

## Functional evidence before independent review

The missing native adapter was observed RED. Twenty-eight authored checks now
exercise actual sequence-DPO updates, improved chosen/rejected logit margin,
original-state inputs, immutable source/reference/RNG, no-pair handling, ordinary
and SST checkpoint round-trips, protected rejection, nested held-out/stale/token
failures, changed registered PEO, explicit beta, review-attempt bounds, sanitized
review exceptions and retained evaluation-failure evidence. The budget and private
review-exception checks were separately observed RED before their fixes.

All 260 focused existing/new DeLoop, LoCD, PEO/coevolution, native backend and SST
checks pass. Syntax, Black and 100-column checks pass. Full local verification is
running; the environment's known isolated-build/network and distributed-socket
limitations are not skipped or weakened. One fresh independent review and full
Python 3.10/3.12 remote CI remain necessary before merge. No research was run.

## Independent review

One fresh read-only reviewer independently passed 285 checks across the native
DeLoop/miner/LoCD/coevolution, Torch backend, SST/probe and PEO continuity suites.
No Critical or Important findings. The Minor stale preamble claiming native DPO
was absent has been corrected. No second review seat is requested.

Declined judgments are resolved explicitly: behavioral efficacy/paper reproduction
and the GRPO-penalty comparison are unrun research, excluded by the operator;
CUDA/AMP/LoRA/distributed DPO remain Unsupported. Actual evidence, source-file
authenticity, normalization and installed reviewer independence remain trusted
host responsibilities. Arbitrarily malicious closures/introspection/concurrency
are outside the synchronous host contract; ordinary packet mutation and errors
are tested. Deployment is governed separately and never granted here. Full local
and final remote CI are separate pending qualification; prior unchanged source
citation history is not re-researched. No empirical effectiveness is asserted.

## Full local verification limits

Full local pytest completed: **6,897 passed, 11 skipped, 10 failed, 37 errors**.
The failures are isolated wheel dependency downloads blocked by the sandbox,
localhost/Gloo socket creation blocked by the sandbox, and four notebook-file
checks because this source snapshot omits upstream notebook files. The complete
remote tree preserves those upstream files. No test was weakened or skipped to
mask these limitations. Full remote CI remains required. Nonisolated wheel/source
equality and pure default imports pass; the final wheel SHA256 is
`cb17d8cbe1a26f3353008901c43f4ef35c5c31879d90213e63f4fb6084214807`.

Exact local failure/error identities (full trace in the local verification log):

```text
FAILED tests/test_controlled_evolution_package.py::test_wheel_bundles_readable_controlled_evolution_resources
FAILED tests/test_llama_cpp_vulkan_integration.py::test_gguf_suffix_without_trusted_format_metadata_remains_unknown[model0]
FAILED tests/test_llama_cpp_vulkan_integration.py::test_gguf_suffix_without_trusted_format_metadata_remains_unknown[model1]
FAILED tests/test_notebooks_smoke.py::test_notebook_final_cell_contains_cli[notebook_path0]
FAILED tests/test_notebooks_smoke.py::test_notebook_final_cell_contains_cli[notebook_path1]
FAILED tests/test_notebooks_smoke.py::test_notebook_mentions_abstention_threshold[notebook_path0]
FAILED tests/test_notebooks_smoke.py::test_notebook_mentions_abstention_threshold[notebook_path1]
FAILED tests/test_rl_adapters.py::test_adapter_import_succeeds_from_an_installed_wheel
FAILED tests/test_rl_cuda.py::test_ddp_multi_trajectory_evaluation_converges_across_ranks
FAILED tests/test_rl_cuda.py::test_ddp_grpo_skips_all_ranks_when_only_one_rank_has_reward_variance
ERROR tests/test_llama_cpp_vulkan_backend.py::test_invalid_custom_request_seed_causes_zero_llama_http_requests
ERROR tests/test_llama_cpp_vulkan_backend.py::test_collect_cli_writes_common_nullable_trajectory_schema
ERROR tests/test_llama_cpp_vulkan_backend.py::test_client_normalizes_loopback_endpoint_and_validates_health_models
ERROR tests/test_llama_cpp_vulkan_backend.py::test_client_ignores_environment_proxy_for_loopback_requests
ERROR tests/test_llama_cpp_vulkan_backend.py::test_client_rejects_redirect_without_contacting_target
ERROR tests/test_llama_cpp_vulkan_backend.py::test_missing_server_probabilities_remain_null
ERROR tests/test_llama_cpp_vulkan_backend.py::test_grouped_completions_preserve_order_and_forward_effective_sampling
ERROR tests/test_llama_cpp_vulkan_backend.py::test_greedy_trajectory_records_effective_zero_temperature
ERROR tests/test_llama_cpp_vulkan_backend.py::test_duplicate_case_and_policy_requests_have_distinct_trajectory_ids
ERROR tests/test_llama_cpp_vulkan_backend.py::test_backend_rejects_ambiguous_model_list_before_completion
ERROR tests/test_llama_cpp_vulkan_backend.py::test_backend_binds_explicit_model_to_request_and_response
ERROR tests/test_llama_cpp_vulkan_backend.py::test_backend_rejects_completion_without_bound_model_identity
ERROR tests/test_llama_cpp_vulkan_backend.py::test_capabilities_use_only_observed_server_evidence
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_malformed_or_shape_inconsistent_payload[body0-content]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_malformed_or_shape_inconsistent_payload[body1-tokens]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_malformed_or_shape_inconsistent_payload[body2-selected.*IDs]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_malformed_or_shape_inconsistent_payload[body3-bytes]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_malformed_or_shape_inconsistent_payload[body4-logprob]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_empty_probability_array_does_not_claim_token_probability_evidence
ERROR tests/test_llama_cpp_vulkan_backend.py::test_probability_validation_rejects_every_invalid_top_record[not-an-object]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_probability_validation_rejects_every_invalid_top_record[invalid_top_record1]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_probability_validation_bounds_top_logprobs_to_requested_count
ERROR tests/test_llama_cpp_vulkan_backend.py::test_probability_record_text_must_reconstruct_completion_content
ERROR tests/test_llama_cpp_vulkan_backend.py::test_probability_bytes_reconstruct_utf8_split_across_tokens
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_non_utf8_non_json_and_non_finite_json[\xff\xfe-UTF-8]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_non_utf8_non_json_and_non_finite_json[not json-JSON]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_completion_rejects_non_utf8_non_json_and_non_finite_json[{"content": NaN}-non-finite]
ERROR tests/test_llama_cpp_vulkan_backend.py::test_client_rejects_oversized_response
ERROR tests/test_llama_cpp_vulkan_backend.py::test_client_translates_http_failure
ERROR tests/test_llama_cpp_vulkan_backend.py::test_client_translates_timeout
ERROR tests/test_llama_cpp_vulkan_backend.py::test_client_enforces_total_deadline_for_slow_drip_body
ERROR tests/test_llama_cpp_vulkan_backend.py::test_close_is_idempotent_and_prevents_more_requests
ERROR tests/test_llama_cpp_vulkan_backend.py::test_backend_rejects_invalid_request_before_server_call
ERROR tests/test_llama_cpp_vulkan_backend.py::test_backend_rejects_seed_expansion_overflow_before_health_or_models
ERROR tests/test_llama_cpp_vulkan_backend.py::test_backend_forwards_largest_portable_seed_to_llama_cpp
ERROR tests/test_llama_cpp_vulkan_backend.py::test_backend_close_is_idempotent
ERROR tests/test_llama_cpp_vulkan_integration.py::test_reachable_endpoint_does_not_prove_vulkan
```
