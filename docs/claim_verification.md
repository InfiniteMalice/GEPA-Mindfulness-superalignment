# Public claim verification

Status: experimental, opt-in, diagnostic only. No optimizer or execution authority.

`partition_claims` compares host-normalized requirement values across candidate rollouts. It
reports DISPUTED, CONSENSUS or OMITTED and separately lists candidates missing each requirement.
Each partition starts unresolved. Semantic normalization is the host's responsibility; exact
string agreement does not establish truth. `resolver_alternatives` lists competing values.

`challenge_consensus` supplies a separate falsifier checklist covering sources, time, units,
definitions, omitted requirements, stakeholders, circular evidence, injection and reward pressure.
The host selects applicable checks and records CheckRequest factors. `prioritize_checks` ranks
information gain times decision sensitivity times importance divided by cost within a fixed budget.
The runtime executes any selected check through its existing authorization path.

`adjudicate_check` checks claim/action identity, independent context declarations, authorized
evidence, and canonical LocalVerificationResult/RelationalVerificationResult findings. Every
decisive finding must intersect the check result's evidence. Missing findings, contradiction
mismatches, or host-reported evidence hazards produce unresolved output. Unauthorized evidence
raises ValueError. The original EvidenceState remains unchanged, and superseded claims cannot
be adjudicated as current. Context identifiers and evidence labels do not authenticate principals.
Hosts must perform identity authentication and evidence capture outside this diagnostic module.

To persist a revision, the caller must use the existing `commit_verified_claim` authority gate.
Failures may be attached to existing FailureNode records only after the host supplies the actual
observation event, time and evidence. No failure-cause inference is made from disagreement alone.
`check_failure_node` enforces the local action/evidence join for counterevidence; the host still
validates the full causal event sequence before persisting a failure graph.

Source-supported motivation: *VeriHarness: Scaling Agentic Verification for Long-Horizon Tasks*
([arXiv:2610.00972](https://arxiv.org/abs/2610.00972)) separates disagreement resolution from
consensus challenge using environmental evidence. This repository's exact field bindings,
authorized-boundary validation and V5 integration are design inferences. Combined improvement
remains an experimental hypothesis. Verification: `tests/test_claim_verification.py` and
`tests/test_verifier_interfaces.py`.
