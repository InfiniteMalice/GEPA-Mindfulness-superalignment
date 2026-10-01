# PR 11: relation-flip and behavioral counterfactual benchmarks

## Task spec and reconciliation

Base: merged PR-10, fd8bad1. Implement the next authorized program stage and open a
draft PR. Use Superpowers and repo-quality-gate; execute inline with one fresh
whole-branch review. Reuse this task's isolated clone on codex/relation-flip-benchmarks.

- Existing: SyntheticWorld, boolean conditions/permissions, actor-only rendering,
  permission counterfactuals, observable EvidenceReference, SystemIdentity, non-TRAIN
  eligibility, experimental mechanism audit references and 17 canonical cases.
- Partial: permission flips exist, but six requested relation flips and controlled
  paired behavioral scoring do not share a contract.
- Missing: exact single-variable pair checks, identifiability rejection, nuisance
  controls, prompt-bound observations, coverage-aware separate correctness/sensitivity
  reports and traceability for P-TTT, MechBench and quantum causal ablations.
- Redundant: a second world model, private-CoT mechanism test, new canonical case,
  reward formula, model-training loop or independent attribution framework.
- Experimental: fictional boolean fixtures and host-captured behavioral observations.
  These test intervention sensitivity; they cannot establish internal mechanism recovery.

## Design and decisions

Add synthetic_data/relation_flips.py beside existing worlds. Relation enum covers
authorization, consent, evidence support, freshness, source trust and reversibility;
also consequence acceptability and non-decisive evaluator-presence/reward-pressure
controls. Reuse permission, fact and action fields; do not alter simulator semantics.
A fixed explicitly rendered benchmark policy permits only reversible actions and
otherwise delegates its judgment to the existing world oracle. Irreversibility is
decisive only under this fictional policy, not a universal prohibition.

RelationPair stores before/after worlds and the exact target action. Its constructor
verifies the after-world equals one permitted boolean intervention on the before-world,
including unchanged provenance and visibility. Required facts must be observable;
ambiguous/masked decisive flips are rejected. Controls must preserve the oracle
decision. Surface text shares the same renderer/policy and changes only the targeted
relation. Renderers omit world/pair IDs, digests and expected decisions. Full exports
remain evaluator-only and preserve world eligibility.

Add evaluation/relation_flips.py. BehaviorObservation binds pair digest, arm,
exact prompt digest, declared SystemIdentity, a normalized proceed/abstain/investigate
decision, and observable external/output/action evidence. Host validation must verify
the record really describes that arm and system; hashes do not authenticate references.
Only the host supplies decisions, never a private-CoT classifier.

evaluate_relation_suite requires unique nonempty pairs, exactly one observation per
arm, matching prompt/system identity and no reused evidence across arms. It reports
baseline/intervention correctness separately from correctly directed decisive changes
and correct control invariance. Wrong-direction changes never count as success.
Missing relations are listed and full coverage is explicit; undefined subgroup rates
are null. Retain raw counts, per-pair observations and evaluator source records.
No aggregate score enters training rewards or grants runtime/persistence authority.

Alternatives considered: extend world simulator semantics globally (rejected because
it changes PR-9 contracts); a free-text pair judge (rejected because it cannot prove a
single decisive change); bounded typed world pairs (selected for checkable minimality).
No new dependencies. No automatic model execution; hosts can use existing generation
adapters and retain captured outcome records. No attribution signal is claimed.

## Plan

1. Add failing pair tests for nine relations, both directions, minimal differences,
   unobservability/masking, opt-in, restrictions and renderer identity; implement the
   bounded world-pair factory and contract.
2. Add failing evaluator tests for baseline-correct but insensitive behavior, reversed
   behavior, pressure susceptibility, complete/partial coverage, stale/missing/duplicate/
   private evidence and mismatched systems/prompts. Implement pure evidence-bound reports.
   Verify PR-10 provider preserves the fixture's training exclusion.
3. Add packaged guide/ADR and runnable examples. Extend REC-014 with three resolved
   sources (P-TTT 2609.35109; requested MechBench 2609.35515; quantum models 2609.23016)
   and existing C3-JEPA/CDR references, preserving reciprocal traceability mirrors.
4. Run focused, full, formatting, lint, CI/new-module type checks, build and installed
   wheel smoke. One fresh whole-branch reviewer; verify/fix actionable findings once.
   Record outcomes, commit/push feature branch and create/attach draft PR. Do not merge.

## Research grounding and limits

P-TTT Appendix D reverses preference labels while holding response content fixed and
keeps only unambiguous opposite target preferences. Transfer: valid relation reversal
and exclusion of masked/ambiguous pairs, without fast-weight updates or reward changes.
Requested MechBench is 2609.35515, not the unrelated chemical/mechanical benchmarks
with the same name. It separates phenomenal-law and mechanism recovery with scientifically
meaningful mutations. Transfer: report baseline correctness and intervention behavior
separately; this code does not perform symbolic mechanism recovery.
Quantum-model work validates signals with causal gate ablations and shows that the
same signal can reflect architectural compensation. Transfer: avoid inferring mechanism
from correlations or correctness. No quantum model or classical attribution implementation.
C3-JEPA motivates explicit task-variable binding; CDR motivates an explicit coverage
inventory. Neither a learned JEPA nor clinical knowledge-graph revision is reproduced.
Design hypothesis: these controlled pairs expose models that ignore decisive relations.
Experiment: compare constant, reversed and oracle fixture policies before any model study.

## Implementation rulings and validation

- Final full offline Python 3.12 run: **4251 passed, 18 skipped, 16 warnings** in
  206.02s. Command: `python -m pytest -q --disable-warnings`, with repository/src/modules
  on PYTHONPATH, offline Hugging Face flags and one thread per numerical library.
  After the traceability fixes, all 74 focused registry/documentation checks passed;
  rebuilt wheel/source artifacts, installed-wheel smoke, Ruff and Black also passed.
- Duplicate rendered prompt pairs are rejected even if their pair IDs or evaluator
  metadata differ. This prevents aliases from reweighting one suite. The regression
  first failed with DID NOT RAISE, then passed after prompt-pair uniqueness was enforced.
- Tasks 1 and 2 each started with a failing missing-module test before implementation.
  All 56 new tests pass. Existing world/PEO/curriculum integration plus new tests:
  140 passed, with 93% statement coverage across both new modules. Research registry
  and reciprocal documentation checks: 47 passed.
- Ruff passed; Black verified 608 files. Mypy passed the nine CI modules, two new
  modules, recommendation registry and separate logging-schema check (13 modules).
  Changed Python files satisfy the 100-character limit. Diff whitespace checks pass.
- Wheel and source distribution built. Outside-checkout wheel smoke passed source
  hashes, packaged guide examples, all nine relations, JSON restoration, admission,
  17 canonical cases, 62 references and installed CLI help.
- One fresh whole-branch reviewer (gpt-6-astra, high) found no actionable issues or
  documentation blockers. Its independent read-only smoke passed all 18 forward/reverse
  pairs together across all three renderer styles. Deferred judgment: real-model
  effectiveness and internal mechanism recovery have no empirical captures here;
  external capture authentication remains the host's responsibility. Both limitations
  are explicit in the guide. Parent validation covers full-suite and wheel checks.
- The first full-suite run found two traceability integration failures (4249 passed):
  the reader evidence list lacked the new guide, and the old REC-014 test expected
  only its original PR-7 files. Added the missing link and extended the exact expected
  inventories for REC-014 while retaining the original inventory checks for REC-011–013.
