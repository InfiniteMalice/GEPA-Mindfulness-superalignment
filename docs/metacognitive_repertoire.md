# Metacognitive strategy repertoire

`verification.metacognitive_repertoire.get_repertoire()` returns fresh immutable declarations
for eight diagnostic methods: assumption/boundary audit, discriminative testing, evidence
verification, robustness/sensitivity analysis, belief revision, clarification/abstention,
evidence recovery, and transfer calibration.

Every entry has a stable ID, applicability conditions and finding kinds, required evidence,
expected outputs, verification requirements, resource implications, prohibited effects and a
source. The inspiration is [Sage-Taiyi v2](https://arxiv.org/html/2611.12762v2); these are small
repository-specific methods rather than a scientific research agent or a new policy hierarchy.

`applicable_strategies(assessment)` matches diagnostic findings to methods. It does not verify
the findings, select a final action, execute a test, or rewrite evidence. Actual selection belongs
to the bounded proposal controller. Source authenticity and test semantics remain host-owned.

Clarification and abstention are separate proposal outputs: ambiguous consequential instructions
need clarification; insufficient factual grounding may warrant epistemic abstention. The
repertoire also supports continuing when independently supported evidence makes abstention
unnecessary. Neither self-reported confidence nor fluent critique is independent verification.

Every method prohibits execution, accepted-evidence modification, reward changes, permission
grants, root-objective changes, constitutional rewriting and canonical-case changes. Unknown
fields, unsupported schemas, malformed declarations and executable output kinds are rejected.
These declarations do not replace existing governance, calibration or abstention rules.

Run `python -m pytest -q tests/test_metacognitive_repertoire.py` for method contracts,
applicability, exact round trips, output separation and detached catalogue reads.
