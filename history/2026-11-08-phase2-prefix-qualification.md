# Causal-prefix qualification refinement

The resumed publication review found a substantive split leak. Full sequences can differ while
a causal transformer's activation at a measured position sees the same visible prefix. Masked
padding IDs must not manufacture distinct contexts. Original source/probe versions stay in the
bundle and retain their measured values; their old whole-sequence held-out labels do not qualify
causal-prefix isolation.

Extend ExampleRef with per-position visible-prefix digests (include absolute positions, ignore
masked token IDs) under native-example-reference-v2. Legacy v1 wire records must remain readable
and roundtrip byte-identically as inert diagnostics. They cannot authorize a new held-out result
without rebuilding references from original public inputs. New qualified freezing, evaluation,
activation simulation and dictionary comparison require fingerprints for the declared measured/
output positions and reject cross-generation/development/locked overlap before inference. Shared
BOS prefixes remain usable for discovery; qualification can select later disjoint positions.
Use strict rejection, not silently dropping measurements or inferring independence. Digests are
metadata, not source authenticity, privacy, temporal custody or semantic-independence proofs.
Trusted host authentication and independent replay remain required for evidence admission.

Update bounded synthetic fixture inputs/positions so their causal prefixes genuinely separate.
Retain old probes and rerun corrected versions under new names with model/seed/code provenance.
Near-miss context fingerprints and replacement vocabularies exclude masked padding. Add tests for
future-only differences, masked jitter, position identity, development leakage, later-position
qualification, tampered prefix metadata and exact legacy-wire preservation. Preserve all V5 and
reward/authority owners. Document the experimental qualification migration explicitly.

Other verified review corrections: include false claimed-attempt reports in the faithful-report
metric denominator; clarify target-string versus proof-integer APIs; preserve explicit unresolved
reasons after nonspecific replay; make bypass-pressure equivalence use the same legitimate action.

Ruling: standard-vocabulary causal tracing remains outside the documented bounded full-evidence
prototype, whose value/report limits reject before inference. Expanding it requires a separately
reviewed immutable external-evidence representation, not lifting the four-MiB limit or truncating
measurements. Cost: no claim of support for standard pretrained vocabulary/parameter geometries.

Ruling: artifact SHA-256 values identify retained bytes; rerunning measured probes need not recreate
wall-clock timestamps, runtime versions or timing costs. Preserve original clocks and explain
numeric versus byte identity rather than rewriting original research metadata to match a hash.

Ruling: historical manifests bind their original source commits, while publication manifests bind
review corrections. The source pins, original-history bundle and provenance guide already separate
these snapshots; do not overwrite historical hashes with current-head values.


Fresh bounded static review: one Important finding, nested v1 comparison commitments compared
against the new v2 factory and could not roundtrip. The reviewer did not run measurements because
PyTorch temporary-directory permissions blocked its attempted reproduction. Original nested probe
records independently reproduced the exact failure locally. Fix: compare legacy generation source
bindings under original raw-token/mask digest semantics while retaining the wire representation;
qualified comparison entry points still reject legacy prefix metadata before model execution.
All six original comparison commitments now roundtrip and remain ineligible for qualified execution.

Additional remote Minor findings were verified and fixed in the same pass: distinguish nonempty
short v2 prefix geometry from missing legacy metadata; prioritize declared simulation geometry and
control mismatch errors; describe the configurable default 50M (maximum configurable 1B) parameter
budget accurately; document claimed-attempt denominator opportunities; add adjacent historical
manifest snapshot identities without rewriting historical artifacts. New geometry regressions were
observed failing first. Affected suite: 120 passed in96.46s. Final full gate and source-pinned
probe reruns are recorded in the reviewed validation report. No Critical finding was identified.

Review exclusions accepted: hashes do not authenticate truth/custody/semantic independence;
shared-BOS discovery and future-context retrospective hypotheses remain permitted; standard large
pretrained geometries/external evidence storage are deferred; timestamps/costs are actual run
metadata; no trained alignment or semantic validity is inferred. Root verification independently
checks raw source hashes and actual probe outputs, not the language-model reviewer's authority.
