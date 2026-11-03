# Native localized correction

The operator requests working, opt-in features and has excluded research runs. This
stage connects the existing LoCD local forward-KL objective to the portable CPU
backend, without combining it with ActionOPD or changing reward eligibility.

The approved repo-quality-gate fallback applies: specification, design,
implementation, tests, independent review, and evidence. Superpowers workflows
apply; the repo-quality-gate skill remains unavailable. No beads commands or
records are changed.

## Design and acceptance

An explicitly enabled, versioned configuration bounds one localized correction.
Only an explicit TRAIN source with retained provenance, matching model checksum,
registered correction and closed source epoch is eligible. Fresh host source and
correction-use reviews authenticate the binding and observable teacher evidence.
The teacher is the unchanged source policy with privileged context: this is the
shared-model context self-distillation subset of REF-LOCD, not a new teacher API.

Deep-copy the complete portable backend, retaining optimizer and optional SST
ownership. Check that cloned tensors share no storage with the source. Generate
only the declared teacher span, align response predictors, and use the existing
masked forward KL. Student inputs contain only the original public prompt and
prefix plus the corrected continuation. Use native backward, optimizer-step and
checkpoint contracts. Preserve global RNG, teacher modes and the caller's backend.

Persist the actual isolated checkpoint under a new immutable trial directory.
Register its digest through CoevolutionStore. Independently evaluate the updated
student through a bounded generation-only interface, including a mandatory
public-input resumption. Accept only existing store-issued validation bundles
through the existing acceptance owner. Catalog acceptance never deploys a model.
An unchanged policy yields no candidate and no claim of improvement.

Deterministic CPU tests must demonstrate a real parameter update and public-only
reevaluation, source isolation, privileged-context separation, denied/mutating
reviews, held-out provenance rejection, unsupported-runtime and budget rejection,
protected regression rejection, artifact integrity, SST freeze and serialization.
Existing source/epoch, distillation, SST and reward tests remain required.

No empirical alignment, retention or efficiency improvement is claimed. Native
CUDA, mixed precision, distribution, LoRA, external teachers and multi-example
curricula are outside this bounded implementation. Research protocols remain
documented, unrun, and separately qualified.

## Initial implementation evidence

The missing-runtime import was observed RED before implementation. The native
functional and existing distillation/coevolution/backend/engine/SST suites pass:
223 checks. The original backend is unchanged in accepted, denied and failed
trials. Actual public outputs determine the synthetic evaluator's acceptance or
rejection. Native optimizer and SST checkpoint round-trips are checked, and
full-rank SST returns no-model-change without catalog admission. These results
qualify software behavior only. Independent review and final CI remain required.

## Independent review and verified fixes

One fresh read-only reviewer independently ran 223 checks and confirmed one
Important finding and one Minor finding. Both were reproduced with three failing
regressions before changes: inherited Transformers beam configuration expanded
batch-one logit allocation, and invalid validation bundles left pending audit
state without a completed public rollout. Teacher and evaluation generation now
pin one beam/sequence and disable expanded generation outputs; inherited settings
and effective overrides are recorded. Complete public evidence is saved before
the host evaluation, and all subsequent failures retain a terminal audit status.
All 29 native runtime checks pass, including the real tiny GPT-2 reproductions.

The review's declined-to-judge items are explicitly resolved as follows. Empirical
gains remain unclaimed because the operator excluded research runs. Source,
teacher-evidence and localization meaning are authenticated by installed host
reviews; the software does not certify them. Real organizational role separation
is a trusted-host responsibility, not established by different strings. Deliberate
host introspection of closures or filesystem mutation remains inside the stated
trusted-host boundary; actual artifact/model digests are still checked. Unsupported
CUDA/AMP/distribution/LoRA/external teachers/full curricula are documented without
qualification claims. Early EOS is intentionally rejected under exact-span
semantics. Local isolated-build/Gloo sandbox limits require remote CI, without
weakened tests. Earlier credit functionality is a separately reviewed parent and
is not silently reevaluated by this feature review.
