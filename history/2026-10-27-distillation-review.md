# PR 804 review follow-up

The review identified an input-validation ordering bug in registered LoCD segment preparation: an invalid rollout request could trigger a store read and then an `AttributeError` before reaching the existing exact-type validation.

The correction adds `RewardRequest` to the initial exact-type guard. Disabled behavior and the later segment validation remain unchanged. A parameterized regression test supplies `None` and an unrelated object, requiring `ValueError` before registered evidence is read. The test reproduced the store-access failure before the fix and passed after it. A small import cleanup moves nonoptional imports to module scope in the adapter tests and the related coevolution test.

The task uses the previously approved fallback for the unavailable repo-quality-gate skill: specify the input contract, choose the minimal early guard, implement, test, inspect the diff, and record evidence. The receiving-code-review, test-driven-development, and verification-before-completion workflows were applied.

Validation: 367 focused CPU tests passed, with four real-PyTorch tests skipped. The changed Python files pass Black formatting, Python 3.10 syntax, and the 100-column limit. The full local suite stops during collection at `tests/test_dapo_hybrid_trainer.py` because PyTorch is not installed (six skips precede that collection error). Full repository CI remains required.

No training, reward eligibility, source-evidence, canonical-case, or governance behavior is expanded. This is a software-correctness fix, with no research-effectiveness claim.

## Trajectory binding before evidence access

A subsequent finding was verified against the current code: the registered evidence read preceded the source-trajectory identity check. The minimal correction moves that existing check before `correction_source_events`, preserving `localized source trajectory mismatch`. The existing successful-source/foreign-rollout test now rejects any store access for the foreign rollout; it failed before the ordering change and passed afterward. The model-version check still follows evidence retrieval.

Validation after this change: 367 focused tests passed, four PyTorch tests skipped; Black, Python 3.10 syntax, and 100-column checks passed for both changed Python files. Full local collection again stopped at `tests/test_dapo_hybrid_trainer.py` due to missing PyTorch, with six skips. Full repository CI remains required. The previously approved quality-gate fallback was used, and the production diff was limited to moving the existing evidence read below the trajectory guard.
