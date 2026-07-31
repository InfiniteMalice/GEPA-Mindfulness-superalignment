# Task 1 report: canonical runtime configuration

## Scope

Added a frozen, strictly validated canonical RL configuration schema and two CPU-only PyTorch
examples. The existing trainer and training configuration loader APIs remain available; they now
emit `DeprecationWarning` at their legacy loader boundary and expose the canonical loader and
translator.

## TDD evidence

RED command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

RED output: `ModuleNotFoundError: No module named 'gepa_mindfulness.training.runtime_config'`
during collection of `tests/test_rl_runtime_config.py`; exit code 1.

Initial GREEN command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py -q
```

Initial GREEN output: `8 passed in 1.19s`; exit code 0.

Final GREEN command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

Final GREEN output: `17 passed in 0.67s`; exit code 0.

## Compatibility behavior

- `RLRunConfig.from_mapping()` only parses canonical nested sections and emits no warning.
- `load_rl_config()` parses canonical files without warnings; flat or old nested files are routed
  through `translate_legacy_config()`.
- `translate_legacy_config()`, `load_trainer_config()`, and `load_training_config()` emit
  `DeprecationWarning(..., stacklevel=2)`.
- `load_trainer_config()` still returns the existing PPO/GRPO trainer dataclasses.
- `load_training_config()` still returns the existing `TrainingConfig`.

## Validation and checks

- Canonical runtime, policy, algorithm, reward, dataset, checkpoint, and logging dataclasses are
  frozen.
- Every canonical section rejects unknown keys; invalid device strings and non-string device
  values are rejected without coercion.
- Both `configs/rl/pytorch_cpu_ppo.yaml` and `configs/rl/pytorch_cpu_grpo.yaml` are exercised by
  the canonical loader test.
- `black --check --line-length 100` passed for all changed Python files.
- `ruff check` passed for all changed Python files.
- `compileall -q` passed for the changed training modules.
- Source line-length check confirmed all changed Python source lines are at most 100 characters.
- `git diff --check` passed.

`mypy gepa_mindfulness/training/runtime_config.py` could not complete because the configured
Python 3.10 check parses the installed `numpy` stub, whose `type` statement requires Python 3.12.
This is an environment/dependency incompatibility; it did not report a project-source error.

## Self-review

- Canonical parsing remains warning-free, while all newly introduced deprecation warnings are
  limited to legacy loaders/translators.
- Translation preserves flat `trainer_type` configurations and the existing nested `ppo`/`grpo`
  shape, including the required GRPO group-size case.
- The implementation adds no filesystem side effects beyond reading requested configuration files.
- No documentation precision BLOCK applies: the new YAML files are concrete configuration
  examples, and their observable behavior is covered by loader tests.

## Changed files

- `gepa_mindfulness/training/runtime_config.py`
- `gepa_mindfulness/training/config.py`
- `gepa_mindfulness/training/configs.py`
- `configs/rl/pytorch_cpu_ppo.yaml`
- `configs/rl/pytorch_cpu_grpo.yaml`
- `tests/test_rl_runtime_config.py`
- `tests/test_config.py`
- `tests/test_training_configs.py`

## Commit

`feat: add canonical RL runtime config`

## Review fix round 1/5

### Dispositions

1. Fixed canonical-file detection so partial canonical mappings, including `{seed: 7}` and a
   canonical `dataset.train_path`, use `RLRunConfig.from_mapping()` without a warning.
2. Preserved established `dataset.path` during legacy translation as canonical
   `dataset.train_path`; verified against `configs/training/phi3_dual_path.yml`.
3. Rejected `nan`, positive infinity, and negative infinity for every canonical floating-point
   algorithm and reward field through the shared numeric parser.
4. Kept direct `translate_legacy_config()` warning attribution at its external caller and moved
   legacy-file warning emission to `load_rl_config()`, also with `stacklevel=2`, so both public
   entry points identify their external caller.
5. Added regression coverage for partial canonical files, the real legacy file, all canonical
   floating-point paths, and exact warning filename/line attribution.

### TDD evidence

RED command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

RED output: `12 failed, 18 passed in 1.31s`; exit code 1. Failures covered both partial canonical
files, the real legacy dataset, eight non-finite float cases, and loader warning attribution.

GREEN command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

GREEN output: `30 passed in 0.81s`; exit code 0.

### Static checks

- Black check: `6 files would be left unchanged`; exit code 0.
- Ruff: `All checks passed!`; exit code 0.
- Compileall for `runtime_config.py`: exit code 0.
- Changed Python line-length check: all lines at most 100 characters; exit code 0.
- `git diff --check`: exit code 0.

### Self-review

- Canonical and legacy detection now uses explicit legacy markers plus the established
  `dataset.path` spelling; unknown canonical keys still reach strict canonical validation.
- Compatibility-only string-to-number handling lives exclusively in the private legacy
  translator. Canonical parsing still rejects type coercion.
- The public translator and loader each emit one warning from their own boundary, avoiding an
  internal warning frame and duplicate warnings.
- Frozen dataclasses, strict devices, existing loader return types, and both CPU YAML files are
  unchanged.
- Documentation precision review found no BLOCK or WARN: the appended dispositions and commands
  identify actors, inputs, behavior, and verification results explicitly.

### Fix commit

`fix: harden canonical RL config compatibility`

## Review fix round 2/5

### Disposition

- Canonical section presence now takes precedence over all legacy root markers. Mixed files are
  parsed canonically, so the legacy root keys fail strict unknown-key validation instead of being
  translated.
- All ten legacy markers recognized by the compatibility detector are covered in mixed-format
  regression cases.
- Dataset detection inspects subkeys: `train_path`, `validation_path`, or `format` selects strict
  canonical parsing; legacy `path` alone selects translation; canonical plus legacy or unknown
  subkeys are rejected by canonical validation.
- Pure legacy device, GRPO, and `dataset.path` configurations remain translatable.
- Both warning-attribution tests now assert that exactly one warning is emitted.

### TDD evidence

RED command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

RED output: `11 failed, 34 passed, 11 warnings in 1.71s`; exit code 1. All ten mixed
canonical/legacy root-marker cases and the dataset containing both `train_path` and `path` failed
because they were incorrectly translated.

GREEN command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

GREEN output: `45 passed in 1.02s`; exit code 0.

### Static checks

- Black check: `6 files would be left unchanged`; exit code 0.
- Ruff: `All checks passed!`; exit code 0.
- Compileall for `runtime_config.py`: exit code 0.
- Changed Python line-length check: all lines at most 100 characters; exit code 0.
- `git diff --check`: exit code 0.

### Self-review

- Detection precedence is explicit and local: canonical sections, canonical dataset subkeys,
  legacy dataset `path`, legacy root markers, then strict canonical fallback.
- The strict fallback ensures unknown-only mappings are rejected rather than silently translated.
- No schema, frozen dataclass, device, numeric, loader-return, warning-stacklevel, or example-YAML
  behavior changed outside the reviewed classification fix.
- Documentation precision review found no BLOCK or WARN; the detector rules and observable
  outcomes are explicit and covered by focused tests.

### Fix commit

`fix: reject mixed canonical legacy RL config`

## Review fix round 3/5

### Disposition

- Inventoried the repository's legacy dataset mappings. The two training configs use exactly
  `path`, `train_split`, `val_split`, and `test_split`; these keys now form the explicit legacy
  dataset classification allowlist.
- A dataset containing `path` is legacy only when every nested key belongs to that allowlist.
  Adding a typo, an unknown key, or any canonical dataset key routes the whole file through strict
  canonical parsing and produces an unknown-key error.
- The direct legacy translator independently rejects unknown and mixed canonical dataset keys, so
  callers cannot bypass the loader's classification gate.
- Legacy split metadata is type-, finiteness-, and range-validated. The canonical runtime consumes
  only the resolved training path, so valid split metadata is intentionally omitted after
  validation; this compatibility decision is documented beside the validator.
- Real legacy config translation and canonical dataset parsing remain covered and unchanged.

### TDD evidence

RED command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

RED output: `5 failed, 53 passed, 5 warnings in 1.76s`; exit code 1. Failures covered `path` plus a
typo, three invalid split values, and an unknown nested key passed directly to the translator.

GREEN command:

```powershell
& 'C:\Users\evanh\Documents\Codex\work\g\Scripts\python.exe' -m pytest tests/test_rl_runtime_config.py tests/test_config.py tests/test_training_configs.py -q
```

GREEN output: `58 passed in 1.42s`; exit code 0 with no warning summary.

### Static checks

- Black check: `6 files would be left unchanged`; exit code 0.
- Ruff: `All checks passed!`; exit code 0.
- Compileall for `runtime_config.py`: exit code 0.
- Changed Python line-length check: all lines at most 100 characters; exit code 0.
- `git diff --check`: exit code 0.

### Self-review

- The legacy allowlist is defined once and shared by classification and translation validation.
- Unknown nested data cannot be silently ignored by either public entry point.
- Invalid dataset container and path types continue to fail at the strict mapping/string boundary;
  split metadata now has equivalent strict checks.
- Frozen configs, canonical unknown-key/device/numeric validation, loader return types, warning
  attribution, partial canonical configs, real legacy translation, and CPU examples are preserved.
- Documentation precision review found no BLOCK or WARN; the accepted legacy keys and intentional
  split-metadata disposition are explicit and verified.

### Fix commit

`fix: validate legacy RL dataset compatibility`
