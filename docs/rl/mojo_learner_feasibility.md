# Pure Mojo learner feasibility

Evidence reviewed: 2026-08-01 (America/New_York).

## Decision

**NO-GO for replacing the Phase 5 hybrid learner with a pure Mojo learner.**

The result is **3 `supported`, 0 `unsupported`, and 6 `unknown`**. The decision rule is strict:
replacement is a no-go unless all nine capabilities are `supported`. Keep the Phase 5 Python/PyTorch
learner as the training authority. Mojo may be evaluated as an actor, coordinator, or inference
runtime without implying that it can own learning.

This assessment separates three different claims:

- A mutable tensor or an inference graph is not evidence of automatic differentiation.
- Loading or applying a trained LoRA adapter is not evidence that the runtime can train it.
- Serving a converted PyTorch model is not evidence of optimizer, checkpoint-resume, or backward
  compatibility.

## Scope and labels

A **pure Mojo learner** owns the complete update path without delegating loss differentiation,
backward kernels, optimizer steps, or trainable-state persistence to Python/PyTorch. Calling Python
for offline comparison or invoking llama.cpp after export does not make those tools part of the
learner.

Only these labels are used:

- `supported`: current official evidence documents the required capability. Where noted, support is
  conditional on an explicit input contract.
- `unsupported`: current official evidence explicitly rules the capability out.
- `unknown`: the acceptance evidence is absent or has not been demonstrated. Absence from the
  reviewed documentation is not proof that implementation is impossible.

The labels apply to narrowly defined capabilities, not to Mojo or MAX as products.

## Evidence and local environment

The official Mojo documentation and release pages reviewed here identify Mojo `1.0.0b2` as the
stable release. The MAX release page identifies MAX `26.4` as stable. Those are documentation and
release-page versions only. Neither Mojo, Pixi, nor MAX was installed or executed on the local host,
so this assessment does **not** claim a locally tested Mojo or MAX version.

Official version sources:

- <https://mojolang.org/releases/>
- <https://docs.modular.com/releases/>
- <https://mojolang.org/docs/requirements/>

Local observations:

| Item | Observed on 2026-08-01 |
| --- | --- |
| Host OS | Microsoft Windows 11 Home `10.0.26200`, 64-bit |
| CPU | AMD Ryzen 5 7520U with Radeon Graphics, x64 |
| GPU | AMD Radeon 610M, driver `32.0.21030.1005` |
| WSL | Ubuntu under WSL2; kernel `6.6.87.2-microsoft-standard-WSL2`, x86_64 |
| Windows `mojo` / `pixi` | Not found |
| Ubuntu WSL2 `mojo` / `pixi` | Not found |
| Local Mojo/MAX execution | Not tested because the toolchain is not installed |

Reproduce the non-mutating host probe in PowerShell:

```powershell
Get-Date -Format o
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption, Version, OSArchitecture
Get-CimInstance Win32_Processor |
  Select-Object Name, Architecture
Get-CimInstance Win32_VideoController |
  Select-Object Name, DriverVersion
Get-Command mojo -ErrorAction SilentlyContinue
Get-Command pixi -ErrorAction SilentlyContinue
wsl.exe --list --verbose
$wslProbe = 'command -v mojo || echo mojo:not-found; ' +
  'command -v pixi || echo pixi:not-found; uname -srm'
wsl.exe -d Ubuntu -- sh -lc $wslProbe
```

## Capability matrix

| # | Capability | Status | Current boundary |
| ---: | --- | --- | --- |
| 1 | Trainable tensors | `supported` | Mutable CPU/GPU storage substrate; no implied gradients |
| 2 | Autodiff or explicit backward | `unknown` | Forward/eager interfaces do not establish a backward path |
| 3 | Optimizer state | `unknown` | Numeric types do not establish an optimizer or resumable state |
| 4 | Transformer backward kernels | `unknown` | Reviewed attention interfaces document forward computation |
| 5 | LoRA parameter updates | `unknown` | Bespoke Mojo updates unproven; MAX LoRA is inference-only |
| 6 | Trainable checkpoint format | `unknown` | Weight loading is documented; training-state save/resume is not |
| 7 | llama.cpp adapter transfer | `supported` | Conditional on producing the exact Hugging Face PEFT inputs |
| 8 | Numerical parity against PyTorch reference | `unknown` | Forward-logit comparison does not cover gradients or updates |
| 9 | Hardware/runtime coverage | `supported` | Runtime substrate only; local GPU and learner remain unqualified |

### 1. Trainable tensors — `supported`

**Required acceptance evidence.** The learner needs mutable, typed tensor storage, allocation on a
usable device, indexed load/store operations, and shared mutation semantics sufficient to represent
parameters and gradient/optimizer buffers.

**Observed evidence.** Mojo's `LayoutTensor` manual documents CPU and GPU allocation, indexed
loading and storing, and views that share the underlying allocation. This is sufficient as a manual
mutable storage substrate. The word *trainable* here means that learner code can mutate values; it
does not mean the tensor records a gradient or participates in autodiff.

**Reproduce.** Review the allocation, load/store, and shared-view sections. After installing the
pinned toolchain, run a minimal program that allocates a tensor, mutates it through a view, and
checks the value from the original tensor on each intended device.

```powershell
curl.exe -fsSL https://mojolang.org/docs/manual/layout/tensors/ |
  Select-String -Pattern 'alloc|load|store|share'
```

Source: <https://mojolang.org/docs/manual/layout/tensors/>

### 2. Autodiff or explicit backward — `unknown`

**Required acceptance evidence.** Either a documented differentiation API must produce gradients
through the proposed loss, or the project must supply explicit backward functions validated by
finite differences for every operation in the training graph.

**Observed evidence.** The reviewed MAX eager-execution material describes immediate execution of
operations, while the Mojo standard-library index exposes available library APIs. Neither source,
as reviewed, establishes a complete differentiation tape, gradient API, or project-specific explicit
backward implementation. A documentation search is discovery evidence only; it cannot prove the
capability absent.

**Reproduce.** Inspect the current API surfaces, then require a locally executable scalar test such
as `y = sum(x*x)` whose analytic gradient agrees with a centered finite difference before changing
this label.

```powershell
curl.exe -fsSL https://docs.modular.com/develop/eager-execution/ |
  Select-String -Pattern 'eager|gradient|backward'
curl.exe -fsSL https://mojolang.org/docs/std/ |
  Select-String -Pattern 'gradient|backward|differentiat'
```

Sources:

- <https://docs.modular.com/develop/eager-execution/>
- <https://mojolang.org/docs/std/>

### 3. Optimizer state — `unknown`

**Required acceptance evidence.** A learner must update parameters with the selected algorithm and
maintain all per-parameter and global state (for example moments and step count). It must control
accumulation precision and produce the same next step after save/reload.

**Observed evidence.** Mojo documents numeric types and mixed-precision building blocks. That is
necessary for optimizer arithmetic but does not establish an optimizer API, state ownership model,
loss scaling, or restart-equivalent optimizer implementation for this learner.

**Reproduce.** Pin dtypes and hyperparameters; execute two deterministic optimizer steps; serialize
after step one; reload; and byte- or tolerance-compare parameters and every optimizer buffer after
step two against the uninterrupted run.

```powershell
curl.exe -fsSL https://mojolang.org/docs/reference/numeric-types/ |
  Select-String -Pattern 'Float16|BFloat16|Float32|precision'
```

Source: <https://mojolang.org/docs/reference/numeric-types/>

### 4. Transformer backward kernels — `unknown`

**Required acceptance evidence.** Every operation used by the target transformer, including
attention, normalization, projections, activations, masking, and the loss, needs a
device-appropriate backward implementation with shape, dtype, masking, and numerical tests.

**Observed evidence.** The current MAX kernel pages document multi-head attention and GPU flash
attention forward computations. They do not, by themselves, establish the complete transformer
backward suite required by a learner.

**Reproduce.** Inventory the exact reference graph, map each forward operation to a backward
implementation, and run directional-derivative checks plus end-to-end gradient comparisons for a
tiny transformer. Treat a forward-only kernel match as insufficient.

```powershell
curl.exe -fsSL https://docs.modular.com/max/api/kernels/nn/attention/ |
  Select-String -Pattern 'attention|backward|gradient'
$flashAttention = 'https://docs.modular.com/max/api/kernels/nn/attention/gpu/mha/flash_attention/'
curl.exe -fsSL $flashAttention |
  Select-String -Pattern 'flash|backward|gradient'
```

Sources:

- <https://docs.modular.com/max/api/kernels/nn/attention/>
- <https://docs.modular.com/max/api/kernels/nn/attention/gpu/mha/flash_attention/>

### 5. LoRA parameter updates — `unknown`

**Required acceptance evidence.** A pure Mojo learner must identify the adapter matrices as the only
trainable parameters, propagate gradients into them, apply optimizer updates, keep frozen base
weights unchanged, and demonstrate decreasing loss on a controlled example.

**Observed evidence.** A bespoke pure Mojo update path has not been demonstrated. The documented MAX
LoRA facility consumes already trained Hugging Face PEFT adapters and applies them during model
inference; the same page explicitly limits that facility to inference. That documented facility is
therefore not an update mechanism, but it does not prove that a separately implemented Mojo learner
is impossible. The capability remains `unknown`, not `unsupported`.

**Reproduce.** Hash the frozen base weights before and after a deterministic update. Assert that
only the intended LoRA tensors change, and compare their gradients and deltas with the PyTorch
reference.

```powershell
curl.exe -fsSL https://docs.modular.com/serve/lora-adapters/ |
  Select-String -Pattern 'inference only|PEFT|safetensors|trained'
```

Source: <https://docs.modular.com/serve/lora-adapters/>

### 6. Trainable checkpoint format — `unknown`

**Required acceptance evidence.** A restartable checkpoint must preserve adapter parameters,
optimizer buffers, scheduler/scaler state, global step, RNG state, dtype/shape metadata, and model
identity. Reloading it must reproduce the next update.

**Observed evidence.** MAX documents loading model weights from safetensors and GGUF, and its module
API describes symbolic `Weight` values populated from a state dictionary for execution. These
loading paths do not establish a format or API for saving and resuming the complete mutable training
state listed above.

**Reproduce.** Compare an uninterrupted two-step run with a save/reload run at the step boundary.
Require equality or a predeclared tolerance for the next loss, gradients, parameters, optimizer
buffers, step count, and RNG-derived inputs.

```powershell
$weightLoader =
  'https://docs.modular.com/max/api/python/generated/max.graph.weights.load_weights/'
curl.exe -fsSL $weightLoader |
  Select-String -Pattern 'safetensors|gguf|load'
curl.exe -fsSL https://docs.modular.com/develop/modules/ |
  Select-String -Pattern 'Weight|load_state_dict|InferenceSession'
```

Sources:

- <https://docs.modular.com/max/api/python/generated/max.graph.weights.load_weights/>
- <https://docs.modular.com/develop/modules/>

### 7. llama.cpp adapter transfer — `supported`

**Required acceptance evidence.** An official conversion and loading path must accept a defined
adapter contract. For any particular export, a smoke prompt must then show that the adapter is
applied with its intended base model, not merely accepted as a file.

**Observed evidence.** llama.cpp documents loading a GGUF LoRA adapter for inference. Its official
converter accepts a Hugging Face PEFT adapter directory. The converter expects
`adapter_config.json` plus `adapter_model.safetensors` or `adapter_model.bin`, together with
base-model configuration. This makes the transfer mechanism supported **only if** the learner can
export those exact PEFT-compatible inputs. Conversion does not repair missing metadata, incompatible
tensor names, or an unproven training checkpoint.

**Reproduce.** Given a real PEFT export directory and matching base model:

```bash
python convert_lora_to_gguf.py /path/to/peft-adapter \
  --base /path/to/base-model \
  --outfile /path/to/adapter.gguf
./llama-cli -m /path/to/base.gguf --lora /path/to/adapter.gguf \
  -p 'Adapter transfer smoke test'
```

Before conversion, reproduce the converter input contract directly from the official script:

```powershell
$converter =
  'https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_lora_to_gguf.py'
curl.exe -fsSL $converter |
  Select-String -Pattern 'adapter_config.json|adapter_model.safetensors|adapter_model.bin'
```

The command establishes transfer and inference only. It does not establish that Mojo produced a
valid training checkpoint.

Sources:

- <https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md#lora-low-rank-adaptation-adapters>
- <https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py>
- <https://github.com/ggml-org/llama.cpp>

### 8. Numerical parity against PyTorch reference — `unknown`

**Required acceptance evidence.** With matched parameters, inputs, masks, dtypes, seeds, reduction
rules, and optimizer settings, the Mojo candidate must meet declared tolerances for forward logits,
loss, adapter gradients, parameter deltas, and a short loss trajectory.

**Observed evidence.** MAX documents comparison of forward logits with a PyTorch reference using
metrics such as absolute error, cosine similarity, and divergence. That is useful but does not cover
backward gradients or optimizer updates. Mojo's `FastMathFlag` documents performance options that
can relax strict IEEE behavior, so parity tests must record the selected math mode rather than
assume identical floating-point semantics.

**Reproduce.** Emit machine-readable tensors from both implementations at each boundary. Compare
logits, loss, every LoRA gradient, and every update with fixed tolerances. Then repeat with the
exact production dtype and math flags.

```powershell
curl.exe -fsSL https://docs.modular.com/develop/logit-comparison/ |
  Select-String -Pattern 'absolute error|cosine|divergence|logit'
curl.exe -fsSL https://mojolang.org/docs/std/builtin/simd/FastMathFlag/ |
  Select-String -Pattern 'IEEE|fast math|precision'
```

Sources:

- <https://docs.modular.com/develop/logit-comparison/>
- <https://mojolang.org/docs/std/builtin/simd/FastMathFlag/>

### 9. Hardware/runtime coverage — `supported`

**Required acceptance evidence.** There must be at least one officially documented runtime path in
scope. Host qualification is a narrower follow-up claim and additionally requires a successful
local smoke test on the selected path.

**Observed evidence.** Mojo documents Windows use through WSL and an x86-64-v3 CPU baseline, with a
CPU execution path that does not require a GPU. That establishes a supported runtime substrate in
scope. The local AMD Radeon 610M does not appear in the reviewed supported-GPU list, and the absent
local toolchain prevents a CPU or GPU smoke test. Therefore this status does not claim Radeon 610M
acceleration, local host qualification, or learner support.

**Reproduce.** After installing a pinned supported release inside WSL2, record the actual versions,
verify CPU feature requirements, run the tensor smoke test on CPU, and separately test any proposed
accelerator only if it appears in the then-current support matrix.

```powershell
curl.exe -fsSL https://mojolang.org/docs/requirements/ |
  Select-String -Pattern 'WSL|x86-64-v3|AMD|GPU'
wsl.exe -d Ubuntu -- sh -lc 'uname -srm; grep -m1 "^flags" /proc/cpuinfo'
```

Source: <https://mojolang.org/docs/requirements/>

## Inference and training boundary

The official MAX development overview starts from models already trained in frameworks such as
PyTorch/Hugging Face and describes bringing them into MAX for execution and serving. MAX graph and
module documentation likewise centers on compiling an inference graph and loading learned weights.
llama.cpp describes itself as an inference engine. These are valid inference paths, not evidence for
the missing learner capabilities.

Sources:

- <https://docs.modular.com/develop/>
- <https://docs.modular.com/max/develop/graph/>
- <https://github.com/ggml-org/llama.cpp>

The allowed boundary is therefore:

```text
Phase 5 Python/PyTorch learner -> trained PEFT adapter -> GGUF conversion -> llama.cpp inference
                                      |
                                      +-> MAX inference, where model support permits
```

A future pure Mojo learner may replace the leftmost component only after it independently satisfies
all nine capabilities. Successful inference on the right cannot promote a training capability.

## Evidence refresh procedure

Reassess before any replacement proposal and whenever the pinned Mojo/MAX release, target model,
dtype, device, or adapter schema changes:

1. Record the date, official stable release pages, and the exact locally executed versions. Keep
   documentation versions separate from installed versions.
2. Re-run the host probe above and archive its output. Check the current requirements matrix rather
   than assuming that an unlisted GPU became supported.
3. Re-open every direct source in this document. Record redirects or missing pages; replace a source
   only with a current official primary source.
4. Run the executable acceptance test described for each capability against the pinned toolchain.
   Archive commands, source, inputs, outputs, tolerances, and device information.
5. Keep `unknown` when evidence is missing, ambiguous, documentation-only where execution is needed,
   or limited to forward inference. Use `unsupported` only for an explicit current exclusion.
6. Recompute the count and verdict. Replacement remains **NO-GO unless all nine are `supported`**.

Useful source-refresh commands from PowerShell:

```powershell
$urls = @(
  'https://mojolang.org/releases/',
  'https://docs.modular.com/releases/',
  'https://mojolang.org/docs/requirements/',
  'https://mojolang.org/docs/manual/layout/tensors/',
  'https://docs.modular.com/serve/lora-adapters/',
  'https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py'
)
$urls | ForEach-Object {
  $response = Invoke-WebRequest -Uri $_ -Method Head -MaximumRedirection 5
  [pscustomobject]@{ Url = $_; Status = $response.StatusCode }
}
Get-Command mojo -ErrorAction SilentlyContinue
Get-Command pixi -ErrorAction SilentlyContinue
$versionProbe = 'mojo --version 2>/dev/null || echo mojo:not-found; ' +
  'pixi --version 2>/dev/null || echo pixi:not-found'
wsl.exe -d Ubuntu -- sh -lc $versionProbe
```

An HTTP success verifies reachability, not the truth of a capability. Promotion requires the stated
acceptance evidence and a reproducible local result where execution is part of that evidence.
