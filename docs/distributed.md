# Distributed Training with Mindful Trace GEPA

This guide explains how to scale GEPA fine-tuning beyond a single GPU while keeping the default
configuration safe for CPU-based continuous integration jobs.

## Accelerate integration

1. Install the optional dependencies:

   ```bash
   pip install "accelerate>=0.30" deepspeed
   ```

2. Configure Accelerate once per machine:

   ```bash
   accelerate config
   ```

3. Launch any notebook or Python script that relies on GEPA training helpers via `accelerate launch`:

   ```bash
   accelerate launch notebooks/ft_llama3_8b_unsloth_gepa.ipynb
   accelerate launch notebooks/ft_phi3_mini_unsloth_gepa.ipynb
   ```

   The helper `mindful_trace_gepa.train.dist.get_accelerator` reads
   `configs/train/common.yaml` and automatically enables mixed precision,
   gradient accumulation, and gradient checkpointing. If the Accelerate library
   is not installed the utilities gracefully fall back to a single-process
   no-op accelerator so CPU smoke tests keep working.

## DeepSpeed launch

DeepSpeed is available by selecting the `deepspeed` backend in the config and
launching via the DeepSpeed CLI:

```bash
deepspeed --num_gpus=8 trainer/ppo_gepa.py --config configs/ppo/ppo_default.yaml
```

The default configuration uses Zero-2 with BF16 (see
`configs/deepspeed/ds_zero2.yaml`). For larger models use
`configs/deepspeed/ds_zero3.yaml` and set `distributed.zero3_offload: true` to
page parameters to CPU memory when necessary.

### Configuration reference

`configs/train/common.yaml` exposes the following knobs:

- `backend`: `"accelerate"` (default) or `"deepspeed"`.
- `mixed_precision`: `"bf16"`, `"fp16"`, or `"no"`.
- `gradient_accumulation_steps`: number of micro batches to accumulate.
- `gradient_checkpointing`: enable checkpointing for long contexts.
- `deepspeed_config`: path to a DeepSpeed JSON/YAML config file.
- `zero3_offload`: set to true to offload optimizer/parameters when using
  Zero-3.
- `find_unused_parameters`: forwarded to DistributedDataParallel when
  necessary.

### Saving adapters

Use `mindful_trace_gepa.train.dist.save_sharded` to persist LoRA/QLoRA adapters
only once from rank 0. The helper enforces synchronization barriers and avoids
partial checkpoints when multiple processes finish at different times.

### Troubleshooting

- **Accelerate not installed** – the trainer will log a warning and use the
  `NoOpAccelerator`, ensuring local development still works.
- **DeepSpeed config missing** – the helpers fall back to the default
  Accelerate path and emit a warning.
- **OOM errors** – lower `train_micro_batch_size_per_gpu` in the DeepSpeed
  config or increase `gradient_accumulation_steps` in `common.yaml`.
