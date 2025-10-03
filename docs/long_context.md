# Long-Context Training and Analysis

Mindful Trace GEPA now supports traces that exceed 32k tokens by streaming
storage, configurable context windows, and paginated viewing tools.

## Configuring models for long context

Model presets expose new keys that surface rope scaling and attention
strategies. Example (`configs/models/llama3_8b_lora.yaml`):

```yaml
context:
  max_input_tokens: 32768
  sliding_window: 8192
  rope_scaling:
    type: "linear"
    factor: 4.0
attention:
  flash: true
```

At runtime, check that the base model supports the requested rope scaling mode.
If the underlying Hugging Face model rejects the setting the trainer will log a
warning and fall back to the default context limit.

## Streaming traces and sharding

`TraceArchiveWriter` automatically mirrors the primary `trace.jsonl` while
splitting large runs into `runs/shards/trace_00001.jsonl` chunks. A manifest
file (`runs/manifest.json`) records shard paths, counts, and SHA-256 checksums.
Long traces can be scored in streaming mode via:

```bash
gepa score --trace runs/trace.jsonl --out report.html --stream --sharded
```

Pass `--manifest runs/manifest.json` if the manifest lives in a different
location. Zstandard-compressed shards are detected by extension and will emit a
hint if the viewer cannot decode them in-browser.

## Token logging controls

Token logs now capture sampled log probabilities and rolling perplexity to keep
files manageable:

- `--with-logprobs/--no-with-logprobs`
- `--log-topk 3`
- `--log-every 16`

These options are available on `gepa dspy run` and reusable from training
notebooks. Tokens include chunk offsets, making it easy to cross-reference long
outputs during debugging.

## Offline viewer

The offline viewer supports:

- Pagination (`--page-size`) to limit memory usage.
- Lazy shard loading using the manifest metadata.
- Optional down-sampling of token confidence lines via `--max-points`.

Launch with:

```bash
gepa view --trace runs/trace.jsonl --tokens runs/tokens.jsonl \
         --out report_view.html --page-size 200 --max-points 5000
```

If tokens are missing or shards cannot be fetched the viewer surfaces a banner
rather than crashing.

## Retrieval and slicing

Use `mindful_trace_gepa.utils.longctx.select_top_spans` to retrieve the most
relevant trace spans before feeding them into a long-context model. The helper
supports BM25 scoring out of the box and cosine similarity when embeddings are
provided.

Example:

```python
from mindful_trace_gepa.utils.longctx import select_top_spans

spans = [
    {"text": "System review of safety incidents."},
    {"text": "Detailed financial audit"},
]

best = select_top_spans("safety incidents with mitigations", spans, k=1)
```

Combine retrieval with streaming scoring to build RAG-style pipelines that stay
within the effective context window of each model.
