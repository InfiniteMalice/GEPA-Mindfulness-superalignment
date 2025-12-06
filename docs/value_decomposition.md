# Value decomposition (deep vs shallow)

This repository now exposes an optional value decomposition layer that mirrors Deep Value
Benchmark (DVB) style evaluations. All features are disabled by default and can be toggled via
`DSPyConfig` or CLI flags.

## Structures

- **DeepValueVector**: weights for the three GEPA imperatives plus the four contemplative
  principles (mindfulness, empathy, perspective, agency).
- **ShallowPreferenceVector**: tone and style preferences (formal, casual, therapeutic), verbosity,
  hedging/directness, and deference/assertiveness.
- **GepaDecomposition**: splits a GEPA scalar into deep contribution, shallow contribution, and
  residual.
- **DVBExample / DVGR**: utilities to run DVB-style deep-vs-shallow evaluations.

DVGR is currently a placeholder diagnostic: the pipeline constructs a single contrasted pair using
the provided context as a shallow-first stand-in and emits a warning that results are not yet
meaningful until contrasted generations are available.

## Logging additions

When `enable_value_decomposition` is set, traces can contain optional fields:

- `value_decomposition.user_deep`
- `value_decomposition.user_shallow`
- `value_decomposition.output_deep`
- `value_decomposition.output_shallow`
- `value_decomposition.gepa_decomposition`
- `value_decomposition.dvgr`

Existing traces remain valid; consumers should treat these fields as optional. Attribution-graph
code can contrast deep- and shallow-driven circuits by checking these payloads while still running
unchanged on logs that omit them.
