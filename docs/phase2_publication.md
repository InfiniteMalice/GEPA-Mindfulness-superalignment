# Phase 2 published source provenance

Maturity: **Diagnostic** publication/provenance record; software is **Experimental** and trained-model alignment improvement remains **Unqualified**.

The GitHub connector publishes identical Git trees but assigns new commit metadata. Stages 1–8 are byte-for-byte tree mirrors of the independently reviewed, tested local snapshots. Stage 9 first mirrors the tested snapshot at `26eba694e55654a44f5832bcb35ee5366ec4782f` (tree `804911a5844d33a8ccaf23948e2aa7922110c4bc`), then adds this provenance note, a publication manifest, the original-history bundle and an audit-import note. Runtime code and original experiment results are unchanged by that publication refinement.

| Stage | Original audited commit | Published source commit | Identical tree |
|---|---|---|---|
| 1 | `ecad6996b1e5bef2cc8adaa9731a311a8b64ebe6` | `a25941d44ee085a7f62428a7f19b80b4ce3174de` | `98b11d824a002f8b38aad3d62b64399ff45b549a` |
| 2 | `89167c2dac0c14a1dfa9177f54790c30dc88a0fd` | `1ef3377fd553aa09a4a2344ec6740e4b9e1893b4` | `23689fe34558dc6cda401e7026a4d0fdf2229ebd` |
| 3 | `e83b8712668dae8e5182c064ef119f1f3f0cbd95` | `33488e9456e370c359a899569646d3f5ff93d4a2` | `367863393cd7d8620b46c041f72febc4e7d20ccd` |
| 4 | `5e2b686af8792259a765ca353f1c93ed4ed8aead` | `50fa9d7042bb42be5a3f415da71eaa8aaa4cc5e0` | `5ced995dcebe6561f58aa2cd27d70972fdabfcfc` |
| 5 | `428678a85ed27d190f7fe79df01003cf6eb192e5` | `3973ad9953df3272649485bf0df01a2d30503293` | `a85eedddfa64bac3828679450021ea400a5497a8` |
| 6 | `5fe5453122bfbe11ee1a303af207f3be001076e3` | `11b8e568e645a751e82f42f7a945e86085f4db9d` | `f1e7d8d6332eb815c78d5aa94460fede0e949718` |
| 7 | `83ca0e276efd66aae546bb273e432c437699ab11` | `73fc4f335328dd19f54bede54421fe0a3232c52f` | `3ac0553105632b612c814506597c18012488d3fd` |
| 8 | `c92ba24040598a1918168161b7e53f7021d5b04d` | `f07db2b6a81618ea03db28cb8e499962b703075d` | `00b294dabd5e41f1b55a91f8516511b53e7282bd` |

Stage 9 original: `2aff63a289045f73115538faf1e470935a77f015`; source mirror: `26eba694e55654a44f5832bcb35ee5366ec4782f`. Existing frozen reports saying draft/local describe their audit-time status, before publication. PR timelines and the final delivery message establish subsequent merge status; those historical records are retained rather than rewritten.

## Reproduce the pinned historical audit

Ordinary inference needs none of these steps. Historical pins intentionally reference original research commits, preserving the exact versions under which tests and experiments ran. Import the bundled original history into a separate local ref before auditing a fresh connector-published checkout:

```bash
sha256sum docs/experiments/phase2_original_stages.bundle
# Expected: 5c1c2715418b9f96076d055309753194bb2914930149906ead21a8e90bfc2baf
git bundle verify docs/experiments/phase2_original_stages.bundle
git --no-lazy-fetch fetch --no-tags --no-recurse-submodules \
  docs/experiments/phase2_original_stages.bundle \
  refs/heads/codex/phase2-pr9-reproducibility:refs/remotes/phase2-original/pr9
python -m evaluation.phase2_reproducibility --enabled --repository-root . \
  --pins configs/phase2_source_pins.json --output /tmp/new-phase2-historical-audit.json
```

The bundle is 2744707 bytes, requires the already shared baseline `fcc4142b2a5f0bdd9e9f4f7a8b90d5c72d13d923`, and preserves original head `2aff63a289045f73115538faf1e470935a77f015`. It was verified with Git before publishing. It contains original Git objects, not model weights, credentials, generated authority or an executable research claim. Source import changes only a dedicated provenance ref; no working tree reset is required. Current pins cover stages 0–8; the original stage-9 manifest can additionally be audited under its original full commit SHA. All 136 original source/artifact hashes across ten stages were verified after the local stage-9 commit.

No qualified trained-model comparison ran. Synthetic/random-model observations, null intervals, falsified hypotheses and protected-check limitations remain explicit. Byte identity, software correctness, predictive fidelity, causal validity and alignment improvement are separate claims.
