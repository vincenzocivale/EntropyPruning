# Alternative Pruning Methods

EAF supports adding pruning baselines from external papers without vendoring
their full repositories into `src/`.

## External repository policy

Do not commit full third-party repositories unless their code must be imported
at runtime and no small adapter is practical. Prefer this order:

1. Clone upstream repositories outside the repo, for example under `/tmp`,
   `/data/.../external_repos`, or another local cache ignored by git.
2. Record the upstream URL, commit, license, and the specific ideas ported.
3. Add only the integration code needed by EAF under `src/models/` and cover it
   with fast smoke tests under `tests/`.

This keeps EAF reviewable and avoids mixing generated checkpoints, paper logs,
or unrelated training code with the local pipeline.

## Cropr

Source: https://github.com/benbergner/cropr

Reference inspected: `fa259e9030f5fddf4721ac75cdd18561524de6f9`

EAF includes a compact Cropr-style integration in `src/models/cropr.py`.
It uses lightweight auxiliary cross-attention heads to rank spatial patch tokens
and progressively prune them after many transformer blocks. CLS/register prefix
tokens are never scored or removed; this is intentional because EAF backbones
may have more than one prefix token.

Run Cropr fine-tuning with:

```bash
python scripts/finetune_pruned.py \
    --pruning-method cropr \
    --model-name uni \
    --dataset-name crc \
    --base-data-folder /path/to/thunder/data \
    --cropr-pruning-rate 8
```

Cropr does not use EAF's `--prune-layer` and does not need
`--forecaster-ckpt`. Its native control is `--cropr-pruning-rate`: a fixed
number of patch tokens removed by every Cropr module. If this flag is omitted,
EAF derives a constant rate from `--keep-ratio` for convenience, but the actual
Cropr schedule remains constant-rate and progressive. During training the model
optimizes the main classifier plus Cropr auxiliary heads; during evaluation it
returns only the main classifier logits.

Useful Cropr flags:

```bash
--cropr-llf / --no-cropr-llf
--cropr-pruning-rate 8
--cropr-num-queries 1
--cropr-num-heads 1
--cropr-pre-attn-norm
--cropr-q-proj
--cropr-k-proj
--cropr-v-proj
--cropr-no-mlp
--cropr-mlp-ratio 4
```

## EViT

Source: https://github.com/youweiliang/evit

Reference inspected: public `master` branch on 2026-06-18.

EAF includes a compact EViT integration in `src/models/evit.py`. It keeps the
paper's parameter-free token reorganization: selected transformer blocks rank
spatial patch tokens by class-token attention after MHSA, keep the top tokens,
and optionally fuse inattentive tokens into one extra token before the block
MLP. CLS/register prefix tokens are never pruned.

Run EViT fine-tuning with:

```bash
python scripts/finetune_pruned.py \
    --pruning-method evit \
    --model-name uni \
    --dataset-name crc \
    --base-data-folder /path/to/thunder/data \
    --evit-drop-loc 3,6,9 \
    --evit-base-keep-rate 0.7 \
    --evit-fuse-token
```

EViT does not use an EAF `AttentionForecaster` and does not use
`--prune-layer`. Its native controls are `--evit-drop-loc`, the block indices
where token reorganization is applied, `--evit-base-keep-rate`, the per-shrink
keep rate, and `--evit-fuse-token` / `--no-evit-fuse-token`. If
`--evit-base-keep-rate` is omitted, `finetune_pruned.py` uses `--keep-ratio` as
a convenience alias.

Useful EViT flags:

```bash
--evit-drop-loc 3,6,9
--evit-base-keep-rate 0.7
--evit-fuse-token / --no-evit-fuse-token
--evit-shrink-start-epoch 10
--evit-shrink-epochs 0
```
