# Online EAF training on THUNDER

This experiment measures whether adding THUNDER train tiles improves the current
EAF trained on external WSI tiles. It deliberately differs from the historical
multi-THUNDER pipeline:

- the tile foundation model is frozen;
- no classifier head or THUNDER label enters the objective;
- source tokens and target CLS-to-patch attention are computed online;
- only THUNDER `train` splits update EAF;
- validation selects the checkpoint and test is evaluated once at the end;
- dataset-balanced sampling is label-agnostic.

## Recommended three-way comparison

Use the same tile encoder, source/target layers, target normalization, EAF
architecture, seed, and evaluation datasets in every condition.

### 1. External-only checkpoint evaluated on THUNDER

```bash
python scripts/train_thunder_online_forecaster.py \
  --model-name uni \
  --base-data-folder /path/to/thunder \
  --output-dir results/thunder_online \
  --experiment-name external_only_eval \
  --init-checkpoint checkpoints/external_only/best_forecaster.pt \
  --eval-only \
  --source-layer 2 \
  --target-normalization patch \
  --batch-size 32
```

### 2. THUNDER-only online training

```bash
python scripts/train_thunder_online_forecaster.py \
  --model-name uni \
  --base-data-folder /path/to/thunder \
  --output-dir results/thunder_online \
  --experiment-name thunder_only \
  --source-layer 2 \
  --target-normalization patch \
  --sampler dataset_balanced \
  --epochs 30 \
  --batch-size 32 \
  --seed 42
```

### 3. External pretraining followed by THUNDER continuation

```bash
python scripts/train_thunder_online_forecaster.py \
  --model-name uni \
  --base-data-folder /path/to/thunder \
  --output-dir results/thunder_online \
  --experiment-name external_plus_thunder \
  --init-checkpoint checkpoints/external_only/best_forecaster.pt \
  --source-layer 2 \
  --target-normalization patch \
  --sampler dataset_balanced \
  --epochs 10 \
  --lr 3e-5 \
  --batch-size 32 \
  --seed 42
```

The external checkpoint may be either a raw `state_dict` or a dictionary using
`forecaster_state_dict`, `state_dict`, or `model_state_dict`.

## Leave-dataset-out control

To show that improvements are not limited to datasets seen during EAF training:

```bash
python scripts/train_thunder_online_forecaster.py \
  --model-name uni \
  --base-data-folder /path/to/thunder \
  --output-dir results/thunder_online \
  --experiment-name thunder_leave_three_out \
  --n-holdout 3 \
  --source-layer 2 \
  --epochs 30
```

The default evaluation set contains every downloaded THUNDER dataset, while
checkpoint selection uses validation splits only from the selected training
datasets. `data_plan.json` records the exact train, holdout, and evaluation
sets.

## Decision rule

Prefer external-only training for the main paper unless THUNDER produces a
clear and reproducible improvement on both:

1. THUNDER datasets whose train split was included; and
2. leave-dataset-out THUNDER datasets.

Report THUNDER-inclusive training as in-domain adaptation when its gain is
mainly restricted to seen datasets. Match compute with
`--max-steps-per-epoch` when the external and THUNDER corpora differ greatly in
size.

## Automated comparison

After the three runs, generate per-dataset deltas and separate seen from held-out
THUNDER datasets:

```bash
python scripts/compare_thunder_online_runs.py \
  --external-only results/thunder_online/external_only_eval/results.json \
  --candidate thunder_only results/thunder_online/thunder_only/results.json \
  --candidate external_plus_thunder results/thunder_online/external_plus_thunder/results.json \
  --output-dir results/thunder_online/comparison
```

A candidate is convincing only when KL decreases and rank/top-10% recall improve
on held-out datasets as well as on datasets whose train split was used.
