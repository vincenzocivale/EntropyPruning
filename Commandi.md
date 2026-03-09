python scripts/ablations/layer_ablation.py \
        --data-dir /raid/DATASETS/NCT-CRC-HE \
        --layers-source 2 \
        --layers-target 23 22 21 20 \
        --batch-size 32 \
        --epochs 10 \
        --wandb-project layer-ablation-targets

python scripts/ablations/layer_ablation.py \
        --data-dir /raid/DATASETS/NCT-CRC-HE \
        --layers-source 3 4 5 6 7\
        --layers-target 23 \
        --batch-size 32 \
        --epochs 10 \
        --wandb-project layer-ablation-sources