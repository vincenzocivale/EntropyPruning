"""Phase 1: Fine-tune UNI ViT-L with LoRA + classification head."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import set_seed, get_device
from src.models import UNILoRAClassifier
from src.data.loaders import build_loaders


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    # accumulate on GPU — pull to CPU once per epoch instead of once per batch
    total_loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    with torch.set_grad_enabled(training):
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.detach() * len(labels)
            correct += (logits.argmax(1) == labels).sum()
            total += len(labels)

    return (total_loss / total).item(), (correct / total).item()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train classifier")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    dataset_name = Path(args.data_dir).name
    output_dir = Path(args.output_dir) if args.output_dir else \
        Path(f"/raid/DATASETS/checkpoints-Attention-Pruning/{dataset_name}/uni_finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names, n_classes = \
        build_loaders(args.data_dir, args.img_size, args.batch_size,
                      args.num_workers, drop_last_train=True)

    model = UNILoRAClassifier(n_classes).to(device)

    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": args.lr_backbone},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler('cuda')

    best_val_acc, best_epoch = 0., 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scheduler, scaler)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc, best_epoch = vl_acc, epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  Saved best model (val_acc={best_val_acc:.4f})")

    print(f"\nBest val_acc={best_val_acc:.4f} @ epoch {best_epoch}")

    # Test evaluation
    model.load_state_dict(
        torch.load(output_dir / "best_model.pt", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))


if __name__ == "__main__":
    main()
