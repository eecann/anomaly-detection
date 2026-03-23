from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights, ResNet18_Weights

from dataset_loader import Sample, prepare_splits


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainConfig:
    data_root: Path
    model_name: str
    num_classes: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    output_path: Path


class CrimeImageDataset(Dataset):
    def __init__(self, samples: List[Sample], transform: transforms.Compose | None = None) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, sample.class_idx


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, val_transform


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model

    raise ValueError("model_name must be one of: resnet18, densenet121")


def create_loaders(
    train_samples: List[Sample],
    val_samples: List[Sample],
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    train_transform, val_transform = build_transforms(image_size=image_size)

    train_ds = CrimeImageDataset(train_samples, transform=train_transform)
    val_ds = CrimeImageDataset(val_samples, transform=val_transform)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
        total_count += batch_size

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return {"loss": avg_loss, "acc": avg_acc}


def print_shape_summary(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> None:
    model.eval()
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

    print("\nTensor shape ozeti:")
    print(f"  Input images shape : {tuple(images.shape)}  -> [B, C, H, W]")
    print(f"  Input labels shape : {tuple(targets.shape)} -> [B]")
    print(f"  Logits shape       : {tuple(logits.shape)}  -> [B, {num_classes}]")
    print(f"  Probabilities shape: {tuple(probs.shape)}   -> [B, {num_classes}]")
    print("  C=3 (RGB), H=W=image_size, B=batch size")


def train_and_validate(config: TrainConfig) -> None:
    set_global_seed(config.seed)

    class_to_idx, split_to_samples = prepare_splits(
        data_root=config.data_root,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )

    if len(class_to_idx) != config.num_classes:
        raise ValueError(
            f"Beklenen sinif sayisi {config.num_classes}, bulunan {len(class_to_idx)}. "
            "Dataset klasorlerini kontrol edin."
        )

    train_samples = split_to_samples["train"]
    val_samples = split_to_samples["val"]

    if len(train_samples) == 0 or len(val_samples) == 0:
        raise ValueError("Train veya val split bos. Oranlari ve dataset yapisini kontrol edin.")

    print("\nClass-to-index mapping:")
    for class_name, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx:2d} -> {class_name}")

    print(f"\nTrain sample sayisi: {len(train_samples)}")
    print(f"Val sample sayisi  : {len(val_samples)}")

    train_loader, val_loader = create_loaders(
        train_samples=train_samples,
        val_samples=val_samples,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config.model_name, num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print(f"\nDevice: {device}")
    print(f"Model : {config.model_name}")
    print_shape_summary(model, train_loader, device, config.num_classes)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_one_epoch(model, val_loader, criterion, None, device)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {
                "model_name": config.model_name,
                "num_classes": config.num_classes,
                "class_to_idx": class_to_idx,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "image_size": config.image_size,
            }

    if best_state is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, config.output_path)
        print(f"\nEn iyi model kaydedildi: {config.output_path}")
        print(f"Best val acc: {best_val_acc:.4f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="14-class UCF-Crime image classification trainer")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "densenet121"])
    parser.add_argument("--num-classes", type=int, default=14)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints") / "best_model.pt",
    )
    args = parser.parse_args()

    return TrainConfig(
        data_root=args.data_root,
        model_name=args.model,
        num_classes=args.num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        output_path=args.output,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_validate(cfg)
