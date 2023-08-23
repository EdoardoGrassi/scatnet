#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import seaborn
import torch
import torch.optim
import torch.utils.data
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from models.convnet import ConvNet
from models.scatnet import ScatNet
from datasets import KTH_TIPS_Grey


MODEL_SAVE_PATH = Path(R"./checkpoints/")
assert MODEL_SAVE_PATH.is_dir()


def check_point_path(name: str) -> Path:
    return MODEL_SAVE_PATH / name / "cp.pth"


def load_model_params(path: Path, model: torch.nn.Module):
    if path.exists():
        print(f"Loading last checkpoint from {path}")
        checkpoint: Final = torch.load(path)
        model.load_state_dict(checkpoint)  # checkpoint['model_state_dict'])
    else:
        print("No checkpoint found")


def save_model_params(path: Path, model: torch.nn.Module):
    print(f"Saving last checkpoint to {path}")
    torch.save(model.state_dict(), str(path))


def main(batch_size: int, epochs: int):

    device: Final = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    ts: Final = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.CenterCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(mean=0.5, std=0.3),
    ])

    workers, pinning = (4, True) if device.type == 'cuda' else (2, False)
    print(f"Running on {device}")

    dataset: Final = KTH_TIPS_Grey(root="./data", transform=ts, download=True)

    # fixed generator seed for reproducible results
    rng: Final = torch.Generator().manual_seed(0)
    train_subset, valid_subset = torch.utils.data.random_split(
        dataset, [0.8, 0.2], generator=rng)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pinning)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pinning)

    print(f"Training for {epochs} epochs with batch size {batch_size}")

    baseloss: Final = torchmetrics.HingeLoss(
        task='multiclass', num_classes=len(dataset.classes)).to(device)
    accuracy: Final = torchmetrics.Accuracy(
        task='multiclass', num_classes=len(dataset.classes)).to(device)
    metrics: Final[list[torchmetrics.Metric]] = [baseloss, accuracy]

    models = {
        "convnet": ConvNet(shape=dataset[0][0].shape,
                             classes=len(dataset.classes)),
        "scatnet": ScatNet(shape=dataset[0][0].shape,
                             classes=len(dataset.classes)),
    }
    name = "covnet"
    model = ConvNet(shape=dataset[0][0].shape,
                             classes=len(dataset.classes))
    
    cppath: Final = check_point_path(name)

    print(f"Running with model {name}")

    # recover last checkpoint
    load_model_params(cppath, model)

    criterion: Final = CrossEntropyLoss()
    optimizer: Final = SGD(model.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=0.0005)
    scheduler: Final = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(epochs):
        print()
        print(f"Training epoch {epoch}")

        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            predis = model(data)
            output = criterion.forward(predis, target)
            output.backward()
            optimizer.step()

            print(f"Batch {i}: loss={output.sum():.6f}")

        model.eval()
        for m in metrics:
            m.reset()

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)

                predis = model(data)
                for m in metrics:
                    m.update(predis, target)

        loss = baseloss.compute().item()
        acc = accuracy.compute().item()
        print(f"Eval: accuracy={acc:.2f} loss={loss:.6f}")

        scheduler.step()
        # TODO: save current checkpoint
        save_model_params(cppath, model)

    model.eval()
    with torch.no_grad():
        confmat: Final = torchmetrics.ConfusionMatrix(
            num_classes=len())
        confmat.to(device)
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            predis = model(data)
            confmat.update(predis, target)

        # mat = confmat.compute().cpu().numpy()
        # img = seaborn.heatmap(mat).savefig(f"resources/{name}/confmat.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CURET scattering  + hybrid examples')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100,
                        help='training epochs')
    args: Final = parser.parse_args()

    main(args.batch_size, args.epochs)
