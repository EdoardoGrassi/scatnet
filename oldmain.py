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
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from models.convnet import ConvNet2D
from models.scatnet import ScatNet2D

CURET_ROOT_PATH = Path(R"C:/data/curet/")
assert CURET_ROOT_PATH.is_dir()

MODEL_SAVE_PATH = Path(R"./.checkpoints/")
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

    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ts: Final = transforms.Compose([
        transforms.CenterCrop(200),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # TODO: use multiple workers after concurrency bug is fixed
    workers, pinning = (4, True) if device.type == 'cuda' else (1, False)
    print(f"Running on {device}")

    dataset: Final = torchvision.datasets.KMNIST(root="data", download=True)

    # fixed generator seed for reproducible results
    rng: Final = torch.Generator().manual_seed(0)
    valid_set_size = int(0.8 * len(dataset))
    train_set_size = len(dataset) - valid_set_size
    datasets: Final = torch.utils.data.random_split(
        dataset, [train_set_size, valid_set_size], generator=rng)

    loaders: Final = [DataLoader(x, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=workers, pin_memory=pinning)
                      for x, shuffle in zip(datasets, [True, False])]

    print(f"Training for {epochs} epochs with batch size {batch_size}")

    baseloss: Final = torchmetrics.HingeLoss(
        task='multiclass', num_classes=len(dataset.classes)).to(device)
    accuracy: Final = torchmetrics.Accuracy(
        task='multiclass', num_classes=len(dataset.classes)).to(device)
    metrics: Final[list[torchmetrics.Metric]] = [baseloss, accuracy]

    test_img_data, _ = dataset[0]
    #plt.imshow(test_img_data)
    #plt.show()

    models = {
        "convnet": ConvNet2D(shape=test_img_data.size(),
                             classes=len(dataset.classes)),
        "scatnet": ScatNet2D(shape=test_img_data.size(),
                             classes=len(dataset.classes)),
    }
    for name, model in models:
        cppath: Final = check_point_path(name)

        print(f"Running with model {name}")

        # recover last checkpoint
        load_model_params(cppath, model)

        criterion: Final = CrossEntropyLoss(reduction='sum')
        optimizer: Final = SGD(model.parameters(), lr=0.1,
                               momentum=0.9, weight_decay=0.0005)
        scheduler: Final = ExponentialLR(optimizer, gamma=0.9)

        for epoch in range(epochs):
            print()
            print(f"Training epoch {epoch}")

            model.train()
            for i, (data, target) in enumerate(loaders[0]):
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
                for data, target in loaders[1]:
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
            for data, target in loaders[1]:
                data, target = data.to(device), target.to(device)
                predis = model(data)
                confmat.update(predis, target)

            mat = confmat.compute().cpu().numpy()
            img = seaborn.heatmap(mat).savefig(f"resources/{name}/confmat.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CURET scattering  + hybrid examples')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs')
    args: Final = parser.parse_args()

    main(args.batch_size, args.epochs)
