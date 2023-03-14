import argparse
import math
from pathlib import Path
from typing import Final, cast

import seaborn
import torch
import torch.optim
import torch.utils.data
import torchmetrics
from kymatio.torch import Scattering2D
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from curet.data import SimpleCuret, SAMPLES_PER_CLASS, curet_meta_table
from curet.utils import CuretViewSubset, CuretClassSubset
from models.scatnet import LinearSVM


CURET_ROOT_PATH = Path(R"C:/data/curet/")
assert CURET_ROOT_PATH.is_dir()

MODEL_SAVE_PATH = Path(R"./.checkpoints/")
assert MODEL_SAVE_PATH.is_dir()


def check_point_path(classifier: str, order: int) -> Path:
    return MODEL_SAVE_PATH.joinpath(f"model_{classifier}_order_{order}.pth")


def load_model_params(path: Path, model: torch.nn.Module):
    if path.exists():
        print(f"Loading last checkpoint from {path}")
        checkpoint: Final = torch.load(path)
        model.load_state_dict(checkpoint) # checkpoint['model_state_dict'])
    else:
        print("No checkpoint found")


def save_model_params(path: Path, model: torch.nn.Module):
    print(f"Saving last checkpoint to {path}")
    torch.save(model.state_dict(), str(path))


def main():
    BATCH_SIZE = 128  # TODO: make customizable
    CLASSES = [x for x in range(10)]

    parser = argparse.ArgumentParser(
        description='CURET scattering  + hybrid examples')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs')
    parser.add_argument('--mode', type=int, default=1, choices=(1, 2),
                        help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='cnn', choices=('cnn', 'mlp', 'lin'),
                        help='classifier model')
    args: Final = parser.parse_args()
    epochs: Final = cast(int, args.epochs)

    device: Final = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    J = 2

    # spatial support size for input images, as done in Mallat paper
    SCAT_M_I, SCAT_N_I = 200, 200
    SCAT_M_O, SCAT_N_O = SCAT_M_I // (2 ** J), SCAT_N_I // (2 ** J)

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
    workers, pinning = (4, True) if device.type == 'cuda' else (None, False)
    print(f"Device: {device}")

    max_view_angles = (math.radians(360), math.radians(60))

    curet: Final = SimpleCuret(CURET_ROOT_PATH, transform=ts)
    # dataset: Final = CuretViewSubset(curet, max_view_angles=max_view_angles)
    # dataset: Final = CuretClassSubset(curet, classes=[x for x in range(20)])

    view_and_lumi: Final = curet_meta_table()

    # TODO: check correctness of angle range check
    visible: Final = lambda x: abs(x) <= max_view_angles[1]
    samples: Final = [i for i in range(1, SAMPLES_PER_CLASS + 1)
                      if visible(view_and_lumi[i][1])]

    numclasses = len(curet.classes)
    indices = [s + i * numclasses for s in samples for i in CLASSES]
    dataset: Final = torch.utils.data.Subset(curet, indices)

    print("Dataset: CURET with {} classes, {} samples"
          .format(len(CLASSES), len(indices)))

    # fixed generator seed for reproducible results
    rng: Final = torch.Generator().manual_seed(0)
    valid_set_size = int(0.8 * len(dataset))
    train_set_size = len(dataset) - valid_set_size
    datasets: Final = torch.utils.data.random_split(
        dataset, [train_set_size, valid_set_size], generator=rng)

    loaders: Final = [DataLoader(x, batch_size=BATCH_SIZE, shuffle=shuffle,
                                 num_workers=workers, pin_memory=pinning)
                      for x, shuffle in zip(datasets, [True, False])]

    print(f"Classifier: {args.classifier}")
    print(f"Training for {epochs} epochs with batch size {BATCH_SIZE}")


    train_accuracy: Final = torchmetrics.Accuracy().to(device)

    baseloss: Final = torchmetrics.HingeLoss().to(device)
    accuracy: Final = torchmetrics.Accuracy().to(device)
    metrics: Final[list[torchmetrics.Metric]] = [baseloss, accuracy]
    with SummaryWriter() as w:
        for order in (1, 2):
            cppath: Final = check_point_path("svm", order)

            print(f"Running with scatter coefficients order={order}")
            K = {1: 17, 2: 81}[order]

            inputs, outputs = K * SCAT_M_O * SCAT_N_O, len(curet.classes)
            # model: Final = ScatNet2D(inputs, outputs, args.classifier).to(device)
            classifier: Final = LinearSVM(inputs, outputs).to(device)

            shape = (SCAT_M_I, SCAT_N_I)
            scattering: Final = Scattering2D(
                J=J, shape=shape, max_order=order, backend='torch').to(device)

            model: Final = torch.nn.Sequential(
                scattering, torch.nn.Flatten(), classifier).to(device)

            # recover last checkpoint
            load_model_params(cppath, model)

            criterion: Final = CrossEntropyLoss(reduction='sum')
            optimizer: Final = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
            scheduler: Final = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=0.9)

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

                    # acc = metric.compute(predis, target).item()
                    # print(f"Batch {i}: accuracy={acc:.2f} loss={predis:.6f}")
                    print(f"Batch {i}: loss={output.sum():.6f}")

                    w.add_scalar('loss/train', output.item(),
                                 global_step=epoch)
                    #w.add_scalar('accuracy/train', acc, global_step=epoch)

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

                w.add_scalar('loss/test', loss, global_step=epoch)
                w.add_scalar('accuracy/test', acc, global_step=epoch)

                scheduler.step()
                # TODO: save current checkpoint
                save_model_params(cppath, model)

            hparams_as_dict = {"scat-coeff-order": order}
            metrics_as_dict = {"loss/test": 0}
            w.add_hparams(hparams_as_dict, metrics_as_dict)

            model.eval()
            with torch.no_grad():
                confmat: Final = torchmetrics.ConfusionMatrix(
                    num_classes=len(curet.classes))
                confmat.to(device)
                for data, target in loaders[1]:
                    data, target = data.to(device), target.to(device)
                    predis = model(data)
                    confmat.update(predis, target)

                mat = confmat.compute().cpu().numpy()
                img = seaborn.heatmap(mat).get_figure()
                # TODO: add matrix to tensorboard output
                # w.add_image("confmat", img)


if __name__ == '__main__':
    main()
