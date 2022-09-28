import argparse
import io
import math
from pathlib import Path
from typing import Final, cast

import seaborn
import torch
import torch.nn.functional as F
import torch.optim
import torchmetrics
from kymatio.torch import Scattering2D
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from curet.data import SimpleCuret
from curet.utils import CuretSubset
from scatnet import LinearSVM, ScatNet2D


def main():
    CURET_ROOT_PATH = Path(R"C:/data/curet/")
    assert CURET_ROOT_PATH.is_dir()

    MODEL_SAVE_PATH = Path(R"./.checkpoints/")
    assert MODEL_SAVE_PATH.is_dir()

    BATCH_SIZE = 128  # TODO: make customizable

    parser = argparse.ArgumentParser(
        description='CURET scattering  + hybrid examples')
    parser.add_argument('--epochs', type=int, default=10,
                        help='training epochs')
    parser.add_argument('--mode', type=int, default=1, choices=(1, 2),
                        help='scattering 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='cnn', choices=('cnn', 'mlp', 'lin'),
                        help='classifier model')
    args: Final = parser.parse_args()

    device: Final = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    J = 2
    K = {1: 17, 2: 81}[args.mode]

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

    MAX_V_ANGLE = math.radians(60)
    MAX_H_ANGLE = math.radians(60)

    curet: Final = SimpleCuret(CURET_ROOT_PATH, transform=ts)
    dataset: Final = CuretSubset(curet, max_view_angles=[MAX_V_ANGLE, MAX_H_ANGLE])

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=workers, pin_memory=pinning)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=workers, pin_memory=pinning)
    print(f"Dataset: CURET with {len(curet.classes)} classes")

    inputs, outputs = K * SCAT_M_O * SCAT_N_O, len(curet.classes)
    # model: Final = ScatNet2D(inputs, outputs, args.classifier).to(device)
    model: Final = LinearSVM(inputs, outputs).to(device)

    criterion: Final = CrossEntropyLoss(reduction='sum')
    optimizer: Final = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler: Final = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9)

    metric: Final = torchmetrics.Accuracy().to(device)

    print(f"Classifier: {args.classifier}")
    print(f"Training for {args.epochs} epochs with batch size {BATCH_SIZE}")

    # recover last checkpoint
    check_point_path: Final = MODEL_SAVE_PATH.joinpath(
        args.classifier).with_suffix(".pth")
    if check_point_path.exists():
        print(f"Loading last checkpoint from {check_point_path.resolve()}")
        checkpoint: Final = torch.load(check_point_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found")

    baseloss: Final = torchmetrics.HingeLoss().to(device)
    accuracy: Final = torchmetrics.Accuracy().to(device)
    all_test_metrics: Final[torchmetrics.Metric] = [baseloss, accuracy]
    with SummaryWriter() as w:
        for order in (1, 2):

            scattering: Final = Scattering2D(
                J=J, shape=(SCAT_M_I, SCAT_N_I), max_order=order, backend='torch').to(device)

            for epoch in range(args.epochs):
                print()
                print(f"Training epoch {epoch}")

                model.train()
                for i, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()

                    coeffs = cast(torch.Tensor, scattering(data))
                    coeffs = coeffs.view(coeffs.size(0), -1)
                    predis = model(coeffs)
                    predis = criterion.forward(predis, target)
                    predis.backward()
                    optimizer.step()

                    acc = metric(predis, target)
                    print(f"Batch {i}: accuracy={acc:.2f} loss={predis:.6f}")

                    w.add_scalar('loss/train', predis.item(), global_step=epoch)
                    w.add_scalar('accuracy/train', acc, global_step=epoch)

                model.eval()
                for m in all_test_metrics:
                    m.reset()
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)

                        coeffs = cast(torch.Tensor, scattering(data))
                        coeffs = coeffs.view(coeffs.size(0), -1)
                        predis = model(coeffs)

                        for m in all_test_metrics:
                            m.update(predis, target)

                loss = baseloss.compute()
                acc = accuracy.compute()
                print(f"Eval: accuracy={acc:.2f} loss={loss:.6f}")

                w.add_scalar('loss/test', baseloss, global_step=epoch)
                w.add_scalar('accuracy/test', acc, global_step=epoch)

                scheduler.step()

            hparams = {"scat-coeff-order": order}
            metrics = {"loss/test": 0}
            w.add_hparams(hparamas_dict=hparams, metrics_dict=metrics)

            model.eval()
            with torch.no_grad():
                confmat: Final = torchmetrics.ConfusionMatrix(
                    num_classes=len(dataset.classes))
                confmat.to(device)
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    preds = model(scattering(data))
                    confmat.update(preds, target)

                mat = confmat.compute().cpu().numpy()
                img = seaborn.heatmap(mat).get_figure()
                # TODO: add matrix to tensorboard output
                # w.add_image("confmat", img)


if __name__ == '__main__':
    main()
