import argparse
import math
from pathlib import Path
from typing import Final, cast

import seaborn
import torch
import torch.nn.functional as F
import torch.optim
import torchmetrics
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from kymatio.torch import Scattering2D
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
    print(f"Running on {device} device")

    MAX_V_ANGLE = math.radians(60)
    MAX_H_ANGLE = math.radians(60)

    curet: Final = SimpleCuret(CURET_ROOT_PATH, transform=ts)
    # dataset: Final = CuretSubset(curet, max_view_angles=[
    #                              MAX_V_ANGLE, MAX_H_ANGLE])

    train_loader = DataLoader(curet, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=workers, pin_memory=pinning)
    test_loader = train_loader
    # test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
    #                          num_workers=workers, pin_memory=pinning)

    inputs, outputs = K * SCAT_M_O * SCAT_N_O, len(curet.classes)
    # model: Final = ScatNet2D(inputs, outputs, args.classifier).to(device)
    model: Final = LinearSVM(inputs, outputs).to(device)

    criterion: Final = CrossEntropyLoss(reduction='sum')
    optimizer: Final = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    scheduler: Final = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.9)

    metric: Final = torchmetrics.Accuracy().to(device)

    print(f"Run with {args.classifier} classifier for {args.epochs} epochs")

    # recover last checkpoint
    check_point_path: Final = MODEL_SAVE_PATH.joinpath(
        args.classifier).with_suffix(".pth")
    if check_point_path.exists():
        print(f"Loading last checkpoint from {check_point_path.resolve()}")
        checkpoint: Final = torch.load(check_point_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found")

    with SummaryWriter() as w:
        for order in (1, 2):

            scattering: Final = Scattering2D(
                J=J, shape=(SCAT_M_I, SCAT_N_I), max_order=order, backend='torch').to(device)

            for epoch in range(args.epochs):
                # _ = train(model, device, train_loader, optimizer, criterion, scattering)
                # test(model, device, test_loader, criterion, scattering)

                print(f"Training epoch {epoch}")

                model.train()
                for i, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()

                    coeffs = cast(torch.Tensor, scattering(data))
                    coeffs = coeffs.view(coeffs.size(0), -1)
                    output = model(coeffs)
                    # loss = criterion(output, target)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

                    acc = metric(output, target)
                    print(
                        f"Batch {i:3}:\taccuracy: {acc:.6f}\tloss: {loss:.6f}")

                    # if batch_idx % 3 == 0:
                    #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                    #         epoch, batch_idx *
                    #         len(data), len(train_loader.dataset),
                    #         100. * batch_idx / len(train_loader), loss.item()))

                    w.add_scalar('loss/train', loss.item(), global_step=epoch)
                    w.add_scalar('accuracy/train', acc, global_step=epoch)

                model.eval()
                loss = 0
                correct = 0
                test_metric = torchmetrics.Accuracy().to(device)
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)

                        coeffs = cast(torch.Tensor, scattering(data))
                        coeffs = coeffs.view(coeffs.size(0), -1)
                        output = model(coeffs)
                        # sum up batch loss
                        loss += criterion(output, target).item()
                        # get the index of the max log-probability
                        pred = output.max(1, keepdim=True)[1]
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        test_metric.update(output, target)

                test_loss = loss / len(test_loader.dataset)
                test_accuracy = correct / len(test_loader.dataset)
                # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                #     test_loss, correct, len(test_loader.dataset),
                #     100. * correct / len(test_loader.dataset)))

                test_acc = test_metric.compute()
                print(f"Eval:\taccuracy {test_acc:.6f}\tloss: {loss:.6f}")

                w.add_scalar('loss/test', test_loss, global_step=epoch)
                w.add_scalar('accuracy/test', test_acc, global_step=epoch)

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
