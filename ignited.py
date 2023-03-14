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
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import KMNIST
from torchvision.transforms import Compose, ToTensor

from models.convnet import ConvNet2D
from models.scatnet import ScatNet2D

device: Final = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = Compose([
    ToTensor(),
])

train_dataset = KMNIST(root="data", train=True,
                       transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

valid_dataset = KMNIST(root="data", train=False,
                       transform=transform, download=True),
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

test_img_data, _ = train_dataset[0]
shape = test_img_data.squeeze().shape
nclasses = len(train_dataset.classes)
print(shape)

model: Final = ScatNet2D(shape=shape, classes=nclasses).to(device)
criterion: Final = CrossEntropyLoss(reduction='mean')
optimizer: Final = SGD(model.parameters(), lr=0.1,
                       momentum=0.9, weight_decay=0.0005)
scheduler: Final = ExponentialLR(optimizer, gamma=0.9)


def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = batch[0].to(device), batch[1].to(device)
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


trainer = Engine(train_step)


def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch[0].to(device), batch[1].to(device)
        y_pred = model(x)
        return y_pred, y


train_evaluator = Engine(validation_step)
valid_evaluator = Engine(validation_step)


metrics = {
    "accuracy": Accuracy(is_multilabel=True, device=device),
    "loss": Loss(criterion, device=device),
}

# attach metrics to the evaluators
for name, metric in metrics.items():
    metric.attach(train_evaluator, name)

for name, metric in metrics.items():
    metric.attach(valid_evaluator, name)


to_save = {'trainer': trainer, 'model': model,
           'optimizer': optimizer, 'lr_scheduler': scheduler}
handler = Checkpoint(to_save, DiskSaver('./tmp/training', create_dir=True))
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)


@trainer.on(Events.ITERATION_COMPLETED(every=100))
def log_training_loss(engine: Engine):
    print(
        f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}], Loss: {engine.state.output:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer: Engine):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(
        f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer: Engine):
    valid_evaluator.run(valid_loader)
    metrics = valid_evaluator.state.metrics
    print(
        f"Validation Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")


logger = TensorboardLogger(log_dir="logs/")
logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)
with logger:
    trainer.run(train_loader, max_epochs=5)
