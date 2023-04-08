#!/usr/bin/env python3

import argparse
import contextlib
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import seaborn
import torch
import torch.optim
import torch.utils.data
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import KMNIST, DTD
import torchvision.transforms as transforms

from models.convnet import ConvNet2D
from models.scatnet import ScatNet2D

device: Final = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomCrop(200),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

#train_dataset = KMNIST(root='data', train=True, transform=transform, download=True)
train_dataset = DTD(root='data', split='train', transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

#valid_dataset = KMNIST(root='data', train=False, transform=transform, download=True)
valid_dataset = DTD(root='data', split='val', transform=transform, download=True)
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

test_img_data, _ = train_dataset[0]
shape = test_img_data.shape
nclasses = len(train_dataset.classes)
print("Dataset format:", shape, "classes:", nclasses, "samples:", len(train_dataset))
print("Dataset splits:", "train =", len(train_dataset), "valid =", len(valid_dataset))

model: Final = ScatNet2D(shape=shape, classes=nclasses).to(device)
#model: Final = ConvNet2D(shape=shape, classes=nclasses).to(device)
criterion: Final = CrossEntropyLoss(reduction='mean')
optimizer: Final = SGD(model.parameters(), lr=0.1,
                       momentum=0.9, weight_decay=0.0005)
scheduler: Final = ExponentialLR(optimizer, gamma=0.9)

metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion),
}


trainer: Final[Engine] = create_supervised_trainer(
    model, optimizer=optimizer, loss_fn=criterion, device=device)
train_evaluator: Final[Engine] = create_supervised_evaluator(
    model, metrics=metrics, device=device)
valid_evaluator: Final[Engine] = create_supervised_evaluator(
    model, metrics=metrics, device=device)


to_save = {'trainer': trainer, 'model': model,
           'optimizer': optimizer, 'lr_scheduler': scheduler}
saver = DiskSaver('checkpoints', create_dir=True, require_empty=False)
handler = Checkpoint(to_save, saver)
trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)


@trainer.on(Events.ITERATION_COMPLETED(every=10))
def log_training_loss(engine: Engine):
    print("Epoch[{}], Iter[{}], Loss: {:.2f}".format(
        engine.state.epoch, engine.state.iteration, engine.state.output))


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer: Engine):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print("Train - Epoch[{}] Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
        trainer.state.epoch, metrics['accuracy'], metrics['loss']))


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer: Engine):
    valid_evaluator.run(valid_loader)
    metrics = valid_evaluator.state.metrics
    print("Valid - Epoch[{}] Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
        trainer.state.epoch, metrics['accuracy'], metrics['loss']))


logger = TensorboardLogger(log_dir='logs/')
logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt):
        EPOCHS = 100
        print(f"Training model {model} for {EPOCHS} epochs")
        with logger:
            trainer.run(train_loader, max_epochs=EPOCHS)
