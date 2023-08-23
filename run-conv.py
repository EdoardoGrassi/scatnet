
import contextlib
from typing import Final

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
# from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                            create_supervised_trainer)
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data.dataloader import DataLoader

from datasets import KTH_TIPS_Grey, KTH_TIPS_Color
from models.convnet import ConvNet
from models.scatnet import ScatNet

def main(epochs: int):
    device: Final = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomCrop(200, pad_if_needed=True),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 256


    dataset = KTH_TIPS_Color(root='data', transform=transform, download=True)
    train_subset, valid_subset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_subset, batch_size=VALID_BATCH_SIZE,
                              shuffle=False, pin_memory=True)

    test_img_data, _ = dataset[0]
    shape = test_img_data.shape
    nclasses = len(dataset.classes)
    print("Dataset format:", shape, "classes:", nclasses, "samples:", len(dataset))
    print("Dataset splits:", "train =", len(train_subset), "valid =", len(valid_subset))

    # model: Final = ConvTexNet(classes=nclasses)
    model: Final = ScatNet(shape=shape, classes=nclasses)
    criterion: Final = CrossEntropyLoss()
    optimizer: Final = SGD(model.parameters(), lr=0.1,
                           momentum=0.9, weight_decay=0.0005)
    scheduler: Final = ExponentialLR(optimizer, gamma=0.96)

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

    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(engine: Engine):
    #     print("Epoch[{}], Iter[{}], Loss: {:.2f}".format(
    #         engine.state.epoch, engine.state.iteration, engine.state.output))

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

    @trainer.on(Events.TERMINATE)
    def log_eval_results(trainer: Engine):
        pass
        # TODO: use eval split
        # valid_evaluator.run(valid_loader)
        # metrics = valid_

    # logger = TensorboardLogger(log_dir='logs/')
    # logger.attach_output_handler(
    #     trainer,
    #     event_name=Events.ITERATION_COMPLETED(every=100),
    #     tag="training",
    #     output_transform=lambda loss: {"batch_loss": loss},
    # )

    with contextlib.suppress(KeyboardInterrupt):
        print(f"Training model {model} for {epochs} epochs")
        trainer.run(train_loader, max_epochs=epochs)


if __name__ == "__main__":
    main(epochs=100)