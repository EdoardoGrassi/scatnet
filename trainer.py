import os
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import random_split, DataLoader

from datasets import KTH_TIPS_Grey


class CNN(nn.Module):
    def __init__(self, shape: tuple[int, int, int], blocks: int) -> None:
        super().__init__()

        self.layers = nn.Sequential()

        channels, w, h = shape
        CONV = 32

        self.layers.append(
            nn.Conv2d(in_channels=channels, out_channels=CONV, kernel_size=(3, 3)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        for _ in range(blocks - 1):
            self.layers.append(
                nn.Conv2d(in_channels=CONV, out_channels=CONV, kernel_size=(3, 3)))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Dropout2d(p=0.2))
        # TODO: set fixed input size for classifier
        self.layers.append(nn.LazyLinear(out_features=10))
        self.layers.append(nn.Softmax())

        # inputs = keras.Input(shape=(200,200,3))
        # x = inputs
        # for i in range(hp.Int("cnn_layers", 1, 3)):
        #     x = layers.Conv2D(
        #         hp.Int(f"filters_{i}", 32, 128, step=32), #128
        #         kernel_size=(3, 3),
        #         activation="relu",)(x)
        #     x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        # x = layers.Flatten()(x)
        # Dropout(0.2),
        # outputs = layers.Dense(units=10, activation="softmax")(x)
        # model = keras.Model(inputs=inputs, outputs=outputs)

    def forward(self, x: torch.Tensor):
        x = self.layers.forward(x)
        return x


# from https://www.kaggle.com/code/thanakornchaisen/cnn-1-kth-tips/notebook
# model = Sequential()
# model.add(Conv2D(100, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(len(label_names)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])


def load_data(root: str = "./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(200, pad_if_needed=True),
        # for grayscale images
        transforms.Normalize(mean=0.5, std=0.3),
        # for color images
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = KTH_TIPS_Grey(root=root, transform=transform, download=True)
    testset = KTH_TIPS_Grey(root=root, transform=transform, download=True)

    return trainset, testset


def train_model(config: dict, root: str = None):
    net = CNN(config["blocks"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=0.9)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainset, _ = load_data(root)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8
    )
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            checkpoint=checkpoint,
        )
    print("Finished Training")


def test_accuracy(net, device="cpu"):
    _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples: int, max_num_epochs: int, gpus_per_trial: int):
    data_root = os.path.abspath("./data")
    load_data(data_root)
    config = {
        "blocks": tune.choice([1, 2, 3]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_model, root=data_root),
        name="conv",
        resources_per_trial={"cpu": 4, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        storage_path="./results",
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(
        f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(
        f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = CNN(best_trial.config["blocks"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main(num_samples=10, max_num_epochs=20, gpus_per_trial=1)
