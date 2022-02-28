# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
from filelock import FileLock
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.tune.trial import Trial
from ray.tune.integration.comet import CometLoggerCallback

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class CometLogger(CometLoggerCallback):
    """Override Log Trial Save to save checkpoints as assets.
    This method can be modified to save checkpoints either as models
    or Artifacts depending on the use case.

    Args:
        CometLoggerCallback (_type_): _description_
    """

    def log_trial_save(self, trial: "Trial"):
        iteration = trial.last_result["training_iteration"]
        experiment = self._trial_experiments[trial]
        checkpoint_dir = trial.checkpoint.value
        experiment.log_asset_folder(checkpoint_dir, step=iteration)


def train(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def get_data_loaders():
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=64,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=False, download=True, transform=mnist_transforms
            ),
            batch_size=64,
            shuffle=True,
        )
    return train_loader, test_loader


def train_mnist(config):
    epochs = 10
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders()
    model = ConvNet().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    for epoch in range(1, epochs + 1):
        # Use range starting from 1 for step since Ray starts from step 1
        train(model, optimizer, train_loader, device)
        acc = test(model, test_loader, device)

        # Ray recommends calling checkpoint before tune.report
        # to keep checkpoint in sync
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "model.pth")
            torch.save(model.state_dict(), path)

        tune.report(mean_accuracy=acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    parser.add_argument(
        "--ray-address",
        help="Address of Ray cluster for seamless distributed execution.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        required=False,
        help="The address of server to connect to if using " "Ray Client.",
    )
    args, _ = parser.parse_known_args()

    if args.server_address:
        ray.init(f"ray://{args.server_address}")
    elif args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    # Create the Comet Logger and add tags to the tune runs to organize the experiments.
    logger = CometLogger(log_env_cpu=True, tags=["my-sweep"])
    analysis = tune.run(
        train_mnist,
        local_dir="./results",
        checkpoint_at_end=False,
        metric="mean_accuracy",
        mode="max",
        name="exp",
        callbacks=[logger],
        resources_per_trial={"cpu": 1, "gpu": int(args.cuda)},  # set this for GPUs
        num_samples=1,
        config={
            "lr": tune.grid_search([1e-4, 1e-2]),
            "momentum": tune.grid_search([0.1, 0.9]),
        },
    )

    print("Best config is:", analysis.best_config)
