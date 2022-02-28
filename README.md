# Setup

To enable Comet Logging for this example, we will have to install Ray from Source.

## 1. Create a Virtual Environment

We will use `conda` to create our virtual environment, but feel free to use your preferred method for creating Python virtual environments.

```
conda create -n comet-ray-tune python=3.8
conda activate comet-ray-tune
```

## 2. Run the Setup Helper Script

The helper script, `setup.sh` will install Ray in the virtual environment using Python wheels and symlink the `tune` library to the latest version present in the `master` branch of the `ray` repository. This is a tempororary workaround until the next offical `ray` release on PyPi.

```
chmod +x setup.sh && ./setup.sh
```

# Set your Comet Credentials

In order to run the example, you will need to set the following Comet credentials using environment variables.

```
export COMET_API_KEY=<Your Comet API Key>
export COMET_WORKSPACE=<Your Comet Workspace>
export COMET_PROJECT_NAME=<Your Comet Project Name>
```

# Run the Example

The following command will run a hyperparameter sweep using `tune` on the MNIST dataset, and log metrics, hyperparameters, and checkpoints to Comet.

Each Ray `trial_id` will be logged as an individual experiment in Comet.

```
python tune_comet_mnist_pytorch_example.py
```

# Example Tune Run in Comet

Here is an example project with a completed `tune` sweep

[Ray Tune Example Project](https://www.comet.ml/team-comet-ml/ray-tune-example/view/D2AbZI0Rh9ZdXFV5VbqwW0vCC/experiments)