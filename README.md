# Setup

To enable Comet Logging for this example, we will have to install Ray from Source.

## 1. Create a Virtual Environment

We will use `conda` to create our virtual environment, but feel free to use your preferred method for creating Python virtual environments.

```
conda create -n comet-ray-tune python=3.8
conda activate comet-ray-tune
```

## 2. Run the Setup Helper Script
The helper script, `setup.sh` will install Ray in the virtual environment

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

## Example using the `CometLoggerCallback`
```
python tune_comet_callback.py
```

## Example using the Ray Functional API
```
python tune_comet_functional.py
```

## Example using the Ray Trainable Class API
```
python tune_comet_trainable.py
```

# Example Tune Run in Comet

Here is an example project with a completed `tune` sweep

[Ray Tune Example Project](https://www.comet.ml/team-comet-ml/ray-tune-example/view/D2AbZI0Rh9ZdXFV5VbqwW0vCC/experiments)