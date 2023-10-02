# FLUIDOS Energy Demand Predictor

## Overview

This is a project to predict energy demand for a FLUIDOS node. It uses a neural network that takes as input, given a
certain machine,

- the past workload of the machine (at the moment, a week's worth of data)
- a power profile of the machine (more information below)

and outputs the predicted energy demand for the following day.

-----
> **As of October 2nd, 2023, this program is meant as a proof-of-concept, still in development and is
not ready for production use. Features may be added or removed at any time. Accuracy may not be perfect.
The program is provided as-is, with no guarantees of any kind.**

> The program is currently only tested on macOS and Linux platforms. Windows support is not yet verified.
----

## Installation

Due to the machine-specific nature of the projects, no pre-trained models are provided. To use the project, you will
need to train your own model.

You may choose between the following installation methods:

### Classic

Create a `python3` virtual environment using the software of your choice (`venv`, `conda`, or anything else). Install
`python3==3.11.4` on it. Then, install the dependencies using `pip install -r requirements.txt`.

On macOS platforms running an Apple Silicon processor, you may need to install `tensorflow-macos` instead of
`tensorflow`. [This page](https://developer.apple.com/metal/tensorflow-plugin/) provides precise information on how to
correctly install the Tensorflow plugin on these platforms.

### Docker

A Dockerfile is provided within the project. If you want to build the image for yourself, run
`docker build -t fluidos-energy-demand-predictor .`

Then, run the image using the following command:

```bash
docker run -it \
    --name=predictor \
    -v /path/to/data/folder:/app/data \
    -v /path/to/models/folder:/app/models \
    -v /path/to/out/folder:/app/out \
    -e TZ=Europe/Rome \
    --restart=unless-stopped \
    ghcr.io/risingfbk/fluidos-energy-predictor:github
```

Before deploying, make sure to edit the following lines:

```bash
  -v /path/to/data/folder:/app/data
  -v /path/to/models/folder:/app/models
  -v /path/to/output/folder:/app/out
```

assigning the path to a folder on your machine to the:

- `/app/data` folder, which will be used for fetching the training data (read below for more information on the folder 
  structure);
- `/app/models` folder, which will be used for storing the trained models;
- `/app/out` folder, which will be used for storing the predictions and test results.

## Data folder structure

The data folder must contain the following files:

```bash
├── gcd
├── spec2008_agg
└── spec2008
```

Samples of these files are provided in the [releases](https://github.com/risingfbk/fluidos-energy-predictor/releases)
section of this repository.

The `gcd` file contains the workload data from the Google Cluster Data version 3
([repository](https://github.com/google/cluster-data),
[document](https://drive.google.com/file/d/10r6cnJ5cJ89fPWCgj7j4LtLBqYN9RiI9/view)), while the SPEC2008 contains power 
profiles from the SPEC2008 benchmark suite ([link](https://www.spec.org/power_ssj2008/results/power_ssj2008.html)).
The `spec2008_agg` file contains the aggregated power profiles from the SPEC2008 benchmark suite.

For your convenience, scripts for fetching `spec2008` data and generating both `gcd` and `spec2008` data are provided in
the `src/datasets/` folder.

Scripts for pulling data from the `gcd` dataset are not included in the scripts, as it requires a Google Cloud Platform
account, a project with billing enabled, and a lot of patience. Our data was retrieved with both `bq` and manual
downloads, both methods of which are described in the aforementioned link.

## Usage

Run `python3 src/main.py --help` for a list of available commands. The program either catches flags passed to it and if
not provided, automatically asks for the required information. 

```bash
options:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Model name (if unspecified, will be prompted)
  --curve CURVE, -c CURVE
                        Power curve file (if unspecified, will be chosen randomly)
  --epochs EPOCHS, -e EPOCHS
                        Number of epochs (if unspecified, will be prompted)
  --action ACTION, -a ACTION
                        Action to perform (train, search hyperparameters, test)
  --machine MACHINE, -M MACHINE
                        GCD machine files to use (if unspecified, will be chosen randomly)
```

The program requires at least a model name (`--model`), a power curve for the machine (`--power`), and a machine to use
for training (`--machine`). If unspecified, the model name will be prompted, while the power curve and the machine will
be chosen randomly from the available ones.

Then, depending on the action (`--action`), the program will either search for hyperparameters, train, or test a certain
model (eventually providing a number of epochs for training, `--epochs`).

If the program is set to train a model, it will automatically save the model and the logs in the `models` and `out` 
folders, respectively. If the program is set to test a model, it will automatically load the model from the `models`
(as a convenience, the contents of the `models` folder are printed when the program is run) and save the predictions and
the test results in the `out` folder.

Finally, the program will automatically generate a number of plots in the `out` folder, including the training and
validation loss and the predictions. Predictions are additionally saved in the `pred` folder as `.csv` and `.npy` files.

By the way the model is currently implemented, it is suggested to use a healthy amount of epochs (at least 1000) for
training. The program will automatically stop the training if the validation loss does not improve for a certain amount
of epochs (see below for more information).

### Specifying variables

The `src/parameters.py` file contains a number of constants that can be used to specify the parameters of the program.

#### Training parameters

```python
TEST_FILE_AMOUNT = 24
TRAIN_FILE_AMOUNT = 24
```

The program uses the `TRAIN_FILE_AMOUNT` and `TEST_FILE_AMOUNT` constants to specify the number of files to use for
training and testing, respectively. These files are pulled from the `gcd` and `spec2008` folders, and specifically,
from the subfolder the user specifies when running the program, or at random if the user does not specify one.

```python
SPLIT = 0.25
```

The `SPLIT` constant specifies the percentage of the training data to use for validation.

```python
PATIENCE = 150
LEARNING_RATE = 0.02
```

The `PATIENCE` constant specifies the number of epochs to wait before stopping the training if the validation loss does
not improve. The `LEARNING_RATE` constant specifies the learning rate to use for the training. Take note that the code
will adjust the learning rate automatically if the validation loss does not improve.

#### Model parameters

```python
N_FEATURES = 2                              # Number of features (CPU, memory)
STEPS_OUT = 1                               # Number of output steps from the model (a single value)
STEPS_IN = WEEK_IN_MINUTES // GRANULARITY   # Number of input steps to the model (see below)
```

These parameters should not be changed, as they are hardcoded in the model. They specify the number of features and the
number of output steps from the model. The `WEEK_IN_MINUTES` constant specifies the number of minutes in a week, and can
be found in the `support/dt.py` file along with similar constants.

```python
GRANULARITY = 15         # Granularity of the data in minutes
```

The `GRANULARITY` constant specifies the granularity of the data in minutes. This depends on how the data was generated.
With `GRANULARITY = 15`, thus, the model will have `10080 // 15 = 672` input steps for a week of data.

```python
FILTERS = 144
KSIZE = 3
```

The `FILTERS` and `KSIZE` constants specify the number of filters and the kernel size of the convolutional layers.

```python
OVERHEAD = 1
```

The `OVERHEAD` constant specifies by how much should the energy consumption be increased to account for the overhead
introduced by the machine.

#### Folder structure and other parameters

```python
LOG_FOLDER = "out"
DEFAULT_MODEL = "model1"
MODEL_FOLDER = "models"
GCD_FOLDER = "data/gcd"
SPEC_FOLDER = "data/spec2008_agg"
CACHE_FOLDER = "data/cache"
BANLIST_FILE = "banlist"
```

These constants specify the folder structure of the program. Make sure you reflect any changes in the folder structure
when changing these constants, especially if using Docker.

### Using the banlist

The banlist is a list of files that shall not be used for training. It is useful when downloading and generating large
batches of data: it might happen that some files are corrupt, badly formatted, or otherwise unusable. In this case, the
program will automatically skip the file and continue with the next one. However, if the file is not skipped, it may 
cause a crash (although this is unlikely). To prevent this, the banlist can be used to prevent the program from using
the file.

To use the banlist, create a file named `banlist` in the root folder of the project (or in the `/app` folder, if you are
using Docker). The file must contain a list of file names, one per line. The program will automatically skip the files
listed in the banlist. File paths must be specified from the root of the `data/gcd` folder. At the moment, skipping
power curves from the `spec2008` dataset is not supported.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as part of the FLUIDOS project, funded by the European Union's Horizon 2020 research and
innovation programme under grant agreement `101070473 - HORIZON-CL4-2021-DATA-01`.
It is an integral part of the Work Package 6 of the project, which aims to define an energy- and carbon-aware
computation model that can shift loads both in time and geography; devise cost-effective infrastructure optimisations
for industrial environments; use Artificial Intelligence and Machine Learning methods for performance prediction and
enhancement.