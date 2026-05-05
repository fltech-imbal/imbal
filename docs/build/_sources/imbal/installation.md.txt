# `imbal` Installation Guide

## Requirements

As an extension of many of the features available in TensorFlow, `imbal` is subject to
the sample hardware, system, and software requirements as TensorFlow.

A full overview of how to ensure these requirements are met can be found on [this TensorFlow
documentation page](https://www.tensorflow.org/install/pip).

All other `imbal` dependencies should be handled automatically during installation.

## Download and Installation

The `imbal` package can be downloaded from [this GitHub repository](https://github.com/fltech-imbal/imbal).
If you have Git installed, you can clone the repository using the following command.

```shell
git clone https://github.com/fltech-imbal/imbal
```

Alternatively, you can download the `.zip` version of the repository from GitHub, and extract the files
on your local machine.

Once downloaded locally, install `imbal` by running the following command within the root directory
of your local copy of the repository:

```shell
pip install .
```

If you are using a Python virtual environment to install `imbal`, make sure you activate the virtual environment
before running the command above.

## Updating `imbal`

To update imbal, download the latest version, activate your Python virtual environment (if you choose to 
use one), and run the following command within the root directory of your local copy of the updated repository:

```shell
pip install --upgrade .
```