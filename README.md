# Transformers for Time-series

## Introduction

This section introduces a variety of time series models that are designed to handle and analyze sequential data effectively.

### >>>Fully Integrated Functions<<<

The ease of use is prioritized in this framework. Simply select your data and the model of your choice. The framework takes care of the rest, including training, validation, testing, logging, metrics, visualizations, and more for a wide array of time series tasks.

### >>>Retaining Flexibility<<<

The framework is designed to accommodate user-specific requirements. You can effortlessly incorporate your datasets using the provided utilities. Additionally, introducing new models for different time series tasks is as straightforward as passing a `backbone: nn.Module` and a `head: nn.Module`, while preserving all the aforementioned functionalities.

**Play and Enjoy!**

## Architecture

The project structure is organized as follows:

```
|
|- Repo
    |- data
    |- logs
    |
    |- Modules
    |   |- components
    |   |   |- activations.py
    |   |   |- other components
    |
    |   |- framework
    |   |   |- backbones.py
    |   |   |- heads.py
    |
    |   |- classification.py
    |   |- other tasks
    |
    |- main.py
```

## Tutorial

### Step 1: Install Required Packages

Run the following commands to clone the repository and install the necessary packages:

```shell
git clone https://jihulab.com/ml-sunluohao/bearing-fault-classification.git
cd bearing-fault-classification
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

NOTE: Python version >= 3.12 is required. Check your Python version with:

```shell
python --version
# Ensure Python 3.12.0 or higher. If not, install it using the following command:
# pip install python=3.12
# or
# conda install python=3.12
```

### Step 2: Conduct Your First Experiment

Refer to the [example.ipynb](example.ipynb) for a guided walkthrough.

## Customizing Your Own Datasets and Models

### Custom Datasets

...

### Custom Models

...