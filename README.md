# Transformers for Time-series


## Overview

Welcome to the Transformers for Time-series, a comprehensive framework tailored for the effective management and analysis of sequential data. This repository contains a suite of models that leverage the power of transformers to handle time series data with precision and efficiency.

## Key Features

### User-Friendly Interface

We've designed this framework with simplicity in mind. Choose your dataset and the model that best fits your needs, and let the framework handle the complexities of training, validation, testing, and more. It offers a seamless experience, covering logging, metrics computation, and visualizations for diverse time series tasks.

### Flexibility at Its Best

The framework is built to be adaptable to your unique requirements. Integrate your custom datasets with ease using the provided utilities. Furthermore, adding new models for specialized time series tasks is as simple as defining a `backbone: nn.Module` and a `head: nn.Module`, without compromising any of the framework's built-in functionalities.

### Based on Lightning

**Get ready to explore and have fun!**

## Project Structure

Here's a quick overview of the project layout:

```
/
└── Repo/
    ├── data/                  # Directory for datasets
    ├── logs/                 # Directory for storing logs
    │
    ├── Modules/              # Core modules of the framework
    │   ├── components/       # Various components
    │   │   ├── activations.py # Activation functions
    │   |   ├── framework.py    # Framework specific modules
    │   │   └── ...            # More components
    │   │
    │   └── classification.py # Time series classification models
    │   └── ...                # Other tasks
    │
    └── main.py               # Main entry point of the application
```

## Getting Started

### Step 1: Set Up Your Environment

To get started, clone the repository and install the required packages using the following commands:

```shell
git clone https://github.com/LuohaoSun/Transformers-for-Time-Series.git
cd Transformers-for-Time-Series
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Make sure you have Python version **>= 3.12**. You can check your Python version by running:

```shell
python --version
```

If your Python version is lower than 3.12.0, update it with the following command:

```shell
# pip install python=3.12
# or
# conda install python=3.12
```

### Step 2: Dive Into Your First Experiment

To familiarize yourself with the framework, follow the steps outlined in the [example.ipynb](example.ipynb) Jupyter notebook.

## Customization

### Tailor Your Datasets

...

### Create Custom Models

...

## Additional Resources

- Found an issue or have a suggestion? Feel free to open an issue.

## License

This project is licensed under the MIT License.