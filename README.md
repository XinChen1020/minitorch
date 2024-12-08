# MiniTorch Project

This repository contains my **pure Python re-implementation of PyTorch API**, completed as part of the [MiniTorch](https://minitorch.github.io/) project guide. The goal of this project was to deepen my understanding of deep learning frameworks by building key components like tensors, automatic differentiation, and training workflows.

## Overview

This project is an educational exercise in understanding:
- How tensors and operations are implemented from scratch.
- The mechanisms of automatic differentiation and backpropagation.
- The construction and training of neural networks.
- Optimization techniques and GPU acceleration.

It was implemented entirely in Python, with enhancements like CUDA integration for GPU computation and Numba for just-in-time compilation to improve performance.

## Features

- **Basic Tensor Operations**: Implemented core operations like addition, matrix multiplication, and reshaping.
- **Automatic Differentiation**: Built a simple autograd engine to compute gradients for optimization.
- **Model Definition and Training**: Constructed custom modules and trained them using SGD.
- **Optimizations**: Added CUDA support with optimizations like sequential addressing to reduce memory access conflicts.

## Installation

To set up the project, clone this repository and install the required dependencies:

```bash
git clone https://github.com/XinChen1020/minitorch.git
cd minitorch
pip install -r requirements.txt
pip install -r requirements.extra.txt

```

Ensure you have the appropriate CUDA toolkit installed if you plan to run the project on a GPU.


## Example: MNIST Classification

An example demonstrating how to define and train a model using MiniTorch is available in the [project/run_mnist_multiclass.py](project/run_mnist_multiclass.py). It includes:
- Custom module and layer definitions.
- A simple training loop.
- Logsoftmax-based output for classification.

To view and run the example:

```bash
python project/run_mnist_multiclass.py
```
Make sure the [MNIST](https://yann.lecun.com/exdb/mnist/) dataset is downloaded under project/data/ folder

---

### TODOs

- [ ] Implement CUDA Optimization for CNN: Extend the project by integrating CUDA kernels for convolutional layers to improve performance when running CNN-based models.

---

## Results

All experimental results are stored in the `results` folder for analysis and future reference.

---

