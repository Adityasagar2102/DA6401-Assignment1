# DA6401 Assignment 1

## Neural Network from Scratch using NumPy

**Author:** Aditya Kumar Sagar  
**Roll Number:** CS25M006  
**Course:** DA6401 – Deep Learning  
**Institute:** IIT Madras

---

## GitHub Repository

- **you can find the complete project on**: [GitHub](https://github.com/Adityasagar2102/DA6401-Assignment1)

## Weights & Biases Report

- **Public Report Link**: [W&B Report](INSERT_YOUR_URL_HERE)

## Project Overview

This project implements a **fully connected neural network (Multi-Layer Perceptron)** from scratch using **NumPy** without using deep learning frameworks such as PyTorch or TensorFlow.

The objective of this assignment is to understand the internal mechanics of neural networks, including:

- Forward propagation
- Backpropagation
- Gradient computation
- Optimization algorithms
- Hyperparameter tuning

The model is trained and evaluated on the following datasets:

- **MNIST** – Handwritten digit classification
- **Fashion-MNIST** – Clothing classification

All experiments are tracked using **Weights & Biases (W&B)**.

---

## Features Implemented

### Neural Network Architecture

- Fully connected dense layers
- Support for multiple hidden layers
- Configurable layer sizes

### Activation Functions

- ReLU
- Sigmoid
- Tanh

### Loss Functions

- Cross Entropy Loss
- Mean Squared Error (MSE)

### Optimizers

- SGD
- Momentum
- Nesterov Accelerated Gradient (NAG)
- RMSProp

### Additional Features

- Xavier and random weight initialization
- L2 regularization (weight decay)
- Gradient norm logging
- Hyperparameter sweeps with W&B
- Model saving and inference

---

## Project Structure

da6401_assignment_1
│
├── src
│ ├── train.py # Training script
│ ├── inference.py # Model evaluation
│ ├── gradient_check.py # Gradient verification
│
│ ├── ann
│ │ ├── neural_network.py # Neural network implementation
│ │ ├── neural_layer.py # Dense layer implementation
│ │ ├── activations.py # Activation functions
│ │ ├── objective_functions.py # Loss functions
│ │ ├── optimizers.py # Optimization algorithms
│
│ ├── utils
│ │ └── data_loader.py # Dataset loader
│
├── models
│ ├── best_model.npy # Saved model weights
│ └── best_config.json # Configuration of best model
│
├── README.md
└── requirements.txt

---

## Installation

Clone the repository:

```bash
git clone <your_repo_link>
cd da6401_assignment_1
```

Create and activate virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run the following command from the src folder:

```bash
python train.py -wp da6401_assignment_1 -d mnist -e 10 -b 64 -l cross_entropy -o rmsprop -lr 0.001 -nhl 3 -sz "128 128 128" -a relu -w_i xavier
```

Arguments
Argument Description
-wp W&B project name
-d Dataset (mnist or fashion_mnist)
-e Number of epochs
-b Batch size
-l Loss function
-o Optimizer
-lr Learning rate
-nhl Number of hidden layers
-sz Hidden layer sizes
-a Activation function
-w_i Weight initialization

During training, the best model (based on F1 score) is automatically saved.

Saved files:

```bash
models/best_model.npy
models/best_config.json
```

## Running Inference

To evaluate the saved model on the test set:

```bash
python inference.py -mp ../models/best_model.npy -d mnist
```

Output metrics include:

- Loss
- Accuracy
- Precision
- Recall
- F1 Score

## Gradient Checking

To verify the correctness of backpropagation:

```bash
python gradient_check.py
```

Expected output:

```bash
Gradient check passed!
```

## Weights & Biases Report

All experiments, sweeps, and visualizations are logged using Weights & Biases.

- The report includes:
- Hyperparameter sweeps
- Optimizer comparison
- Gradient norm analysis
- Confusion matrix
- Model failure visualization

Best Model Configuration

The best-performing configuration on MNIST was:

Parameter Value
Architecture 3 hidden layers
Hidden Units 128 each
Activation ReLU
Optimizer RMSProp
Learning Rate 0.001
Batch Size 64

Performance:

Train Accuracy ≈ 98%
Test Accuracy ≈ 97%
