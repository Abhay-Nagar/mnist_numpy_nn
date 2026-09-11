# Neural Network From Scratch

A fully connected neural network implemented from scratch using **Python and NumPy**, without PyTorch, TensorFlow, automatic differentiation, or other machine-learning frameworks.

I built this project to develop a deeper understanding of the mathematics behind neural networks by manually implementing forward propagation, backpropagation, gradient descent, and the training process.

## Results

| Dataset | Test Accuracy |
|---------|--------------:|
| MNIST | 94.7% |
| Fashion-MNIST | 86.49% |

## Testing the Pretrained Models

This repository includes two pretrained neural networks:

| Model | Dataset | Test Accuracy |
|-------|---------|--------------:|
| `model.npz` | MNIST | 94.7% |
| `model_fashion.npz` | Fashion-MNIST | 86.49% |

To verify the results without retraining the networks, simply run:

```bash
python test_model.py
```

The script automatically loads both pretrained models and evaluates each one
against the test data for its corresponding dataset.

It will report the test accuracy for both the MNIST and Fashion-MNIST models.

The pretrained `.npz` files contain the weights and biases learned during
training. New models can also be trained from scratch using `train_model.py`
and `train_model_fashion.py`.

## How It Works

The neural network uses NumPy for matrix operations while implementing the underlying neural-network algorithms manually.

During forward propagation, each layer performs a linear transformation followed by a sigmoid activation function. The resulting activations are then passed to the next layer.

During training, gradients for each weight and bias are calculated manually using backpropagation and the chain rule. These gradients are then used to update the network parameters using gradient descent.

The implementation includes:

- Fully connected neural network layers
- Sigmoid activation functions
- Forward propagation
- Backpropagation
- Manual gradient calculation
- Mini-batch training
- Gradient descent
- Model evaluation

No automatic differentiation or machine-learning frameworks are used.

## Project Structure

```text
.
├── main.py
├── train_model.py
├── train_model_fashion.py
└── README.md
```

### `main.py`

Contains the neural network implementation and the essential functions used for training and evaluating the network.

### `train_model.py`

Trains a new neural network on the **MNIST handwritten digit dataset** and evaluates its accuracy on the test dataset.

Run with:

```bash
python train_model.py
```

### `train_model_fashion.py`

Trains a new neural network on the **Fashion-MNIST dataset** and evaluates its accuracy on the test dataset.

Run with:

```bash
python train_model_fashion.py
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install the required dependencies:

```bash
pip install numpy
```

Add any additional dependencies here if required.

## Usage

To train and test a model on MNIST:

```bash
python train_model.py
```

To train and test a model on Fashion-MNIST:

```bash
python train_model_fashion.py
```

Training parameters such as the number of epochs, learning rate, batch size, and network architecture can be modified in the training scripts.

## Network Architecture

The network is a multilayer fully connected neural network.

For MNIST and Fashion-MNIST, each 28 × 28 image is flattened into a vector of **784 input values**.

```text
28 × 28 Image
      │
      ▼
784 Input Neurons
      │
      ▼
 Hidden Layer(s)
      │
      ▼
10 Output Neurons
      │
      ▼
Predicted Class
```

The 10 output neurons correspond to the 10 possible classes in each dataset.

Input pixel values are normalized before being passed through the network.

## Training

Training is performed using mini-batches.

For each batch, the network:

1. Performs forward propagation.
2. Calculates the prediction error.
3. Propagates the error backward through each layer.
4. Calculates gradients for the weights and biases using the chain rule.
5. Updates the network parameters using gradient descent.

This entire process is implemented manually using NumPy operations.

## Why I Built This

Modern machine-learning frameworks make it possible to create neural networks with only a few lines of code. While useful in practice, these abstractions can hide much of what is actually happening during training.

I built this project to understand those underlying operations directly.

Implementing the network from scratch helped me develop a better understanding of:

- How information propagates through a neural network
- How loss propagates backward through the network
- How the chain rule is used during backpropagation
- How gradients are calculated for weights and biases
- How matrix operations allow many neurons and training examples to be processed efficiently
- How learning rate, batch size, initialization, and training duration affect convergence
- How neural networks can be debugged when training does not behave as expected

## Datasets

### MNIST

MNIST contains 28 × 28 grayscale images of handwritten digits belonging to 10 classes (0–9).

**Achieved test accuracy: 94.7%**

### Fashion-MNIST

Fashion-MNIST uses the same 28 × 28 image format as MNIST but contains images of clothing belonging to 10 different classes.

**Achieved test accuracy: 86.49%**

Using Fashion-MNIST provided an additional test of whether the implementation could learn a more difficult image-classification problem without changing the fundamental neural-network implementation.

## Future Improvements

Possible future improvements include:

- Additional activation functions such as ReLU
- Softmax output
- Cross-entropy loss
- Improved weight initialization
- Additional optimization algorithms
- Configurable network architectures
- Saving and loading trained models
- Training-loss and accuracy visualization
- Additional datasets

## Technologies

- Python
- NumPy
- MNIST
- Fashion-MNIST

## License

This project is available under the MIT License.
