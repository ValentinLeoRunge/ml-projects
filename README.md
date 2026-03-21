# PyTorch Projects

Learning PyTorch by following Patrick Loeber's full pytorch course, with theoretical foundations from Goodfellow et al. "Deep Learning" (2016).

**Source:** [Deep Learning With PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M) by Patrick Loeber

---

## Projects

### 1. Feedforward Neural Network — Handwriting Recognition
Feedforward neural network trained on the MNIST dataset to classify handwritten digits (0-9).
- Architecture: Input (784) → Hidden Layer (100, ReLU) → Output (10, Softmax)
- Loss: Cross-Entropy | Optimizer: Adam
- Accuracy: ~95% on test set

### 2. Linear Regression
Single-feature linear regression on a synthetic dataset generated with `sklearn`.
- Loss: MSE | Optimizer: SGD
- Demonstrates: forward pass, backpropagation, gradient descent

### 3. Logistic Regression — Breast Cancer Classification
Binary classification on the `sklearn` breast cancer dataset.
- Loss: Binary Cross-Entropy | Optimizer: SGD
- Demonstrates: sigmoid output, binary classification pipeline

---

## Stack
Python · PyTorch · NumPy · scikit-learn · Matplotlib