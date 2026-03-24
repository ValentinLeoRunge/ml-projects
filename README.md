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
- Training monitored with TensorBoard (loss, batch accuracy, PR curves, prediction grids, probability distributions)

### 2. Convolutional Neural Network — CIFAR-10 Image Classification
CNN trained on the CIFAR-10 dataset to classify 32×32 color images across 10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck).
- Architecture: Conv1 (3→6, 5×5) → MaxPool → Conv2 (6→16, 5×5) → MaxPool → FC(480→120) → FC(120→84) → FC(84→10)
- Loss: Cross-Entropy | Optimizer: SGD
- Evaluates overall accuracy and per-class accuracy on the test set

### 3. Linear Regression
Single-feature linear regression on a synthetic dataset generated with `sklearn`.
- Loss: MSE | Optimizer: SGD
- Demonstrates: forward pass, backpropagation, gradient descent

### 4. Logistic Regression — Breast Cancer Classification
Binary classification on the `sklearn` breast cancer dataset.
- Loss: Binary Cross-Entropy | Optimizer: SGD
- Demonstrates: sigmoid output, binary classification pipeline

---

## Stack
Python · PyTorch · NumPy · scikit-learn · Matplotlib · TensorBoard