# PyTorch Projects

Learning PyTorch by following Patrick Loeber's full pytorch course, with theoretical foundations from Goodfellow et al. "Deep Learning" (2016).
---

## Projects

### 1. Feedforward Neural Network — Poker Hand Win Probability Estimator
Custom regression model trained on a self-generated dataset to predict heads-up win probability given two hole cards.
- Dataset: 10,000 samples generated via Monte Carlo simulation (1,000 random opponent/board rollouts per hand)
- Features: Card rank (2-14) and suit (0-3) encoding → 4 input features
- Architecture: Input (4) → Hidden Layer (20, ReLU) → Output (1)
- Loss: MSE | Optimizer: Adam
- Test loss: ~0.0009 (≈ ±3% win rate accuracy)
- Interactive CLI predictor with input validation
- Training monitored with TensorBoard

### 2. Feedforward Neural Network — Handwriting Recognition
**Source:** [Deep Learning With PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M) by Patrick Loeber

Feedforward neural network trained on the MNIST dataset to classify handwritten digits (0-9).
- Architecture: Input (784) → Hidden Layer (100, ReLU) → Output (10, Softmax)
- Loss: Cross-Entropy | Optimizer: Adam
- Accuracy: ~95% on test set
- Training monitored with TensorBoard (loss, batch accuracy, PR curves, prediction grids, probability distributions)

### 3. Convolutional Neural Network — CIFAR-10 Image Classification
**Source:** [Deep Learning With PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M) by Patrick Loeber

CNN trained on the CIFAR-10 dataset to classify 32×32 color images across 10 classes (plane, car, bird, cat, deer, dog, frog, horse, ship, truck).
- Architecture: Conv1 (3→6, 5×5) → MaxPool → Conv2 (6→16, 5×5) → MaxPool → FC(480→120) → FC(120→84) → FC(84→10)
- Loss: Cross-Entropy | Optimizer: SGD
- Evaluates overall accuracy and per-class accuracy on the test set

### 4. Linear Regression
**Source:** [Deep Learning With PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M) by Patrick Loeber

Single-feature linear regression on a synthetic dataset generated with `sklearn`.
- Loss: MSE | Optimizer: SGD
- Demonstrates: forward pass, backpropagation, gradient descent

### 5. Logistic Regression — Breast Cancer Classification
**Source:** [Deep Learning With PyTorch - Full Course](https://www.youtube.com/watch?v=c36lUUr864M) by Patrick Loeber

Binary classification on the `sklearn` breast cancer dataset.
- Loss: Binary Cross-Entropy | Optimizer: SGD
- Demonstrates: sigmoid output, binary classification pipeline

---

## Stack
Python · PyTorch · NumPy · scikit-learn · Matplotlib · TensorBoard · pandas