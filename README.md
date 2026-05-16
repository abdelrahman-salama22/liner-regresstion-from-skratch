# Linear Regression from Scratch

A clean implementation of **Linear Regression using Gradient Descent**, built entirely with NumPy — no scikit-learn or any ML framework.

---

## Table of Contents

- [Overview](#overview)
- [Math Behind the Model](#math-behind-the-model)
- [Gradient Descent Explained](#gradient-descent-explained)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Class API](#class-api)
- [Utility Functions](#utility-functions)
- [Example Output](#example-output)

---

## Overview

Linear Regression is the simplest supervised learning algorithm. Given a dataset of input-output pairs `(X, y)`, the goal is to find a straight line (or hyperplane in higher dimensions) that best fits the data.

The model learns two things:

- **Weights `w`** — how much each feature contributes to the prediction
- **Bias `b`** — the baseline offset (the y-intercept)

The prediction for a single sample is:

```
ŷ = w · x + b
```

---

## Math Behind the Model

### Loss Function — Mean Squared Error (MSE)

To measure how wrong our predictions are, we use MSE:

```
L(w, b) = (1/n) * Σ (ŷᵢ − yᵢ)²
```

- `n` is the number of training samples
- `ŷᵢ` is the predicted value
- `yᵢ` is the true value

The lower the MSE, the better the model fits the data. Our job is to find the `w` and `b` that minimize this loss.

### Gradients

To minimize the loss, we compute its partial derivatives with respect to `w` and `b`:

```
∂L/∂w = (1/n) * Xᵀ · (ŷ − y)

∂L/∂b = (1/n) * Σ (ŷ − y)
```

These gradients tell us the direction in which the loss increases. We move in the opposite direction to reduce it.

---

## Gradient Descent Explained

Gradient descent is an iterative optimization algorithm. At each step, we nudge `w` and `b` slightly in the direction that reduces the loss:

```
w ← w − α · ∂L/∂w
b ← b − α · ∂L/∂b
```

Where `α` (alpha) is the **learning rate** — a small positive number that controls the step size.

### Why the learning rate matters

| Learning Rate | Effect |
|---|---|
| Too large | Overshoots the minimum; loss oscillates or diverges |
| Too small | Converges very slowly; takes many iterations |
| Just right | Smoothly converges to the minimum |

### Convergence

The algorithm repeats for a fixed number of iterations. Over time, the loss curve should steadily decrease and flatten out — that's when the model has converged.

```
Iteration  100 | Loss: 4.823100
Iteration  200 | Loss: 2.371042
Iteration  300 | Loss: 2.309871
Iteration  400 | Loss: 2.309567
Iteration  500 | Loss: 2.309566   ← converged
```

---

## Project Structure

```
linear_regression_scratch.py
│
├── class LinearRegression
│   ├── __init__()      — set learning rate & iterations
│   ├── fit(X, y)       — run gradient descent
│   ├── predict(X)      — compute ŷ = X·w + b
│   ├── score(X, y)     — compute R² metric
│   └── loss_history    — MSE recorded at each iteration
│
├── train_test_split()  — shuffle & split data into train/test
├── standardize()       — zero-mean, unit-variance feature scaling
│
└── __main__            — demo: generate data, train, evaluate, plot
```

---

## How to Run

**Requirements:** Python 3.8+ with NumPy and Matplotlib.

```bash
pip install numpy matplotlib
python linear_regression_scratch.py
```

This will:
1. Generate a synthetic dataset (`y = 3x + 7 + noise`)
2. Split it into train/test sets (80/20)
3. Standardize the features
4. Train the model using gradient descent
5. Print the R² score and learned parameters
6. Save a plot of the regression fit and loss curve

---

## Class API

### `LinearRegression(learning_rate, n_iterations)`

```python
model = LinearRegression(learning_rate=0.1, n_iterations=500)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | float | 0.01 | Step size for gradient descent |
| `n_iterations` | int | 1000 | Number of training iterations |

### `.fit(X, y)`

Trains the model. `X` must be a 2D NumPy array of shape `(n_samples, n_features)` and `y` a 1D array of shape `(n_samples,)`.

```python
model.fit(X_train, y_train)
```

### `.predict(X)`

Returns predicted values `ŷ` for the given input.

```python
y_pred = model.predict(X_test)
```

### `.score(X, y)`

Returns the **R² score** — a value between 0 and 1. A score of 1.0 means a perfect fit.

```python
r2 = model.score(X_test, y_test)
```

### `.loss_history`

A list of MSE values recorded at every iteration, useful for plotting the training curve.

```python
plt.plot(model.loss_history)
```

---

## Utility Functions

### `train_test_split(X, y, test_size=0.2, seed=42)`

Randomly shuffles and splits the dataset. Returns `X_train, X_test, y_train, y_test`.

### `standardize(X_train, X_test)`

Standardizes features to have zero mean and unit variance using training set statistics:

```
X_scaled = (X − μ) / σ
```

This is essential — gradient descent converges much faster on scaled features.

---

## Example Output

```
Iteration  100 | Loss: 4.823100
Iteration  200 | Loss: 2.371042
Iteration  300 | Loss: 2.309871
Iteration  400 | Loss: 2.309567
Iteration  500 | Loss: 2.309566

Train R²: 0.9612
Test  R²: 0.9588
Learned weight: 2.9814 | bias: 6.9973
```

The learned weight (~3) and bias (~7) closely match the true values used to generate the data (`y = 3x + 7`), confirming the model works correctly.

---

## Key Takeaways

- Linear regression minimizes MSE using calculus — the gradients tell us exactly how to adjust each parameter.
- Gradient descent is a general-purpose optimizer, not just for linear regression. The same loop pattern appears in neural networks and deep learning.
- Feature standardization is not optional in practice — without it, gradient descent can fail to converge or converge very slowly.
- R² is a normalized metric: 0 means the model is no better than predicting the mean, and 1 means a perfect fit.
