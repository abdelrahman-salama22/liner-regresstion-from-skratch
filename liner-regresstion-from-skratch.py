import numpy as np
import matplotlib.pyplot as plt
import os


class LinearRegression:
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        
        n_samples, n_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            # Forward pass: compute predictions
            y_pred = self._predict(X)

            # Compute MSE loss
            loss = self._mse(y, y_pred)
            self.loss_history.append(loss)

            # Compute gradients
            error = y_pred - y                          # (n_samples,)
            dw = (1 / n_samples) * X.T @ error         # (n_features,)
            db = (1 / n_samples) * np.sum(error)       # scalar

            # Gradient descent update
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1:>5} | Loss: {loss:.6f}")

        return self

    def predict(self, X):
        """Return predictions for X."""
        return self._predict(X)

    def _predict(self, X):
        return X @ self.weights + self.bias

    @staticmethod
    def _mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def score(self, X, y):
        """R² score (coefficient of determination)."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def __repr__(self):
        return (
            f"LinearRegression(lr={self.lr}, "
            f"n_iterations={self.n_iterations})"
        )


# ─── Helper: train/test split ────────────────────────────────────────────────

def train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ─── Helper: feature scaling (standardization) ───────────────────────────────

def standardize(X_train, X_test):
    mu  = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8      # avoid division by zero
    return (X_train - mu) / std, (X_test - mu) / std


# ─── Demo ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Synthetic dataset: y = 3x + 7 + noise
    n = 200
    X_raw = rng.uniform(-3, 3, size=(n, 1))
    y     = 3 * X_raw[:, 0] + 7 + rng.normal(0, 1.5, size=n)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y)

    # Scale
    X_train_s, X_test_s = standardize(X_train, X_test)

    # Train
    model = LinearRegression(learning_rate=0.1, n_iterations=500)
    model.fit(X_train_s, y_train)

    # Evaluate
    train_r2 = model.score(X_train_s, y_train)
    test_r2  = model.score(X_test_s,  y_test)
    print(f"\nTrain R²: {train_r2:.4f}")
    print(f"Test  R²: {test_r2:.4f}")
    print(f"Learned weight: {model.weights[0]:.4f} | bias: {model.bias:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Regression fit
    ax = axes[0]
    ax.scatter(X_train[:, 0], y_train, alpha=0.5, label="Train", color="#378ADD")
    ax.scatter(X_test[:, 0],  y_test,  alpha=0.7, label="Test",  color="#E24B4A", zorder=3)
    x_line = np.linspace(X_raw.min(), X_raw.max(), 200).reshape(-1, 1)
    x_line_s, _ = standardize(x_line, x_line)          # scale for prediction
    y_line = model.predict(x_line_s)
    ax.plot(x_line[:, 0], y_line, color="#533AB7", linewidth=2, label="Fit")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Linear Regression Fit")
    ax.legend()

    # 2. Loss curve
    ax2 = axes[1]
    ax2.plot(model.loss_history, color="#1D9E75", linewidth=2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Training Loss")

    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "regression_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to: {save_path}")