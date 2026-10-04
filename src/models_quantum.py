"""
Hybrid quantum-classical model and classical ablation replacement.

Architecture:
- AngleEmbedding (RY) on n_qubits
- StronglyEntanglingLayers with configurable depth
- PauliZ expectation values on all qubits
- Linear(n_qubits, 1) + Sigmoid

Uses PennyLane's qml.qnn.TorchLayer for seamless PyTorch integration.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def create_quantum_circuit(
    n_qubits: int = 8,
    n_layers: int = 2,
    diff_method: str = "backprop",
    device_name: str = "default.qubit",
    noise_level: Optional[float] = None,
) -> Tuple[qml.QNode, Dict[str, Tuple]]:
    """Create the quantum circuit as a QNode.

    Args:
        n_qubits: Number of qubits (= number of input features).
        n_layers: Number of StronglyEntanglingLayers.
        diff_method: Differentiation method.
        device_name: PennyLane device name.
        noise_level: If set, use depolarizing noise after each layer.

    Returns:
        (qnode, weight_shapes) for TorchLayer construction.
    """
    if noise_level is not None:
        dev = qml.device("default.mixed", wires=n_qubits)
    else:
        dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        """Quantum circuit with angle embedding and strongly entangling layers."""
        # Angle embedding: encode each feature as RY rotation
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        # Variational layers
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # Add noise if specified
        if noise_level is not None:
            for w in range(n_qubits):
                qml.DepolarizingChannel(noise_level, wires=w)

        # Measure PauliZ on all qubits
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    # Weight shapes for StronglyEntanglingLayers
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}

    return circuit, weight_shapes


class HybridQuantumModel(nn.Module):
    """Hybrid quantum-classical model for binary classification.

    Combines a parameterized quantum circuit with a classical output layer.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        diff_method: str = "backprop",
        device_name: str = "default.qubit",
        noise_level: Optional[float] = None,
    ):
        """Initialize the hybrid model.

        Args:
            n_qubits: Number of qubits.
            n_layers: StronglyEntanglingLayers depth.
            diff_method: PennyLane differentiation method.
            device_name: PennyLane device name.
            noise_level: Optional depolarizing noise probability.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_level = noise_level

        circuit, weight_shapes = create_quantum_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            diff_method=diff_method,
            device_name=device_name,
            noise_level=noise_level,
        )

        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.classical_output = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantum circuit and classical layer.

        Args:
            x: Input tensor of shape (batch, n_qubits) with values in [0, pi].

        Returns:
            Predictions of shape (batch, 1) with values in [0, 1].
        """
        q_out = self.quantum_layer(x)
        if isinstance(q_out, tuple):
            q_out = torch.stack(q_out, dim=-1)
        out = self.classical_output(q_out)
        return self.sigmoid(out)

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters.

        Returns:
            Dict with quantum, classical, and total parameter counts.
        """
        quantum_params = sum(
            p.numel() for name, p in self.named_parameters() if "quantum" in name
        )
        classical_params = sum(
            p.numel() for name, p in self.named_parameters() if "classical" in name
        )
        total = quantum_params + classical_params
        return {
            "quantum": quantum_params,
            "classical": classical_params,
            "total": total,
        }


class ClassicalAblationModel(nn.Module):
    """Classical model matching or bounding the hybrid model's parameter count.

    Replaces the quantum circuit with a classical neural network layer.
    Using total target parameters (57):
        hidden = (57 - 1) // (8 + 2) = 5 units -> 51 parameters (lower bound)
        hidden = 6 units -> 61 parameters (upper bound)
    The 57-parameter hybrid model is thus sandwiched between the 51- and 61-parameter ablations.
    """

    def __init__(
        self,
        n_inputs: int = 8,
        hidden_units: Optional[int] = None,
        total_target_params: int = 57,
        n_quantum_params: Optional[int] = None,
    ):
        """Initialize the ablation model.

        Args:
            n_inputs: Number of input features (= n_qubits in hybrid).
            hidden_units: Explicit number of hidden units (5 for 51 params, 6 for 61 params).
            total_target_params: Total parameters of the hybrid model (default 57).
            n_quantum_params: Kept for backwards compatibility.
        """
        super().__init__()
        self.n_inputs = n_inputs

        if hidden_units is not None:
            self.hidden = hidden_units
        else:
            # Formula using total parameters (57):
            # Network: n_inputs * hidden + hidden (bias)
            # Output: hidden * 1 + 1 (bias)
            # Total = hidden * (n_inputs + 2) + 1 = 10 * hidden + 1
            # Solve for hidden: (total_target_params - 1) // (n_inputs + 2)
            self.hidden = max(1, (total_target_params - 1) // (n_inputs + 2))

        self.network = nn.Sequential(
            nn.Linear(n_inputs, self.hidden),
            nn.Tanh(),  # Similar range to quantum expectation values [-1, 1]
        )
        self.classical_output = nn.Linear(self.hidden, 1)
        self.sigmoid = nn.Sigmoid()

        total_p = self.count_parameters()["total"]
        logger.info(
            f"ClassicalAblationModel: hidden={self.hidden}, "
            f"total_params={total_p} (target={total_target_params})"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, n_inputs).

        Returns:
            Predictions of shape (batch, 1) with values in [0, 1].
        """
        h = self.network(x)
        out = self.classical_output(h)
        return self.sigmoid(out)

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters.

        Returns:
            Dict with network, hidden units, and total parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        return {"hidden_units": self.hidden, "classical_replacement": total, "total": total}


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced data.

    Args:
        labels: Array of 0s and 1s.

    Returns:
        Tensor with positive class weight for BCEWithLogitsLoss.
    """
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    logger.info(f"Class weight (pos_weight): {pos_weight.item():.4f}")
    return pos_weight


def train_torch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 0.02,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 7,
    pos_weight: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> Tuple[nn.Module, Dict]:
    """Train a PyTorch model with early stopping.

    Args:
        model: PyTorch model.
        X_train: Training features (numpy).
        y_train: Training labels (numpy).
        X_val: Validation features (numpy).
        y_val: Validation labels (numpy).
        lr: Learning rate.
        batch_size: Batch size.
        max_epochs: Maximum epochs.
        patience: Early stopping patience.
        pos_weight: Class weight for BCE loss.
        seed: Random seed.

    Returns:
        (trained_model, history_dict)
    """
    torch.manual_seed(seed)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    # DataLoader
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCELoss()

    # If using BCEWithLogitsLoss, we need to remove the sigmoid from forward
    # Actually let's use BCELoss since our model outputs sigmoid
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    start_time = time.time()

    for epoch in range(max_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            # Apply class weighting manually to BCELoss
            if pos_weight is not None:
                weights = torch.where(
                    batch_y == 1, pos_weight.item(), 1.0
                )
                loss = nn.functional.binary_cross_entropy(preds, batch_y, weight=weights)
            else:
                loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            if pos_weight is not None:
                weights = torch.where(y_val_t == 1, pos_weight.item(), 1.0)
                val_loss = nn.functional.binary_cross_entropy(
                    val_preds, y_val_t, weight=weights
                ).item()
            else:
                val_loss = criterion(val_preds, y_val_t).item()
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}"
            )

    train_time = time.time() - start_time
    history["train_time"] = train_time

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(
        f"Training complete in {train_time:.1f}s. "
        f"Best val loss: {best_val_loss:.4f}"
    )

    return model, history
