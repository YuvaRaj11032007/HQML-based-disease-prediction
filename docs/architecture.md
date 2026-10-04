# Hybrid Quantum-Classical ML Architecture

## System Architecture Overview

```
                      +-----------------------------------+
                      |   TCGA-BRCA Gene Expression Data  |
                      |   (FPKM-UQ, log2(x+1) normalized) |
                      +-----------------+-----------------+
                                        |
                                        v
                      +-----------------------------------+
                      |   Patient-Level Stratified Split  |
                      |   Train/CV Pool (80%) | Test (20%)|
                      +-----------------+-----------------+
                                        |
               +------------------------+------------------------+
               | (Inside CV Fold / Training Portion Only)         |
               v                                                 v
  +-------------------------+                       +-------------------------+
  | Differential Expression |                       | Differential Expression |
  |   (Welch's t-test top   |                       |   (Welch's t-test top   |
  |       200 genes)        |                       |       200 genes)        |
  +------------+------------+                       +------------+------------+
               |                                                 |
               v                                                 v
  +-------------------------+                       +-------------------------+
  | Mutual Information Rank |                       | Mutual Information Rank |
  |     (Top 8 genes)       |                       |     (Top 50 genes)      |
  +------------+------------+                       +------------+------------+
               |                                                 |
               v                                                 v
  +-------------------------+                       +-------------------------+
  | MinMaxScaler -> [0, pi] |                       | MinMaxScaler -> [0, pi] |
  |   (Fit on train only)   |                       |   (Fit on train only)   |
  +------------+------------+                       +------------+------------+
               |                                                 |
       +-------+-----------------------+                         |
       |                               |                         |
       v                               v                         v
+--------------+               +---------------+         +---------------+
| Quantum Circuit              | Classical     |         | Classical     |
| - 8 Qubits (RY angles)       | Ablation      |         | Baselines     |
| - StronglyEntanglingLayers   | (Equal param  |         | (SVM-RBF, RF, |
| - PauliZ Expectations        | count ~41-57) |         |  XGBoost)     |
| - Linear(8,1) + Sigmoid      |               |         |               |
+-------+------+               +-------+-------+         +-------+-------+
        |                              |                         |
        +------------------------------+-------------------------+
                                       |
                                       v
                      +-----------------------------------+
                      |     Statistical Evaluation        |
                      | - Threshold tuned on Val (Youden) |
                      | - Repeated Stratified 5-Fold CV   |
                      | - Paired Wilcoxon Signed-Rank Test|
                      | - Held-Out Test Set Evaluation    |
                      | - Permutation Importance & SHAP   |
                      | - Learning Curves & Noise Models  |
                      +-----------------------------------+
```

## Quantum Circuit Specification

1. **State Preparation / Embedding**:
   - $\text{AngleEmbedding}(x, \text{wires}=[0..7], \text{rotation}=\text{'Y'})$
   - Maps normalized features $x_i \in [0, \pi]$ directly to individual qubit rotations:
     $$|\psi(x)\rangle = \bigotimes_{i=0}^7 R_Y(x_i)|0\rangle$$

2. **Variational Entangling Layers**:
   - $\text{StronglyEntanglingLayers}(\theta, \text{wires}=[0..7])$ with depth $L=2$
   - Parameter tensor shape: $(L, N, 3) = (2, 8, 3) = 48$ trainable weights.
   - Each layer applies arbitrary single-qubit rotations $R(\alpha, \beta, \gamma) = R_Z(\gamma)R_Y(\beta)R_Z(\alpha)$ followed by CNOT entanglement in a circular topology with varying range.

3. **Measurement**:
   - Pauli-Z expectation values across all qubits:
     $$\langle Z_i \rangle = \langle \psi(x, \theta) | Z_i | \psi(x, \theta) \rangle \in [-1, 1], \quad i \in \{0, \dots, 7\}$$

4. **Classical Output Layer**:
   - $\text{Linear}(8, 1)$: 8 weights + 1 bias = 9 parameters.
   - Total hybrid parameters: $48 + 9 = 57$.
   - Activation: $\text{Sigmoid}(z) \in [0, 1]$.
