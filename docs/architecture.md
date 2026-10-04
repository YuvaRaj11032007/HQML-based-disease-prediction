# Hybrid Quantum-Classical ML Architecture

## System Architecture Overview

```
+----------------------------------------------------------------------------------------+
|                            TCGA-BRCA Gene Expression Data                              |
|                 UCSC Xena GDC FPKM-UQ Matrix (log2(x+1) transformed)                   |
+-------------------------------------------+--------------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------------+
|                               Labeling & Filtering Step                                |
|  - Barcode chars 14-15: '01' = Primary Tumor (Class 1), '11' = Solid Normal (Class 0)  |
|  - Drop all other sample types (metastatic, blood normal, formalin-fixed)              |
+-------------------------------------------+--------------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------------+
|                          Patient-Level Stratified Partition                            |
|  - Patient ID = Barcode chars 1-12; group samples by patient to prevent cross-leakage  |
|  - 80% Train / Cross-Validation Pool  |  20% Held-Out Test Set (touched once at end)   |
+-------------------------------------------+--------------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------------+
|                           Class Imbalance Handling Protocol                            |
|  - Stratified patient splits to preserve ~11:1 tumor-to-normal ratio                   |
|  - Positive class weighting in BCE loss: w_pos = N_normal / N_tumor                    |
|  - Report balanced metrics: Sensitivity & Specificity alongside Accuracy & ROC-AUC     |
+-------------------------------------------+--------------------------------------------+
                                            |
                                            v
+----------------------------------------------------------------------------------------+
|                Shared In-Fold Feature Selection (Training Portion Only)                |
|  - Filter near-zero variance (Var > 0.1) & low expression (Mean > 1.0)                 |
|  - Welch's two-sample t-test (unequal variance): Select Top 200 DE Genes               |
+-------------------------------------+--------------------------------------------------+
                                      |
         +----------------------------+-----------------------------+
         |                                                          |
         v                                                          v
+----------------------------------+       +---------------------------------------------+
|    Mutual Information Ranking    |       |         Mutual Information Ranking          |
|      Select Top 8 Genes          |       |            Select Top 50 Genes              |
|     (1 feature per qubit)        |       |        (Reference high-feature set)         |
+----------------+-----------------+       +----------------------+----------------------+
                 |                                                |
                 v                                                v
+----------------------------------+       +---------------------------------------------+
|    MinMaxScaler -> [0, pi]       |       |           MinMaxScaler -> [0, pi]           |
|    (Fit on train portion only)   |       |         (Fit on train portion only)         |
+----------------+-----------------+       +----------------------+----------------------+
                 |                                                |
  +--------------+---------------+                                |
  |              |               |                                |
  v              v               v                                v
+--------------+ +-------------+ +------------------+ +----------------------------------+
| Hybrid       | | Classical   | | Classical        | | Reference Classical Baselines    |
| Quantum      | | Ablation    | | Baselines (Same  | | (50 Genes - Reference Set)       |
| Circuit      | | Model       | | 8 Genes & Folds) | |                                  |
| - 8 Qubits   | | - Linear    | | - Tuned SVM-RBF  | | - Tuned SVM-RBF                  |
| - Depth L=2  | |   (8 -> 4)  | | - Random Forest  | | - Random Forest (200 trees)      |
| - 57 Params  | | - Tanh      | |   (200 trees)    | | - XGBoost (200 trees)            |
|   (48 Q+9 C) | | - Linear    | | - XGBoost        | |                                  |
|              | |   (4 -> 1)  | |   (200 trees)    | |                                  |
|              | | - 41 Params | |                  | |                                  |
+-------+------+ +------+------+ +--------+---------+ +----------------+-----------------+
        |               |                 |                            |
        +---------------+-----------------+----------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
|                               Robustness Experiments                                   |
|  - Learning Curves: Subsampled training fractions (10%, 25%, 50%, 100%)                |
|  - Noise Simulation: Depolarizing channel on default.mixed device (p = 0.01, 0.05)     |
+-----------------------------------------+----------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
|                          Statistical Evaluation & Validation                           |
|  - Validation Threshold Tuning: Optimal threshold chosen via Youden's J (TPR - FPR)   |
|  - Repeated Stratified 5-Fold CV across 5 random seeds (25 evaluations per model)      |
|  - Statistical Testing: Paired Wilcoxon signed-rank test (Limitation: CV fold overlap  |
|    induces sample dependency; corrected resampled t-test is recommended alternative)   |
+-----------------------------------------+----------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
|                                Final Test-Set Protocol                                 |
|  - Retrain pipeline on full 80% train/CV pool (rerun DE + MI + scaler fit on pool only)|
|  - Calibrate decision threshold on internal validation split from pool                 |
|  - Single final inference on 20% held-out test set (unseen, untouched during CV)       |
+-----------------------------------------+----------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
|                            Explainability & Interpretability                           |
|  - Permutation Feature Importance: Accuracy degradation across 30 shuffles per gene    |
|  - SHAP KernelExplainer: Black-box Shapley value estimation on quantum expectation    |
|  - Pathway Enrichment: GSEApy / Enrichr (KEGG 2021, GO Biological Process, MSigDB)    |
+-----------------------------------------+----------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
|                       External Validation (Independent Cohort)                         |
|  - Validation on independent GEO microarray/RNA-seq dataset with tumor & normal tissue |
|  - Status: [Not Yet Implemented in Codebase]                                           |
+-----------------------------------------+----------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------------+
|                     Clinical Decision Support Interface (app.py)                       |
|  - Streamlit dashboard: CSV gene expression upload, real-time hybrid risk prediction,  |
|    gene importance visualization, and multi-model benchmark summary tables             |
+----------------------------------------------------------------------------------------+
```

---

## Quantum Circuit Specification

### 1. State Preparation / Feature Embedding
Input features $\mathbf{x} = [x_0, x_1, \dots, x_7]^T \in [0, \pi]^8$ are scaled via `MinMaxScaler(feature_range=(0, np.pi))` fit exclusively on the training partition. Each feature is encoded as an individual qubit rotation on ground state $|0\rangle^{\otimes 8}$ using `qml.AngleEmbedding(rotation="Y")`:

$$|\psi_0(\mathbf{x})\rangle = \bigotimes_{j=0}^7 R_Y(x_j)|0\rangle = \bigotimes_{j=0}^7 \begin{pmatrix} \cos\left(\frac{x_j}{2}\right) \\ \sin\left(\frac{x_j}{2}\right) \end{pmatrix}$$

### 2. Variational Entangling Layers
The parameterized quantum circuit employs `qml.StronglyEntanglingLayers` with layer depth $L = 2$ on $N = 8$ qubits (`wires=range(8)`).
- **Weight Tensor Shape**: $(L, N, 3) = (2, 8, 3) = 48$ trainable rotation parameters $\mathbf{\theta}$.
- **Layer Transformation**: Each layer $l \in \{1, \dots, L\}$ applies arbitrary single-qubit Euler rotations followed by circular entangler CNOT gates with execution range $r_l$:
  $$U_l(\mathbf{\theta}_l) = \left( \prod_{j=0}^{N-1} \text{CNOT}_{(j, (j + r_l) \pmod N)} \right) \left( \bigotimes_{j=0}^{N-1} R(\alpha_{l, j}, \beta_{l, j}, \gamma_{l, j}) \right)$$
  where $R(\alpha, \beta, \gamma) = R_Z(\gamma) R_Y(\beta) R_Z(\alpha)$.

### 3. Measurement & Expectation Values
Pauli-Z expectation values are measured across all 8 wires:

$$\langle Z_j \rangle = \langle \psi(\mathbf{x}, \mathbf{\theta}) | Z_j | \psi(\mathbf{x}, \mathbf{\theta}) \rangle \in [-1, 1], \quad \forall j \in \{0, \dots, 7\}$$

### 4. Classical Readout Head
The measured expectation vector $\langle \mathbf{Z} \rangle \in [-1, 1]^8$ passes through a single classical affine layer (`nn.Linear(8, 1)`) followed by a standard logistic sigmoid activation:

$$\hat{y} = \sigma\left(\mathbf{w}^T \langle \mathbf{Z} \rangle + b\right) = \frac{1}{1 + e^{-(\mathbf{w}^T \langle \mathbf{Z} \rangle + b)}}$$

- **Parameter Count Breakdown**:
  - Quantum Circuit Parameters: $L \times N \times 3 = 2 \times 8 \times 3 = 48$
  - Classical Linear Parameters: 8 weights $+ 1$ bias $= 9$
  - **Total Hybrid Trainable Parameters**: $48 + 9 = 57$

---

## Parameter-Matched Classical Ablation Architecture

To rigorously evaluate whether parameterized quantum Hilbert space entanglement provides diagnostic utility over an equivalent classical capacity, the quantum circuit is ablated and replaced by a compact multi-layer perceptron (`ClassicalAblationModel` in `src/models_quantum.py`):

1. **Hidden Layer**: $\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$, where $\mathbf{W}_1 \in \mathbb{R}^{4 \times 8}, \mathbf{b}_1 \in \mathbb{R}^4$ ($8 \times 4 + 4 = 36$ parameters). Hyperbolic tangent ($\tanh$) maps hidden representations to $[-1, 1]$, matching the range of Pauli-Z expectations.
2. **Output Layer**: $\hat{y}_{\text{abl}} = \sigma(\mathbf{w}_2^T \mathbf{h} + b_2)$, where $\mathbf{w}_2 \in \mathbb{R}^4, b_2 \in \mathbb{R}$ ($4 \times 1 + 1 = 5$ parameters).
3. **Exact Parameter Comparison**:
   - Hybrid Model: **57** parameters (48 quantum + 9 classical)
   - Classical Ablation: **41** parameters (36 hidden + 5 output)
   - **Parameter Matching Rationale**: The ablation layer size is computed dynamically as $\text{hidden} = \max\left(1, \lfloor(n_{\text{quantum\_params}} - 1) / (n_{\text{inputs}} + 2)\rfloor\right) = \lfloor 47 / 10 \rfloor = 4$. Due to discrete integer dimensions of neural layer matrices, a hidden width of 4 yields 41 total parameters, whereas a hidden width of 5 would yield $(8 \times 5 + 5) + (5 \times 1 + 1) = 51$ parameters. The ablation model thus operates with fewer total parameters (41 vs. 57) than the hybrid model, providing a strict lower bound on classical capacity.

---

## Statistical Testing & Methodological Limitations

- **Current Implementation**: The codebase executes a paired two-sided Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) comparing fold ROC-AUC scores between the hybrid quantum model and each classical baseline.
- **Methodological Limitation**: Because cross-validation folds resample and share training instances across folds, the resulting test-fold error estimates violate the independent and identically distributed (i.i.d.) assumption of the standard Wilcoxon test. This induced variance underestimation can lead to artificially optimistic p-values (inflated Type I error rates).
- **Recommended Alternative**: The corrected resampled t-test proposed by Nadeau and Bengio (2003) explicitly adjusts the variance estimate by a correction factor $\left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)$ to account for fold overlap, and represents the preferred statistical test for future iterations.

---

## Variant: Data Re-uploading (Experimental)

### Conceptual Formulation
Data re-uploading (Pérez-Salinas et al., 2020) provides an alternative variational circuit ansatz where classical features are repeatedly encoded into the quantum state at multiple stages of the circuit rather than strictly at state initialization.

For $L$ variational blocks, the state evolves as:

$$|\psi(\mathbf{x}, \mathbf{\theta})\rangle = \left( \prod_{l=1}^L W_l(\mathbf{\theta}_l) S(\mathbf{x}) \right) |0\rangle^{\otimes N}$$

where $S(\mathbf{x}) = \bigotimes_{j=0}^{N-1} R_Y(x_j)$ re-embeds the normalized input features, and $W_l(\mathbf{\theta}_l)$ applies entangling operations. By interleaving feature re-encoding with entanglement, data re-uploading enables single- and few-qubit circuits to approximate higher-frequency Fourier components, expanding expressivity without expanding physical qubit counts.

### Implementation Status
- **Status**: **Optional / Experimental benchmark; not yet implemented in the current codebase.**
- The current implementation in `src/models_quantum.py` utilizes single-shot initial `qml.AngleEmbedding` followed by $L=2$ uninterrupted `qml.StronglyEntanglingLayers`.
