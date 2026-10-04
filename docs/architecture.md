# Hybrid Quantum-Classical ML Architecture

## System Architecture Overview

```
+----------------------------------------------------------------------------------------------+
|                                TCGA-BRCA Gene Expression Data                                |
|                     UCSC Xena GDC FPKM-UQ Matrix (log2(x+1) Transformed)                     |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                                  Labeling & Filtering Step                                   |
|  - Barcode Chars 14-15: '01' = Primary Tumor (Class 1), '11' = Solid Tissue Normal (Class 0) |
|  - Drop All Other Barcodes (Metastatic '06', Blood Normal '10', Formalin-Fixed Samples)      |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                              Patient-Level Stratified Partition                              |
|  - Patient ID = Barcode Chars 1-12 (Groups Primary & Normal Tissue per Donor)                |
|  - 80% Train / Cross-Validation Pool   |   20% Held-Out Test Set (Evaluated ONCE at End)     |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                              Class Imbalance Handling Protocol                               |
|  - Empirical Tumor:Normal Ratio Dynamically Computed from Data (TCGA Empirical ~10.9:1)      |
|  - Stratified Patient Splits to Prevent Partition Bias Across Classes                        |
|  - Positive Class Weighting in BCE Loss: w_pos = N_normal / N_tumor                          |
|  - Balanced Metrics: Sensitivity & Specificity Reported Alongside ROC-AUC and Accuracy      |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                   Shared In-Fold Feature Selection (Training Partition Only)                 |
|  - Variance Filtering (Var > 0.1) & Low-Expression Filtering (Mean Expression > 1.0)         |
|  - Single Shared Welch's Two-Sample t-Test (Unequal Variance) -> Top 200 DE Genes           |
+----------------------------------------------+-----------------------------------------------+
                                               |
                 +-----------------------------+-------------------------------+
                 |                                                             |
                 v                                                             v
+----------------------------------------------------------------------+ +--------------------+
|                Top 8 Genes Qubit & Model Branch                      | | Reference Branch   |
|  - Mutual Information Ranking on Top 200 DE -> Top 8 Qubit Genes    | | - MI Ranking ->    |
|  - MinMaxScaler -> [0, pi] (Fit on Training Partition Only)          | |   Top 50 Genes     |
+----------------------------------+-----------------------------------+ | - Scaler -> [0, pi]|
                                   |                                     +---------+----------+
         +-------------------------+-------------------------+                     |
         |                         |                         |                     |
         v                         v                         v                     v
+--------------------+  +----------------------+  +--------------------+ +--------------------+
| Hybrid Quantum     |  | Classical Ablations  |  | Classical Baselines| | Reference Baselines|
| Model              |  | (Sandwich Design)    |  | (Same 8 Genes/Fold)| | (50 Genes / Fold)  |
| - 8 Qubits (Ry)    |  | - Ablation-5: 51 Par |  | - Tuned SVM-RBF    | | - Tuned SVM-RBF    |
| - StronglyEntangle |  |   Linear(8,5) + Tanh |  | - Random Forest    | | - Random Forest    |
|   (Depth L=2)      |  | - Ablation-6: 61 Par |  |   (200 Trees)      | |   (200 Trees)      |
| - 57 Trainable Par |  |   Linear(8,6) + Tanh |  | - XGBoost          | | - XGBoost          |
|   (48 Q + 9 C)     |  | (Bounds 57p Hybrid)  |  |   (200 Trees)      | |   (200 Trees)      |
+---------+----------+  +----------+-----------+  +---------+----------+ +---------+----------+
          |                        |                        |                      |
          +------------------------+------------------------+----------------------+
                                   |
                                   v
+----------------------------------------------------------------------------------------------+
|                         Robustness Experiments & Noisy Simulation                            |
|  - Learning Curves: Empirical Sample-Efficiency (10%, 25%, 50%, 100% Training Subsets)       |
|  - Noisy Simulation: Depolarizing Noise Channel on default.mixed Device (p = 0.01, 0.05)     |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                     Repeated Cross-Validation & Statistical Rigor                            |
|  - Repeated Stratified 5-Fold CV across 5 Random Seeds (25 Resamples Total per Model)        |
|  - Validation Threshold Calibration via Youden's J Statistic (TPR - FPR)                     |
|  - Paired Wilcoxon Signed-Rank Test (Reported with Fold-Overlap Correlation Caveat)          |
|  - Nadeau-Bengio Corrected Resampled t-Test with Variance Factor (1/25 + n_test/n_train)     |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                         Held-Out Test Set Protocol (Evaluated ONCE)                          |
|  - Retrain Full Pipeline on 80% Train/CV Pool (Re-run Welch DE + MI + Scaler on Pool Only)   |
|  - Calibrate Final Optimal Decision Threshold on 15% Internal Split from Train/CV Pool       |
|  - Single Inference on 20% Held-Out Unseen Test Set (Saved to results/test_results.csv)     |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                       Explainability, Stability & Pathway Enrichment                         |
|  - Permutation Feature Importance: Evaluated via ROC-AUC Drop (Accounts for Class Imbalance) |
|  - SHAP KernelExplainer: Black-Box Shapley Attribution for Hybrid Quantum Predictions       |
|  - Selection Stability: Gene Recurrence Frequency Across All 25 Folds/Seeds (Top Qubit Genes)|
|  - Biological Pathway Enrichment: GSEApy/Enrichr on Top 200 DE Genes (KEGG, Reactome, GO)    |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|             External Cohort Validation (GSE42568 Zero-Shot Generalization)                   |
|  - Independent Microarray Cohort: GSE42568 (104 Tumor, 17 Normal Breast Tissue Samples)      |
|  - Cohort Rank Normalization: Within-Cohort Percentile Harmonization Across Shared 8 Genes   |
|  - Zero-Shot Inference: Frozen TCGA 8-Gene Models Tested Without Re-Selection on GEO Cohort  |
+----------------------------------------------+-----------------------------------------------+
                                               |
                                               v
+----------------------------------------------------------------------------------------------+
|                    Clinical Decision Support Web Application (app.py)                        |
|  - Streamlit Dashboard: Real-Time Patient Gene Expression Upload & Feature Validation        |
|  - Risk Stratification Probability, Multi-Model Radar/Metric Comparison & Gene SHAP Display  |
+----------------------------------------------------------------------------------------------+
```

---

## Quantum Circuit Specification

### 1. State Preparation / Feature Embedding
Input features $\mathbf{x} = [x_0, x_1, \dots, x_7]^T \in [0, \pi]^8$ are scaled via `MinMaxScaler(feature_range=(0, np.pi))` fit exclusively on the training partition of each fold. Each feature is encoded as an individual qubit rotation on the ground state $|0\rangle^{\otimes 8}$ using `qml.AngleEmbedding(rotation="Y")`:

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

- **Trainable Parameter Breakdown**:
  - Quantum Circuit Parameters: $L \times N \times 3 = 2 \times 8 \times 3 = 48$
  - Classical Linear Readout: 8 weights $+ 1$ bias $= 9$
  - **Total Hybrid Trainable Parameters**: $48 + 9 = 57$

---

## Parameter-Matched Classical Ablation Architecture (Sandwich Design)

To rigorously determine whether parameterized Hilbert space entanglement provides diagnostic advantage over a classical neural network of equivalent capacity, the quantum circuit is ablated and replaced by classical feedforward neural network layers (`ClassicalAblationModel` in `src/models_quantum.py`).

### Integer Dimension Constraint & The Sandwich Solution
A classical multi-layer perceptron with $N=8$ inputs, $H$ hidden units, and 1 binary output has a total parameter count given by:
$$P(H) = (N \times H + H) + (H \times 1 + 1) = H(N + 2) + 1 = 10H + 1$$

To match the hybrid model's 57 total parameters, setting $10H + 1 = 57$ requires $H = 5.6$, which is mathematically non-integer. Because hidden layer dimensions must be integers, the pipeline constructs two bounding ablation architectures that strictly sandwich the 57-parameter hybrid model:

1. **Ablation-5 (Lower Bound, $H = 5$ hidden units)**:
   - Hidden Layer: $\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$, with $\mathbf{W}_1 \in \mathbb{R}^{5 \times 8}, \mathbf{b}_1 \in \mathbb{R}^5$ ($8 \times 5 + 5 = 45$ parameters).
   - Readout Head: $\hat{y} = \sigma(\mathbf{w}_2^T \mathbf{h} + b_2)$, with $\mathbf{w}_2 \in \mathbb{R}^5, b_2 \in \mathbb{R}$ ($5 \times 1 + 1 = 6$ parameters).
   - **Total Parameters**: $45 + 6 = 51$ parameters ($-6$ relative to hybrid).
2. **Ablation-6 (Upper Bound, $H = 6$ hidden units)**:
   - Hidden Layer: $\mathbf{h} = \tanh(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$, with $\mathbf{W}_1 \in \mathbb{R}^{6 \times 8}, \mathbf{b}_1 \in \mathbb{R}^6$ ($8 \times 6 + 6 = 54$ parameters).
   - Readout Head: $\hat{y} = \sigma(\mathbf{w}_2^T \mathbf{h} + b_2)$, with $\mathbf{w}_2 \in \mathbb{R}^6, b_2 \in \mathbb{R}$ ($6 \times 1 + 1 = 7$ parameters).
   - **Total Parameters**: $54 + 7 = 61$ parameters ($+4$ relative to hybrid).

$$P(\text{Ablation-5}) = 51 < P(\text{Hybrid}) = 57 < P(\text{Ablation-6}) = 61$$

Both ablation models use $\tanh$ hidden activations to constrain internal representations to $[-1, 1]$, mirroring the $[-1, 1]$ bounds of Pauli-Z expectation values. Reporting both models guarantees an unambiguous, fair comparison free from parameter-mismatch artifacts.

---

## Statistical Testing & Resampling Protocol

### 1. Repeated Cross-Validation Structure
Evaluation uses Repeated Stratified 5-Fold Cross-Validation over 5 random seeds (seeds: `[42, 123, 456, 789, 101112]`), yielding $S \times K = 5 \times 5 = 25$ distinct evaluation resamples per model.

### 2. Paired Wilcoxon Signed-Rank Test
The codebase computes a two-sided paired Wilcoxon signed-rank test (`scipy.stats.wilcoxon`) comparing fold ROC-AUC scores between the hybrid model and each classical baseline (results saved to `results/statistical_tests.csv`).
- **Methodological Limitation**: Because cross-validation folds share training instances across folds and repeated seeds, error estimates violate the independent and identically distributed (i.i.d.) assumption. This induced variance underestimation produces optimistic p-values (inflated Type I error rates).

### 3. Nadeau-Bengio Corrected Resampled t-Test
To correct for sample dependency induced by overlapping training folds, the pipeline implements the corrected resampled t-test (Nadeau & Bengio, 2003) in `src/evaluate.py` (results saved to `results/nadeau_bengio_tests.csv`):

$$\hat{\sigma}^2_{\text{corrected}} = \left(\frac{1}{R} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s^2_d$$

$$t = \frac{\bar{d}}{\sqrt{\hat{\sigma}^2_{\text{corrected}}}}, \quad \text{df} = R - 1$$

where $R = 25$ is the total number of resamples ($S=5$ seeds $\times K=5$ folds), $d_r = \text{AUC}_{\text{hybrid}, r} - \text{AUC}_{\text{baseline}, r}$, $\bar{d}$ is the sample mean difference, and $s^2_d$ is the unbiased sample variance of differences. Because $R=25$ resamples are computed, the correction term explicitly incorporates $1/R = 1/25 = 0.04$ rather than $1/K = 1/5 = 0.20$.

---

## Final Held-Out Test Set Protocol

To provide an unbiased assessment of generalization:
1. **Partitioning**: A patient-level stratified 80/20 split partitions the dataset into a train/CV pool (80%) and a held-out test set (20%). The held-out test set is sealed and never accessed during CV or model selection.
2. **Retraining**: Following CV completion, feature selection (`select_features_hierarchical`) and MinMaxScaler fitting are executed exclusively on the 80% train/CV pool.
3. **Threshold Calibration**: A 15% internal validation split from the train/CV pool is used to fit the optimal classification threshold via Youden's $J = \text{Sensitivity} + \text{Specificity} - 1$.
4. **Single Inference**: The retrained models predict on the 20% held-out test set exactly once using the calibrated threshold (saved to `results/test_results.csv`).

---

## External Cohort Validation (GSE42568)

To evaluate zero-shot generalization across distinct profiling technologies (RNA-seq vs. Microarray):
1. **Dataset**: Gene Expression Omnibus (GEO) accession `GSE42568` containing 104 breast tumor biopsies and 17 normal breast tissue controls profiled on the Affymetrix Human Genome U133 Plus 2.0 array.
2. **Frozen Feature Set**: The exact 8 genes selected on the TCGA training pool are evaluated on GEO without re-selection.
3. **Cohort Rank Normalization**: Microarray hybridization intensities and RNA-seq FPKM-UQ values occupy drastically different numerical dynamic ranges. Each cohort is normalized independently via percentile rank transformation across shared genes (`rank_normalize_cohort` in `src/data.py`):
   $$\tilde{x}_{ij} = \frac{\text{rank}(x_{ij}) - 1}{P - 1} \in [0, 1]$$
4. **Zero-Shot Inference**: Scaler parameters fit on the normalized TCGA pool transform the normalized GEO cohort, and frozen TCGA-trained models perform zero-shot classification (saved to `results/geo_test_results.csv`).

---

## Explainability, Stability & Pathway Enrichment

1. **Permutation Importance**: Evaluated by computing the degradation in **ROC-AUC** (rather than accuracy) when permuting each of the 8 genes across 30 iterations, providing robust attribution under severe class imbalance (~10.9:1).
2. **SHAP KernelExplainer**: Black-box Shapley additive attributions computed on quantum expectation states to quantify single-patient gene contributions.
3. **Selection Stability**: Computed across all 25 CV folds/seeds (`compute_selection_stability` in `src/explain.py`) to report recurrence frequencies of qubit genes (saved to `results/selection_stability.csv` and `results/selection_stability.png`).
4. **Pathway Enrichment**: To avoid small-gene-set bias from querying only 8 genes, pathway enrichment (`gseapy` / Enrichr) queries the **top 200 differentially expressed genes** identified by the shared Welch t-test across KEGG 2021, GO Biological Process, and Reactome (saved to `results/pathway_enrichment.csv`).

---

## Variant: Data Re-uploading (Experimental Benchmark)

### Formulation
Data re-uploading (Pérez-Salinas et al., 2020) provides an alternative variational circuit ansatz where input features $\mathbf{x}$ are repeatedly re-encoded between entangling layers rather than encoded once at initialization:

$$|\psi(\mathbf{x}, \mathbf{\theta})\rangle = \left( \prod_{l=1}^L W_l(\mathbf{\theta}_l) S(\mathbf{x}) \right) |0\rangle^{\otimes N}$$

where $S(\mathbf{x}) = \bigotimes_{j=0}^{N-1} R_Y(x_j)$ re-embeds the normalized input features, and $W_l(\mathbf{\theta}_l)$ applies parameterized Euler rotations and circular CNOT entanglement. Interleaving feature re-encoding with entanglement expands the accessible Fourier spectrum, enhancing circuit expressivity without requiring additional physical qubits.

### Implementation Status
- **Status**: **Optional / Experimental benchmark; not implemented in the primary pipeline.**
- The primary pipeline uses single-shot initial `qml.AngleEmbedding` followed by $L=2$ uninterrupted `qml.StronglyEntanglingLayers` on `default.qubit` (and `default.mixed` for noise experiments).

---

## Clinical Decision Support System (app.py)

The project includes an interactive clinical decision support dashboard implemented in Streamlit (`app.py`):
- **Upload Interface**: Accepts patient CSV files containing normalized gene expression values for the selected panel.
- **Model Inference**: Loads the serialized PyTorch hybrid model (`results/trained_model.pt`) and fitted scaler (`results/scaler.pkl`) to compute risk probabilities and binary classifications using the calibrated decision threshold.
- **Explainability Display**: Interactive gene importance bar charts and radar plots illustrating multi-model concordance (Hybrid vs. Ablations vs. Classical baselines).
