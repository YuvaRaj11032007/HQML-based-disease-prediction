"""
Streamlit Dashboard for Hybrid Quantum-Classical Breast Cancer Detection.

Upload a CSV of expression values for the selected genes, get:
- Hybrid model prediction and risk score
- Gene importance chart
- Comparison table of all models

Usage: streamlit run app.py
"""

import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Must be first Streamlit call
st.set_page_config(
    page_title="Breast Cancer Detection - Hybrid QML",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def load_model_and_config():
    """Load trained model, scaler, and selected genes."""
    genes_path = "results/selected_genes.json"
    model_path = "results/trained_model.pt"
    scaler_path = "results/scaler.pkl"

    if not os.path.exists(genes_path):
        return None, None, None, "No trained model found. Run `python -m src.train` first."

    with open(genes_path) as f:
        selected_genes = json.load(f)

    n_qubits = len(selected_genes)

    # Import model class
    from src.models_quantum import HybridQuantumModel

    model = HybridQuantumModel(n_qubits=n_qubits, n_layers=2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler, selected_genes, None


def main():
    st.title("🧬 Hybrid Quantum-Classical ML for Breast Cancer Detection")
    st.markdown(
        """
        This dashboard uses a **hybrid quantum-classical** machine learning model
        to classify breast tissue samples as **tumor** or **normal** based on
        gene expression data from TCGA-BRCA.

        ---
        """
    )

    model, scaler, selected_genes, error = load_model_and_config()

    if error:
        st.error(error)
        st.info(
            "To train the model, run:\n```\npython -m src.train --config config.yaml --synthetic --fast\n```"
        )
        return

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown(
        f"""
        **Model**: Hybrid Quantum-Classical\n
        **Qubits**: {len(selected_genes)}\n
        **Selected Genes**: {', '.join(selected_genes[:4])}...\n
        **Architecture**: AngleEmbedding + StronglyEntanglingLayers
        """
    )

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Prediction", "📊 Results", "📈 Gene Importance"])

    # ================================================================
    # Tab 1: Prediction
    # ================================================================
    with tab1:
        st.header("Upload Gene Expression Data")
        st.markdown(
            f"Upload a CSV file with columns matching the selected genes:\n"
            f"`{', '.join(selected_genes)}`"
        )

        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.dataframe(input_df.head())

                # Validate columns
                missing = [g for g in selected_genes if g not in input_df.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    X_input = input_df[selected_genes].values

                    # Scale
                    if scaler is not None:
                        X_scaled = scaler.transform(X_input)
                    else:
                        from sklearn.preprocessing import MinMaxScaler
                        s = MinMaxScaler(feature_range=(0, np.pi))
                        X_scaled = s.fit_transform(X_input)

                    # Predict
                    with torch.no_grad():
                        X_t = torch.tensor(X_scaled, dtype=torch.float32)
                        probs = model(X_t).numpy().flatten()

                    # Display results
                    results_df = input_df.copy()
                    results_df["Risk Score"] = probs
                    results_df["Prediction"] = [
                        "🔴 Tumor" if p >= 0.5 else "🟢 Normal" for p in probs
                    ]
                    results_df["Confidence"] = [
                        f"{max(p, 1-p)*100:.1f}%" for p in probs
                    ]

                    st.subheader("Predictions")
                    st.dataframe(
                        results_df[["Prediction", "Risk Score", "Confidence"]],
                        use_container_width=True,
                    )

                    # Summary
                    n_tumor = (probs >= 0.5).sum()
                    n_normal = (probs < 0.5).sum()

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Samples", len(probs))
                    col2.metric("Predicted Tumor", n_tumor)
                    col3.metric("Predicted Normal", n_normal)

                    # Risk distribution
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(probs, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
                    ax.axvline(0.5, color="red", linestyle="--", label="Threshold")
                    ax.set_xlabel("Risk Score")
                    ax.set_ylabel("Count")
                    ax.set_title("Risk Score Distribution")
                    ax.legend()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing file: {e}")
        else:
            # Demo mode
            st.info("No file uploaded. Using demo data.")
            demo_data = {gene: np.random.uniform(5, 15) for gene in selected_genes}
            demo_df = pd.DataFrame([demo_data])

            if scaler is not None:
                X_demo = scaler.transform(demo_df[selected_genes].values)
            else:
                X_demo = demo_df[selected_genes].values

            with torch.no_grad():
                demo_prob = model(torch.tensor(X_demo, dtype=torch.float32)).item()

            st.metric(
                "Demo Prediction",
                "Tumor" if demo_prob >= 0.5 else "Normal",
                f"Risk: {demo_prob:.3f}",
            )

    # ================================================================
    # Tab 2: Model Comparison
    # ================================================================
    with tab2:
        st.header("Cross-Validation Results")

        cv_path = "results/cv_results.csv"
        if os.path.exists(cv_path):
            cv_df = pd.read_csv(cv_path, index_col=0)
            st.dataframe(cv_df, use_container_width=True)
        else:
            st.warning("No CV results found. Run training first.")

        test_path = "results/test_results.csv"
        if os.path.exists(test_path):
            st.header("Held-Out Test Results")
            test_df = pd.read_csv(test_path, index_col=0)
            st.dataframe(test_df, use_container_width=True)

        stat_path = "results/statistical_tests.csv"
        if os.path.exists(stat_path):
            st.header("Statistical Tests (Wilcoxon)")
            stat_df = pd.read_csv(stat_path, index_col=0)
            st.dataframe(stat_df, use_container_width=True)

        # Show parameter counts
        param_path = "results/parameter_counts.csv"
        if os.path.exists(param_path):
            st.header("Parameter Counts")
            param_df = pd.read_csv(param_path)
            st.dataframe(param_df, use_container_width=True)

    # ================================================================
    # Tab 3: Gene Importance
    # ================================================================
    with tab3:
        st.header("Gene Importance Analysis")

        # Show importance plots
        for mname in ["hybrid", "svm", "rf", "xgboost"]:
            img_path = f"results/importance_{mname}.png"
            if os.path.exists(img_path):
                st.subheader(f"{mname.upper()} - Permutation Importance")
                st.image(img_path, use_container_width=True)

        shap_path = "results/shap_summary.png"
        if os.path.exists(shap_path):
            st.subheader("SHAP Values - Hybrid Model")
            st.image(shap_path, use_container_width=True)

        roc_path = "results/roc_comparison.png"
        if os.path.exists(roc_path):
            st.subheader("ROC Curve Comparison")
            st.image(roc_path, use_container_width=True)

        lc_path = "results/learning_curve.png"
        if os.path.exists(lc_path):
            st.subheader("Learning Curves")
            st.image(lc_path, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "⚠️ **Disclaimer**: This is a research prototype. Not clinically validated. "
        "Quantum simulation only (not real quantum hardware)."
    )


if __name__ == "__main__":
    main()
