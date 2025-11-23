import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.loaders import (
    load_assets, load_json, load_image
)
from utils.shap_helpers import (
    plot_global_shap, plot_sample_shap
)
from utils.training_helpers import plot_federated_training_curve

# --- Label Mapping ---
LABEL_MAPPING = {
    0: "DoS",
    1: "Exploits",
    2: "Normal",
    3: "Other",
    4: "Reconnaissance",
    5: "Web Attack"
}

# --- FEATURE GLOSSARY (Used in FedAvg/FedProx views) ---
FEATURE_GLOSSARY = {
    "rate": "The number of connections per second observed.",
    "sload": "Source bits per second (bytes/s from source to destination).",
    "dload": "Destination bits per second (bytes/s from destination to source).",
    "dur": "Total duration of the connection.",
    "dpkts": "The number of data packets sent by the destination.",
    "sjlt": "Source-to-Destination Jitter (variation in packet arrival time).",
    "sinpkt": "Source Inter-Packet Time (average time between source packets).",
    "dinpkt": "Destination Inter-Packet Time (average time between destination packets).",
    "dloss": "Percentage of data packets lost from destination to source.",
    "dbytes": "Number of data bytes transferred from destination to source.",
    "sloss": "Percentage of data packets lost from source to destination.",
    "spkts": "The number of data packets sent by the source.",
    "sbytes": "Number of data bytes transferred from source to source.",
    "proto": "Protocol type (e.g., tcp, udp, icmp).",
    "state": "Connection state (e.g., FIN, REQ, INT).",
    "service": "Network service (e.g., http, ftp, dns)."
}

# --- CUSTOM COLOR PALETTE (Muted Tones) ---
CUSTOM_PALETTE = {
    "DecisionTree": "#DC625F", 
    "FedAvg": "#4C9FE6",       
    "FedProx": "#0072B2"        
}

# -------------------------------------------------------------------
# PAGE CONFIGURATION and THEME SETUP (Light Mode)
# -------------------------------------------------------------------
st.set_page_config(page_title="Federated IDS Dashboard", layout="wide")
st.title(" Federated IDS Dashboard")

PLOTLY_THEME = "plotly_white"

# --- PATHS ---
DIRS = {
    "FedAvg": "data/fed_avg",
    "FedProx": "data/fed_prox",
    "DecisionTree": "data/DecisionTree"
}

# --- SIDEBAR CONTROLS ---
st.sidebar.title(" Controls")

choice = st.sidebar.selectbox("Select View", ["FedAvg", "FedProx", "Comparison"])

# -------------------------------------------------------------------
# FEDAVG / FEDPROX VIEW
# -------------------------------------------------------------------
if choice in ["FedAvg", "FedProx"]:
    path = DIRS[choice]
    try:
        model, shap_vals, base_vals, features, X_sample, y_sample, preds, meta = load_assets(path)
    except Exception as e:
        st.error(f"Error loading assets for {choice}: {e}")
        st.stop() 

    st.header(f"Model Results: **{choice}**")
    
    # --- Metrics Summary ---
    st.subheader(" Key Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy ", f"{meta.get('accuracy', 'N/A')}")
    col2.metric("Privacy (1 - ASR) ", f"{meta.get('privacy_score', 'N/A')}")
    col3.metric("Rounds ", meta.get('rounds', 'N/A'))

    # --- Training Curve ---
    st.subheader("Federated Training Curve")
    csv_path = os.path.join(path, "federated_training_results.csv")
    if os.path.exists(csv_path):
        fig_curve = plot_federated_training_curve(path)
        fig_curve.update_layout(template=PLOTLY_THEME)
        st.plotly_chart(fig_curve, use_container_width=True)
    else:
        st.info("No training results found for this model.")

    # --- Confusion Matrices ---
    st.subheader("Client Confusion Matrices")
    colA, colB = st.columns(2)
    img1 = load_image(f"{path}/Client1_confusion_matrix.png")
    img2 = load_image(f"{path}/Client2_confusion_matrix.png")
    if img1:
        colA.image(img1, caption="Client 1 Confusion Matrix")
    if img2:
        colB.image(img2, caption="Client 2 Confusion Matrix")
    
    # --- GLOBAL SHAP Importance ---
    st.subheader("Global Explanatory Analysis")
    
    if shap_vals is not None:
        st.markdown("#### Global Feature Importance (SHAP)")
        
        st.info("""
        The SHAP Summary Plot below displays the overall importance of features. 
        **The top of the plot indicates the features with the greatest influence** on the model's output across the dataset.
        """)
        
        fig_global = plot_global_shap(shap_vals, features)
        fig_global.update_layout(template=PLOTLY_THEME)
        st.plotly_chart(fig_global, use_container_width=True)
        
        shap_vals_arr = np.array(shap_vals)
        if shap_vals_arr.ndim == 3:
            mean_abs_shap = np.mean(np.abs(shap_vals_arr), axis=(0, 2))
        else:
            mean_abs_shap = np.mean(np.abs(shap_vals_arr), axis=0)
            
        top_feature_index = np.argmax(mean_abs_shap)
        top_feature_name = features[top_feature_index]
        
    else:
        st.warning("No SHAP values found for global analysis.")
        
    # --- Local SHAP Analysis ---
    if X_sample is not None and shap_vals is not None:
        st.markdown("#### Explore SHAP for Individual Samples")
        
        if isinstance(X_sample, str) and X_sample.endswith(".csv"):
            X_sample_df = pd.read_csv(X_sample)
        elif isinstance(X_sample, pd.DataFrame):
            X_sample_df = X_sample
        else:
            X_sample_df = pd.DataFrame(X_sample, columns=features[:X_sample.shape[1]])
        
        shap_vals_arr = np.array(shap_vals)
        max_samples = min(len(X_sample_df), shap_vals_arr.shape[0]) - 1

        idx = st.number_input(
            "Enter sample index to visualize SHAP values:",
            min_value=0,
            max_value=max_samples,
            value=0,
            step=1
        )
        idx = int(idx)

        fig_local = plot_sample_shap(shap_vals, X_sample_df, features, sample_index=idx)
        fig_local.update_layout(template=PLOTLY_THEME)
        st.plotly_chart(fig_local, use_container_width=True)

        # --- Text-based SHAP Explanation ---
        st.markdown("### SHAP Explanation Summary")
        
        shap_row_full = np.array(shap_vals)[idx]
        if shap_row_full.ndim > 1:
            shap_row = shap_row_full.mean(axis=1) 
        else:
            shap_row = shap_row_full

        min_len = min(len(features), len(shap_row))
        current_features = features[:min_len]
        shap_row = shap_row[:min_len]

        sorted_feats = sorted(
            zip(current_features, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_features = sorted_feats[:5]
        explanation_lines = []
        
        pred_label_name = "N/A"
        if preds is not None and len(preds) > idx:
            current_pred = preds[idx]
            pred_label_idx = int(np.argmax(current_pred)) if current_pred.ndim > 0 else int(current_pred)
            pred_label_name = LABEL_MAPPING.get(pred_label_idx, f"UNKNOWN ({pred_label_idx})")
        
        st.markdown(f"**Predicted Label:** `{pred_label_name}`")
        st.markdown("**Top 5 features influencing this prediction:**")

        for feat, val in top_features:
            direction = "↑ increases" if val > 0 else "↓ decreases"
            val_display = abs(val) * 1e9
            
            info = FEATURE_GLOSSARY.get(feat, "N/A")
            explanation_lines.append(f"- **{feat}** ({info}): {direction} the model output by **{val_display:.2f}n**") 
            
        st.markdown("\n".join(explanation_lines))
        st.markdown("---")
        
    else:
        st.warning("No sample data or SHAP values found for local analysis.")

    # --- FEATURE GLOSSARY ---
    st.subheader(" Feature Glossary")
    st.info("The meaning of the network flow features used in the model.")
    df_glossary = pd.DataFrame(FEATURE_GLOSSARY.items(), columns=["Feature", "Description"])
    st.dataframe(df_glossary, use_container_width=True, hide_index=True)


# -------------------------------------------------------------------
# COMPARISON TAB (Upliftment now in its own tab)
# -------------------------------------------------------------------
elif choice == "Comparison":
    st.header(" Model Comparison")

    meta_avg = load_json(f"{DIRS['FedAvg']}/federated_accuracy.json")
    meta_prox = load_json(f"{DIRS['FedProx']}/federated_accuracy.json")
    dt_meta_path = os.path.join(DIRS["DecisionTree"], "results.json")
    meta_dt = load_json(dt_meta_path) if os.path.exists(dt_meta_path) else {"accuracy": 0, "privacy_score": 0}

    dt_acc = meta_dt.get("accuracy", 0)
    dt_priv = meta_dt.get("privacy_score", 0)
    avg_acc = meta_avg.get("accuracy", 0)
    avg_priv = meta_avg.get("privacy_score", 0)
    prox_acc = meta_prox.get("accuracy", 0)
    prox_priv = meta_prox.get("privacy_score", 0)
    
    df_comp = pd.DataFrame({
        "Model": ["DecisionTree (Without FL)", "FedAvg", "FedProx"],
        "Accuracy": [dt_acc, avg_acc, prox_acc],
        "Privacy": [dt_priv, avg_priv, prox_priv]
    })
    
    # Define Tabs
    perf_tab, dt_tab, uplift_tab = st.tabs(["Performance Comparison", "Decision Tree Details", "Upliftment"])

    # --- PERFORMANCE COMPARISON TAB (Absolute Values) ---
    with perf_tab:
        st.subheader("Model Metrics (Absolute Values)")
        
        c1, c2 = st.columns(2)
        
        # Accuracy Chart
        with c1:
            st.markdown("#### Accuracy Comparison")
            fig_acc = px.bar(df_comp, x="Model", y="Accuracy", color="Model", text="Accuracy", 
                             title="Model Accuracy Comparison", 
                             color_discrete_map=CUSTOM_PALETTE)
            fig_acc.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_acc.update_layout(yaxis_title="Accuracy", xaxis_title=None, showlegend=False, template=PLOTLY_THEME)
            st.plotly_chart(fig_acc, use_container_width=True)

        # Privacy Chart
        with c2:
            st.markdown("#### Privacy Comparison (1 - ASR)")
            fig_priv = px.bar(df_comp, x="Model", y="Privacy", color="Model", text="Privacy", 
                             title="Model Privacy Comparison", 
                             color_discrete_map=CUSTOM_PALETTE)
            fig_priv.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig_priv.update_layout(yaxis_title="Privacy Score", xaxis_title=None, showlegend=False, template=PLOTLY_THEME)
            st.plotly_chart(fig_priv, use_container_width=True)
            
        st.markdown("#### Tabular Summary (Absolute Scores)")
        st.dataframe(df_comp, use_container_width=True)

    # --- DECISION TREE DETAILS TAB ---
    with dt_tab:
        st.subheader("Decision Tree Model Summary (Traditional ML Baseline)")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{meta_dt.get('accuracy', 'N/A')}")
        col2.metric("Privacy (1 - ASR)", f"{meta_dt.get('privacy_score', 'N/A')}")
        
        st.markdown("### Visual Insights from Decision Tree")
        img_path = os.path.join(DIRS["DecisionTree"], "confusion_matrix.png")
        
        if os.path.exists(img_path):
            img_conf = load_image(img_path)
            c_img, _ = st.columns([0.6, 0.4]) 
            with c_img:
                st.image(img_conf, caption="Decision Tree Confusion Matrix", width=450)
        else:
            st.info("Decision Tree confusion matrix image not found.")
            
        st.markdown("---")
        st.markdown("The Decision Tree serves as the non-federated baseline for comparison.")

    # --- UPLIFTMENT TAB (NEW) ---
    with uplift_tab:
        # CALCULATE UPLIFT OVER DECISION TREE
        
        # Avoid ZeroDivisionError if baseline scores are zero
        def calculate_uplift(fl_score, dt_score):
            return ((fl_score - dt_score) / dt_score) * 100 if dt_score else 0

        uplift_data = {
            "Metric": ["Accuracy Uplift", "Privacy Uplift"],
            "FedAvg Improvement": [
                calculate_uplift(avg_acc, dt_acc),
                calculate_uplift(avg_priv, dt_priv)
            ],
            "FedProx Improvement": [
                calculate_uplift(prox_acc, dt_acc),
                calculate_uplift(prox_priv, dt_priv)
            ]
        }
        df_uplift = pd.DataFrame(uplift_data)
        
        st.subheader(" Performance Uplift: Federated Learning vs. Traditional ML Baseline")
        st.success("This section prominently displays the quantitative value and improvement gained by using Federated Learning.")
        
        # Melt the uplift data for a cleaner bar chart
        df_uplift_melted = df_uplift.melt(id_vars='Metric', 
                                          var_name='Model', 
                                          value_name='Improvement (%)')
        
        # Generate the Uplift Bar Chart
        fig_uplift = px.bar(df_uplift_melted, 
                            x="Model", 
                            y="Improvement (%)", 
                            color="Metric", 
                            barmode="group",
                            title="Percentage Improvement over Decision Tree Baseline",
                            color_discrete_map={'Accuracy Uplift': "#E85C5F", 'Privacy Uplift': '#0072B2'})
        
        fig_uplift.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig_uplift.update_layout(yaxis_title="Improvement (%)", template=PLOTLY_THEME)
        
        st.plotly_chart(fig_uplift, use_container_width=True)
        
        # Display Uplift Table
        st.markdown("#### Tabular Summary of Improvement")
        st.dataframe(df_uplift.set_index('Metric').style.format("{:.2f}%"), use_container_width=True)