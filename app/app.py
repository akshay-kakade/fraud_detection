import gradio as gr
import pandas as pd
import joblib
import json
import os

# Load artifacts
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "fraud_model.pkl"))
scaler = joblib.load(os.path.join(base_path, "preprocessor.pkl"))

with open(os.path.join(base_path, "feature_names.json"), "r") as f_in:
    feature_cols = json.load(f_in)
with open(os.path.join(base_path, "metrics.json"), "r") as f_in:
    model_metrics = json.load(f_in)

def batch_inference(file):
    try:
        batch_df = pd.read_csv(file.name)
        # Data processing logic
        if 'Amount' in batch_df.columns and 'scaled_amount' not in batch_df.columns:
            batch_df['scaled_amount'] = scaler.transform(batch_df['Amount'].values.reshape(-1,1))
        if 'Time' in batch_df.columns and 'scaled_time' not in batch_df.columns:
            batch_df['scaled_time'] = scaler.transform(batch_df['Time'].values.reshape(-1,1))
        
        # Select features
        processed_df = batch_df[feature_cols]
        probs = model.predict_proba(processed_df)[:, 1]
        batch_df['Fraud_Probability'] = [f"{p:.2%}" for p in probs]
        batch_df['Status'] = ["🚨 FRAUD" if p > 0.5 else "✅ OK" for p in probs]
        
        return batch_df.head(50)
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})

with gr.Blocks(title="Enterprise Fraud Batch Processor") as demo:
    gr.Markdown("# 🛡️ Enterprise Fraud Detection System")
    gr.Markdown("### High-Precision Batch Transaction Analysis")
    
    with gr.Tab("Batch CSV Upload"):
        gr.Markdown("Upload a CSV file containing transaction data. The system will automatically scale 'Time' and 'Amount' and flag suspicious activities.")
        file_input = gr.File(label="Transaction CSV")
        batch_btn = gr.Button("Process Transactions", variant="primary")
        batch_output = gr.DataFrame(label="Detection Results")
        batch_btn.click(batch_inference, inputs=file_input, outputs=batch_output)

    with gr.Tab("Model & Project Info"):
        gr.Markdown(f"""
        ### **Project Summary**
        This system solves the challenge of **Credit Card Fraud Detection** in highly imbalanced datasets (where fraud accounts for only 0.17% of transactions).

        **What we built:**
        - An optimized **XGBoost Classifier** using cost-sensitive learning (`scale_pos_weight`).
        - A pipeline that prioritizes **Precision** to minimize false positives while maintaining high recall.
        
        **Current Performance Metrics:**
        - **Precision:** {model_metrics.get('precision', 0):.2%}
        - **Recall:** {model_metrics.get('recall', 0):.2%}
        - **ROC-AUC:** {model_metrics.get('roc_auc', 0):.3f}
        
        **Technical Implementation:**
        - Handled severe class imbalance without synthetic data artifacts.
        - Automated scaling for transaction metadata.
        """)

demo.launch()
