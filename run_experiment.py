import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
import mlflow.sklearn

# 1. Load Real Public E-Commerce Customer Spending Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/Retail_Analysis.csv"
try:
    df = pd.read_csv(url)
except:
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url)
    df.columns = ['month', 'transaction_volume']
    df['transaction_volume'] = pd.to_numeric(df['transaction_volume'], errors='coerce')
    df = df.dropna()

if 'transaction_volume' not in df.columns:
    df.columns = [col.lower() for col in df.columns]
    num_cols = df.select_dtypes(include=[np.number]).columns
    df['transaction_volume'] = df[num_cols[0]]

df['customer_age_proxy'] = np.random.randint(18, 65, size=len(df))
df['items_purchased'] = np.random.randint(1, 10, size=len(df))
df['target'] = df['transaction_volume'] * 1.15

X = df[['transaction_volume', 'customer_age_proxy', 'items_purchased']].dropna()
y = df['target'].loc[X.index]

full_data = X.copy()
full_data['target'] = y

# Safely split dataset into Reference (60%) and Current (40%)
train_size = int(len(full_data) * 0.6)
reference_data = full_data.iloc[:train_size]
current_data = full_data.iloc[train_size:]

# Introduce artificial drift in current data (promotional spike in transaction volume)
current_data['transaction_volume'] = current_data['transaction_volume'] * 1.40  

# 2. Set up MLflow Experiment
mlflow.set_experiment("MLOps_Data_Drift_Experiment")

with mlflow.start_run() as run:
    print(f"--- MLflow Run Started: {run.info.run_id} ---")
    
    X_ref = reference_data.drop(columns=['target'])
    y_ref = reference_data['target']
    
    X_curr = current_data.drop(columns=['target'])
    y_curr = current_data['target']

    # Train Model on Reference Data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_ref, y_ref)

    # Predict & Evaluate (Using RMSE)
    rmse_ref = np.sqrt(mean_squared_error(y_ref, model.predict(X_ref)))
    rmse_curr = np.sqrt(mean_squared_error(y_curr, model.predict(X_curr)))

    # Log Parameters & Metrics to MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("reference_rmse", rmse_ref)
    mlflow.log_metric("current_rmse", rmse_curr)
    
    print(f"Reference RMSE: {rmse_ref:.4f}")
    print(f"Current RMSE (with drift): {rmse_curr:.4f}")

    # 3. Perform Statistical Data Drift Detection (KS-Test per feature)
    drift_results = []
    for col in X_ref.columns:
        stat, p_value = stats.ks_2samp(X_ref[col], X_curr[col])
        is_drift = p_value < 0.05  # Significance threshold
        drift_results.append({
            "feature": col,
            "p_value": p_value,
            "drift_detected": is_drift
        })
    
    drift_df = pd.DataFrame(drift_results)
    drifted_features_count = drift_df['drift_detected'].sum()
    mlflow.log_metric("drifted_features_count", int(drifted_features_count))

    # 4. Generate HTML Drift Report Natively
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/data_drift_report.html"
    
    html_content = f"""
    <html>
    <head>
        <title>E-Commerce Data Drift Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f9f9f9; color: #333; }}
            h1 {{ color: #2c3e50; }}
            table {{ width: 100%; border-collapse: collapse; background: #fff; margin-top: 20px; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #34495e; color: white; }}
            .drift {{ color: red; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>E-Commerce Customer Retail Transaction - Data Drift Analysis Report</h1>
        <p><b>Model:</b> Random Forest Regressor</p>
        <p><b>Reference RMSE:</b> {rmse_ref:.4f}</p>
        <p><b>Current RMSE (with drift):</b> {rmse_curr:.4f}</p>
        <p><b>Total Drifted Features (p < 0.05):</b> {drifted_features_count} / {len(X_ref.columns)}</p>
        
        <h2>Feature-Level Kolmogorov-Smirnov Test Results</h2>
        {drift_df.to_html(classes='table', index=False)}
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    mlflow.log_artifact(report_path)
    print(f"Data Drift Report saved and logged to MLflow at: {report_path}")

print("--- Experiment Completed Successfully! ---")