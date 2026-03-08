import time
from contextlib import contextmanager
from turtle import pd
import math

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb

import torch
import torch.nn as nn
from torch.nn import functional as f
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from itertools import chain

@contextmanager
def timeit(action="Timing"):
    """ Print the execution time of certain Python operations. """
    # Record start time
    print(f"{action} started...")
    start_time = time.time()
    
    # Execute task
    yield
    
    # Compute and show elapsed time
    elapsed_time = time.time()-start_time
    print(f"{action} completed. Elapsed time: {elapsed_time:.2f}s\n")

# Name: recall_at_fixed_fpr
# Description: Calculate recall at a specific false positive rate (FPR) threshold. (Defaults to 1% FPR)
def recall_at_fixed_fpr(y_true, y_scores, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Find the indices where FPR is less than or equal to our target (e.g., 1%)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0
    # Return the recall at the highest allowed FPR
    return tpr[idx[-1]]

# Name: select_threshold_at_fpr
# Description: Find the probability threshold that corresponds to a specific FPR (e.g., 1% FPR) on the validation set.
def select_threshold_at_fpr(y_true, y_scores, target_fpr=0.01):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.5
    return thresholds[idx[-1]]

# Name: load_and_clean_binetflow
# Description: Loads a .binetflow file, cleans whitespace from column names, 
#              and filters out 'Background' traffic to ensure clean labels.
def load_and_clean_binetflow(DATA_DIR, filename):
    path = os.path.join(DATA_DIR, filename)
    #print(f"Loading {filename}...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    initial_count = len(df)
    df = df[df['Label'].str.contains('Botnet|Normal', case=False, na=False)].copy()
    
    #print(f"Filtered {initial_count - len(df)} background rows. {len(df)} rows remaining.")
    return df

# Name : process_labels
# Description: Processes the 'Label' column to create a binary label 'binary_label' 
#              where 1 indicates Botnet activity and 0 indicates Normal traffic.
def process_labels(df):
    df = df.copy()
    df['binary_label'] = df['Label'].apply(
        lambda x: 1 if any(s in str(x) for s in ['Botnet', 'C&C', 'Malware']) else 0
    )
    
    return df

# Name: feature_engineering
# Description: Takes a DataFrame and creates rate-based features
# (bytes per second and packets per second) using one-hot encoding from the 'Proto' column.
def feature_engineering(df):
    df = df.copy()
    
    # Existing rate-based features
    df['bytes_per_sec'] = df['TotBytes'] / (df['Dur'] + 1e-6)
    df['pkts_per_sec'] = df['TotPkts'] / (df['Dur'] + 1e-6)
    
    # NEW: Structural ratio features (highly effective across scenarios)
    df['bytes_per_pkt'] = df['TotBytes'] / (df['TotPkts'] + 1e-6)
    df['src_bytes_ratio'] = df['SrcBytes'] / (df['TotBytes'] + 1e-6)
    
    # Handle Categorical Text Data: 'Proto'
    if 'Proto' in df.columns:
        df['Proto'] = df['Proto'].apply(lambda x: x if x in ['tcp', 'udp', 'icmp'] else 'other')
        proto_dummies = pd.get_dummies(df['Proto'], prefix='proto', dtype=int)
        df = pd.concat([df, proto_dummies], axis=1)
        df = df.drop(columns=['Proto'])
        
    # NEW: Handle Categorical Text Data: 'State' 
    # (Be sure to remove 'State' from your DROP_COLUMNS list above this function!)
    if 'State' in df.columns:
        # Keep top 5 most common states, map the rest to 'other'
        top_states = df['State'].value_counts().nlargest(5).index
        df['State'] = df['State'].apply(lambda x: x if x in top_states else 'other')
        state_dummies = pd.get_dummies(df['State'], prefix='state', dtype=int)
        df = pd.concat([df, state_dummies], axis=1)
        df = df.drop(columns=['State'])
        
    return df

# Name: prepare_features
# Description: This function takes a DataFrame, drops leakage columns, and separates the target variable (y) from the feature matrix (X). 
def prepare_features(df, DROP_COLUMNS):
    df_cleaned = df.drop(columns=DROP_COLUMNS, errors='ignore')
    X = df_cleaned.drop(columns=['binary_label'], errors='ignore')
    y = df_cleaned['binary_label']
    return X, y

import torch.nn.functional as f

# Name: make_ae_models
# Description: Creates and returns both encoder and decoder PyTorch models for an autoencoder
def make_ae_models(feat_dims, hidden_dims, bottleneck_dims, activation=nn.GELU, dropout_ratio=0.1, device="cpu"):
    encoder_model = nn.Sequential(
        nn.Dropout(dropout_ratio),  # NEW: Denoising input layer
        nn.Linear(feat_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims), # NEW: Batch Normalization
        activation(),
        nn.Dropout(dropout_ratio),
        nn.Linear(hidden_dims, bottleneck_dims),
    ).to(device)

    decoder_model = nn.Sequential(
        nn.Linear(bottleneck_dims, hidden_dims),
        nn.BatchNorm1d(hidden_dims), # NEW: Batch Normalization
        activation(),
        nn.Dropout(dropout_ratio),
        nn.Linear(hidden_dims, feat_dims),
    ).to(device)

    return encoder_model, decoder_model

# Name: autoencoder_loss
# Description: Computes the Huber loss between the original features and the reconstructed features.
def autoencoder_loss(feats, reconstructs, bottlenecks=None, l2_reg_factor=None):
    # NEW: Swapped MSE for Huber Loss to handle extreme network traffic outliers
    huber_loss = f.huber_loss(reconstructs, feats, delta=1.0)
    
    if l2_reg_factor:
        l2_reg_loss = bottlenecks.square().sum(-1).mean()
        total_loss = huber_loss + l2_reg_factor * l2_reg_loss
    else:
        total_loss = huber_loss
        
    return total_loss, huber_loss

# Name: train_autoencoder
# Description: This function implements the training loop for the autoencoder. 
def train_autoencoder(encoder_model, decoder_model, optimizer, feats, n_epochs, batch_size, l2_reg_factor=0., device="cpu"):
    dataset = TensorDataset(feats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i in range(n_epochs):
        encoder_model.train()
        decoder_model.train()
        
        for (feats_batch,) in loader:
            feats_batch = feats_batch.to(device)

            # Forward pass
            bottlenecks = encoder_model(feats_batch)
            reconstructs = decoder_model(bottlenecks)
            
            # Compute loss
            total_loss_batch, step_loss_batch = autoencoder_loss(
                feats_batch, reconstructs, bottlenecks, l2_reg_factor
            )

            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()

# Name: evaluate_model_performance
# Description: General Test method for all scenarios
def evaluate_model_performance(
        scenario_name,  # The name of the scenario being evaluated (e.g., "Scenario 1")
        data,           # The raw .binetflow file for the scenario (e.g., "capture20110810.binetflow")
        X_train,        # The original training feature matrix (used to ensure column consistency)
        log_model,      # The trained Logistic Regression model
        rf_model,       # The trained Random Forest model
        xgb_model,      # The trained XGBoost model
        aut_enc_model,  # The trained Autoencoder Encoder model
        aut_dec_model,  # The trained Autoencoder Decoder model
        thresholds,      # The FPR thresholds to evaluate at
        scaler,          # The fitted scaler from the training data (used to transform new scenario features)
        DROP_COLUMNS,     # The list of columns to drop from the feature matrix
        device,           # The device (CPU or GPU) to run the autoencoder evaluation on
        return_results = False # Whether to return the recall values for plotting outside the function (used in multi-scenario evaluation)
        ):
    
    # Sections 1-4: Load, Process, and Prepare the new scenario
    scenario_raw = load_and_clean_binetflow("./", data)
    processed_df = process_labels(scenario_raw)
    engineered_df = feature_engineering(processed_df)
    X, y = prepare_features(engineered_df, DROP_COLUMNS=DROP_COLUMNS)
    
    # Ensure all columns are numeric before proceeding
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Handle missing columns and order
    for col in X_train.columns:
        if col not in X.columns:
            X[col] = 0  # Fill missing one-hot columns with 0
    X = X[X_train.columns]
    
    # Ensure all columns are numeric (catch any remaining non-numeric columns)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
   
    X_scaled = scaler.transform(X)

    print(f"--- Cross-Scenario Performance on {scenario_name} at Various FPR Thresholds ---")
    for thresh in thresholds:
        # Evaluate Logistic Regression
        log_probs = log_model.predict_proba(X_scaled)[:, 1]
        log_auc = roc_auc_score(y, log_probs)
        log_recall = recall_at_fixed_fpr(y, log_probs, target_fpr=thresh)
        # Evaluate Random Forest
        rf_probs = rf_model.predict_proba(X)[:, 1]
        rf_auc = roc_auc_score(y, rf_probs)
        rf_recall = recall_at_fixed_fpr(y, rf_probs, target_fpr=thresh)
        # Evaluate XGBoost
        xgb_probs = xgb_model.predict_proba(X)[:, 1]
        xgb_auc = roc_auc_score(y, xgb_probs)
        xgb_recall = recall_at_fixed_fpr(y, xgb_probs, target_fpr=thresh)
        # Evaluate Autoencoder
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            bottlenecks = aut_enc_model(X_tensor)
            reconstructs = aut_dec_model(bottlenecks)
            mse_scores = torch.mean((reconstructs - X_tensor)**2, dim=1).cpu().numpy()
        
        ae_auc = roc_auc_score(y, mse_scores)
        ae_recall = recall_at_fixed_fpr(y, mse_scores, target_fpr=thresh)
        print(f"FPR Threshold: {thresh:.3%} | Logistic Recall: {log_recall:.2%} | RF Recall: {rf_recall:.2%} | XGB Recall: {xgb_recall:.2%} | AE Recall: {ae_recall:.2%}")
    
    print(f"--- Summary for {scenario_name} ---")
    print(f"Logistic Regression ROC-AUC: {log_auc:.4f}")
    print(f"Random Forest ROC-AUC: {rf_auc:.4f}")
    print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
    print(f"Autoencoder ROC-AUC: {ae_auc:.4f}")
    print("\n")

    #plot the results for better visualization
    # return the the plot values so that they can be plotted outside the function
    results = {
        'thresholds': thresholds,
        'log_recall': [recall_at_fixed_fpr(y, log_probs, target_fpr=t) for t in thresholds],
        'rf_recall': [recall_at_fixed_fpr(y, rf_probs, target_fpr=t) for t in thresholds],
        'xgb_recall': [recall_at_fixed_fpr(y, xgb_probs, target_fpr=t) for t in thresholds],
        'ae_recall': [recall_at_fixed_fpr(y, mse_scores, target_fpr=t) for t in thresholds]
    }

    if return_results:
        return results

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, [recall_at_fixed_fpr(y, log_probs, target_fpr=t) for t in thresholds], marker='o', label='Logistic Regression')
    plt.plot(thresholds, [recall_at_fixed_fpr(y, rf_probs, target_fpr=t) for t in thresholds], marker='o', label='Random Forest')
    plt.plot(thresholds, [recall_at_fixed_fpr(y, xgb_probs, target_fpr=t) for t in thresholds], marker='o', label='XGBoost')
    plt.plot(thresholds, [recall_at_fixed_fpr(y, mse_scores, target_fpr=t) for t in thresholds], marker='o', label='Autoencoder')
    plt.xscale('log')
    plt.xlabel('False Positive Rate Threshold')
    plt.ylabel('Recall')
    plt.title('Recall at Various FPR Thresholds ({})'.format(scenario_name))
    plt.legend()
    plt.grid(True)
    plt.show()

# Name: evaluate_and_plot_multiple_scenarios
# Description: This function takes a list of scenarios and calls the evaluate_model_performance function for each scenario, printing them in a 1 by X plot, where x is the number of scenarios using the plot_Xnum_recall_curves method. 
def evaluate_and_plot_multiple_scenarios(
        scenarios,          # List of (scenario_name, filename)
        X_train,
        log_model,
        rf_model,
        xgb_model,
        aut_enc_model,
        aut_dec_model,
        thresholds,
        scaler,
        DROP_COLUMNS,
        device
    ):

    performance_dict = {}

    # Evaluate each scenario
    for scenario_name, data in scenarios:

        performance_dict[scenario_name] = evaluate_model_performance(
            scenario_name=scenario_name,
            data=data,
            X_train=X_train,
            log_model=log_model,
            rf_model=rf_model,
            xgb_model=xgb_model,
            aut_enc_model=aut_enc_model,
            aut_dec_model=aut_dec_model,
            thresholds=thresholds,
            scaler=scaler,
            DROP_COLUMNS=DROP_COLUMNS,
            device=device,
            return_results=True
        )

    # Plot in 2 x 2 (or dynamic grid)
    numplots = len(performance_dict)

    cols = 2
    rows = math.ceil(numplots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), sharey=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for ax, (scenario_name, perf) in zip(axes, performance_dict.items()):

        ax.plot(perf['thresholds'], perf['log_recall'], marker='o', label='Logistic Regression')
        ax.plot(perf['thresholds'], perf['rf_recall'], marker='o', label='Random Forest')
        ax.plot(perf['thresholds'], perf['xgb_recall'], marker='o', label='XGBoost')
        ax.plot(perf['thresholds'], perf['ae_recall'], marker='o', label='Autoencoder')

        ax.set_xscale('log')
        ax.set_title(scenario_name)
        ax.set_xlabel("False Positive Rate Threshold")
        ax.grid(True)

    # Hide unused subplots if any
    for i in range(len(performance_dict), len(axes)):
        axes[i].set_visible(False)

    axes[0].set_ylabel("Recall")
    axes[0].legend()

    plt.tight_layout()
    plt.show()

# Name: train_data
# Description: General Train method for all scenarios
def train_data (
        scenario_name, # The name of the scenario being evaluated (e.g., "Scenario 1")
        data,           # The raw .binetflow file for the scenario (e.g., "capture20110810.binetflow")
        DROP_COLUMNS,   # The list of columns to drop from the feature matrix (e.g., ['SrcAddr', 'DstAddr', 'Label'])
        RANDOM_STATE    # A fixed random state for reproducibility (e.g., 42)
):
    print(f"--- Training on {scenario_name} ---")
    # Sections 1-4: Load, Process, and Prepare the new scenario
    scenario_raw = load_and_clean_binetflow("./", data)
    processed_df = process_labels(scenario_raw)
    engineered_df = feature_engineering(processed_df)
    X, y = prepare_features(engineered_df, DROP_COLUMNS=DROP_COLUMNS)
    
    # Ensure all columns are numeric before proceeding
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Section 5: Train-test split (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Section 6: Scaling and Class Weights
    scaler = RobustScaler() 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, weights))

    # Section 7: Train Logistic Regression
    with timeit("Training logistic regression classifier"):
        log_model = LogisticRegression(
            class_weight=class_weight_dict,
            max_iter=1000,
            random_state=RANDOM_STATE
        ).fit(X_train_scaled, y_train)
    
    # Section 8: Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight_dict, 
        random_state=RANDOM_STATE,
        n_jobs=-1 
    ).fit(X_train, y_train)

    # Section 8.2: Train XGBoost
    neg_cases = sum(y_train == 0)
    pos_cases = sum(y_train == 1)
    scale_pos = neg_cases / pos_cases
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ).fit(X_train, y_train)

    # Section 9: Train Autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    X_train_normal = X_train_scaled[y_train == 0]
    train_tensor = torch.tensor(X_train_normal, dtype=torch.float32, device=device)
    input_dim = X_train_scaled.shape[1]
    enc_model, dec_model = make_ae_models(
        feat_dims=input_dim,
        hidden_dims=8,
        bottleneck_dims=3,
        activation=nn.GELU,
        dropout_ratio=0.1,
        device=device
    )
    optimizer = AdamW(chain(enc_model.parameters(), dec_model.parameters()), lr=0.01)
    with timeit("Training Enhanced Autoencoder"):
        train_autoencoder(
            enc_model, dec_model, optimizer, train_tensor, 
            n_epochs=50, batch_size=256, l2_reg_factor=0.001, device=device
        )
    
    print(f"--- Training on {scenario_name} complete ---\n")
    return X_train, log_model, rf_model, xgb_model, enc_model, dec_model, scaler
