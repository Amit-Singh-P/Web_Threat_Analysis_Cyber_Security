# streamlit_app.py
import warnings
warnings.filterwarnings("ignore")

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Try to import tensorflow, but keep app functional if it's not installed
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Web Threat Analysis", layout="wide")
st.title("Web Threat Analysis")

# ========== CONFIG: change DATA_PATH to point to your CSV inside the repo ==========
# Example: put your file in a folder named `data` at repo root:
DATA_PATH = "CloudWatch_Traffic_Web_Attack.csv"
# ==================================================================================

# ---- Sidebar: Settings ----
st.sidebar.header("Settings")
max_graph_nodes = st.sidebar.slider("Max nodes in network graph", min_value=20, max_value=300, value=80, step=10)
train_nn = st.sidebar.checkbox("Also train TensorFlow Neural Network (if available)", value=False)
random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42)

if train_nn and not TF_AVAILABLE:
    st.sidebar.warning("TensorFlow not available on the server. Neural network option will be disabled.")

# Utility: safe datetime conversion
def safe_to_datetime(df, col):
    if col in df.columns:
        return pd.to_datetime(df[col], errors='coerce')
    return pd.Series([], dtype="datetime64[ns]")

@st.cache_data(show_spinner=False)
def load_and_transform_from_path(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)

    # Datetime conversions (safely)
    df['creation_time'] = safe_to_datetime(df, 'creation_time')
    df['end_time'] = safe_to_datetime(df, 'end_time')
    df['time'] = safe_to_datetime(df, 'time')

    # Normalize text columns if present
    if 'src_ip_country_code' in df.columns:
        df['src_ip_country_code'] = df['src_ip_country_code'].astype(str).str.upper().replace('NONE', np.nan)

    # Feature: duration
    if 'creation_time' in df.columns and 'end_time' in df.columns:
        df['duration_seconds'] = (df['end_time'] - df['creation_time']).dt.total_seconds()
    else:
        df['duration_seconds'] = np.nan

    # Fill numeric NaNs for scaler stability
    for c in ['bytes_in', 'bytes_out', 'duration_seconds']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        else:
            df[c] = 0

    # One-hot encode country (keep top 10 else "OTHER")
    if 'src_ip_country_code' in df.columns:
        top_countries = df['src_ip_country_code'].value_counts().nlargest(10).index
        df['src_ip_country_code_trunc'] = df['src_ip_country_code'].where(df['src_ip_country_code'].isin(top_countries), other='OTHER')
    else:
        df['src_ip_country_code_trunc'] = 'UNKNOWN'

    # Label: is_suspicious (binary)
    if 'detection_types' in df.columns:
        df['is_suspicious'] = (df['detection_types'].astype(str).str.lower() == 'waf_rule').astype(int)
    else:
        df['is_suspicious'] = 0

    return df

# Try to load from repo path
if os.path.exists(DATA_PATH):
    with st.spinner(f"Loading data from {DATA_PATH} ..."):
        df = load_and_transform_from_path(DATA_PATH)
else:
    st.error(f"CSV file not found at repo path: `{DATA_PATH}`. Please add your CSV to that path in the repository.")
    st.info("For testing you can still upload a CSV below (fallback).")
    uploaded_file = st.file_uploader("Upload CSV file (fallback)", type=["csv"])
    if not uploaded_file:
        st.stop()
    else:
        with st.spinner("Loading uploaded CSV..."):
            df = load_and_transform_from_path(uploaded_file)

# Show basic preview
st.header("Raw data preview")
st.dataframe(df.head(200))

# ---- Exploratory plots ----
st.header("Exploratory Visualizations")

# Correlation heatmap (matplotlib; numeric fields only)
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr = numeric_df.corr()
    st.subheader("Correlation Matrix (numeric columns)")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, aspect='auto', interpolation='none')
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)
else:
    st.info("No numeric columns available for correlation matrix.")

# Stacked bar chart: detection_types by country (truncated)
if 'src_ip_country_code_trunc' in df.columns and 'detection_types' in df.columns:
    st.subheader("Detection Types by Country (top countries + OTHER)")
    ctab = pd.crosstab(df['src_ip_country_code_trunc'], df['detection_types'])
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ctab.plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_xlabel('Country Code (truncated)')
    ax2.set_ylabel('Count')
    ax2.set_title('Detection Types by Country')
    ax2.legend(title='Detection Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    st.pyplot(fig2)

# Time-series: bytes_in and bytes_out over time if time index exists
if 'creation_time' in df.columns and not df['creation_time'].isna().all():
    st.subheader("Time Series: Bytes In / Bytes Out")
    ts_df = df.set_index('creation_time').sort_index()
    if ts_df.shape[0] > 2000:
        ts_df = ts_df.resample('1T').median().ffill()
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(ts_df.index, ts_df['bytes_in'], label='Bytes In', marker='o', markersize=3, linewidth=1)
    ax3.plot(ts_df.index, ts_df['bytes_out'], label='Bytes Out', marker='o', markersize=3, linewidth=1)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Bytes")
    ax3.set_title("Web Traffic Over Time")
    ax3.legend()
    ax3.grid(True)
    fig3.autofmt_xdate()
    st.pyplot(fig3)
else:
    st.info("No valid 'creation_time' column present for time series plot.")

# ---- Network graph (trimmed) ----
st.header("Network Interaction Graph (source -> destination)")

if 'src_ip' in df.columns and 'dst_ip' in df.columns:
    combined = pd.concat([df['src_ip'], df['dst_ip']])
    top_ips = combined.value_counts().nlargest(max_graph_nodes).index
    G = nx.Graph()
    for _, row in df.iterrows():
        s = row.get('src_ip')
        d = row.get('dst_ip')
        if s in top_ips and d in top_ips:
            G.add_edge(s, d)

    if G.number_of_nodes() == 0:
        st.info("No network edges among the top IPs for the selected max node limit.")
    else:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.15, iterations=20, seed=random_state)
        nx.draw_networkx_nodes(G, pos, node_size=50, ax=ax4)
        nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax4)
        nx.draw_networkx_labels(G, pos, font_size=7, ax=ax4)
        ax4.set_title("Trimmed Network Graph (top IPs)")
        ax4.axis('off')
        st.pyplot(fig4)
else:
    st.info("Missing 'src_ip' or 'dst_ip' columns for network graph.")

# ---- Modeling ----
st.header("Modeling — Random Forest (and optional Neural Net)")

required = ['bytes_in', 'bytes_out', 'duration_seconds', 'is_suspicious']
if all(col in df.columns for col in required):
    X = df[['bytes_in', 'bytes_out', 'duration_seconds']].values
    y = df['is_suspicious'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    with st.spinner("Training Random Forest..."):
        rf.fit(X_train_scaled, y_train)
        y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.subheader("Random Forest Results")
    st.write(f"Accuracy: **{acc:.4f}**")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    fi = rf.feature_importances_
    fig5, ax5 = plt.subplots(figsize=(6, 3))
    ax5.bar(['bytes_in', 'bytes_out', 'duration_seconds'], fi)
    ax5.set_title("Feature Importances (Random Forest)")
    st.pyplot(fig5)

    if train_nn and TF_AVAILABLE:
        st.subheader("Keras Dense Neural Network")
        n_features = X_train_scaled.shape[1]
        model = Sequential([
            Dense(32, activation='relu', input_shape=(n_features,)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        with st.spinner("Training neural network (this may take several seconds)..."):
            history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
            loss, acc_nn = model.evaluate(X_test_scaled, y_test, verbose=0)

        st.write(f"NN Test Accuracy: **{acc_nn:.4f}**")

        fig6, ax6 = plt.subplots(1, 2, figsize=(12, 4))
        ax6[0].plot(history.history['accuracy'], label='train')
        ax6[0].plot(history.history['val_accuracy'], label='val')
        ax6[0].set_title("NN Accuracy")
        ax6[0].legend()

        ax6[1].plot(history.history['loss'], label='train')
        ax6[1].plot(history.history['val_loss'], label='val')
        ax6[1].set_title("NN Loss")
        ax6[1].legend()
        st.pyplot(fig6)

    if train_nn and not TF_AVAILABLE:
        st.warning("TensorFlow is not installed in this environment. Add 'tensorflow' to requirements.txt if you want NN training (note: it's large).")

else:
    st.warning(f"Required columns for modeling not found. Need: {required}")

st.write("---")
st.markdown("App reads CSV from repo path. If you want the app to load a different path, change the `DATA_PATH` variable at the top of this file.")
