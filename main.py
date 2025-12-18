import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import io

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Web Threat Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🛡️ Web Threat Analysis Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
    st.title("Navigation")
    page = st.radio("Select Page", [
        "📊 Data Overview",
        "🔍 Data Analysis",
        "📈 Visualizations",
        "🤖 ML Models",
        "📋 Reports"
    ])
    
    st.divider()
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'transformed_df' not in st.session_state:
    st.session_state.transformed_df = None

# Load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache_data
def preprocess_data(data):
    # Remove duplicates
    df_unique = data.drop_duplicates()
    
    # Convert time columns
    df_unique['creation_time'] = pd.to_datetime(df_unique['creation_time'])
    df_unique['end_time'] = pd.to_datetime(df_unique['end_time'])
    df_unique['time'] = pd.to_datetime(df_unique['time'])
    
    # Standardize country codes
    df_unique['src_ip_country_code'] = df_unique['src_ip_country_code'].str.upper()
    
    # Feature engineering
    df_unique['duration_seconds'] = (df_unique['end_time'] - df_unique['creation_time']).dt.total_seconds()
    
    # Scaling numerical features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_unique[['bytes_in', 'bytes_out', 'duration_seconds']])
    
    # Encoding categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df_unique[['src_ip_country_code']])
    
    # Create transformed dataframe
    scaled_columns = ['scaled_bytes_in', 'scaled_bytes_out', 'scaled_duration_seconds']
    encoded_columns = encoder.get_feature_names_out(['src_ip_country_code'])
    
    scaled_df = pd.DataFrame(scaled_features, columns=scaled_columns, index=df_unique.index)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=df_unique.index)
    
    transformed_df = pd.concat([df_unique, scaled_df, encoded_df], axis=1)
    transformed_df['is_suspicious'] = (transformed_df['detection_types'] == 'waf_rule').astype(int)
    
    return df_unique, transformed_df

# Load data if uploaded
if uploaded_file is not None:
    st.session_state.data = load_data(uploaded_file)
    df_unique, st.session_state.transformed_df = preprocess_data(st.session_state.data)

# Main content
if st.session_state.data is None:
    st.info("👆 Please upload a CSV file to begin analysis")
    st.markdown("""
    ### Expected CSV Format:
    - `creation_time`: Timestamp
    - `end_time`: Timestamp
    - `time`: Timestamp
    - `src_ip`: Source IP address
    - `dst_ip`: Destination IP address
    - `bytes_in`: Bytes received
    - `bytes_out`: Bytes sent
    - `src_ip_country_code`: Country code
    - `detection_types`: Type of detection
    """)
else:
    data = st.session_state.data
    transformed_df = st.session_state.transformed_df
    
    # Page: Data Overview
    if page == "📊 Data Overview":
        st.markdown('<p class="sub-header">Dataset Overview</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            st.metric("Unique IPs", data['src_ip'].nunique())
        with col3:
            st.metric("Countries", data['src_ip_country_code'].nunique())
        with col4:
            st.metric("Suspicious Records", transformed_df['is_suspicious'].sum())
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Dataset Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
        
        st.divider()
        
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Page: Data Analysis
    elif page == "🔍 Data Analysis":
        st.markdown('<p class="sub-header">Data Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detection Types Distribution")
            detection_counts = data['detection_types'].value_counts()
            st.bar_chart(detection_counts)
        
        with col2:
            st.subheader("Country Code Distribution")
            country_counts = data['src_ip_country_code'].value_counts().head(10)
            st.bar_chart(country_counts)
        
        st.divider()
        
        st.subheader("Top Source IPs")
        top_ips = data['src_ip'].value_counts().head(10)
        st.dataframe(top_ips.reset_index().rename(columns={'index': 'IP Address', 'src_ip': 'Count'}), 
                     use_container_width=True)
    
    # Page: Visualizations
    elif page == "📈 Visualizations":
        st.markdown('<p class="sub-header">Visualizations</p>', unsafe_allow_html=True)
        
        # Correlation Heatmap
        st.subheader("Correlation Matrix Heatmap")
        numeric_df = transformed_df.select_dtypes(include=['float64', 'int64'])
        correlation_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        st.pyplot(fig)
        
        st.divider()
        
        # Detection Types by Country
        st.subheader("Detection Types by Country Code")
        detection_types_by_country = pd.crosstab(
            transformed_df['src_ip_country_code'], 
            transformed_df['detection_types']
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        detection_types_by_country.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Detection Types by Country Code')
        ax.set_xlabel('Country Code')
        ax.set_ylabel('Frequency of Detection Types')
        plt.xticks(rotation=45)
        ax.legend(title='Detection Type')
        st.pyplot(fig)
        
        st.divider()
        
        # Traffic Over Time
        st.subheader("Web Traffic Analysis Over Time")
        temp_data = data.copy()
        temp_data['creation_time'] = pd.to_datetime(temp_data['creation_time'])
        temp_data = temp_data.set_index('creation_time')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(temp_data.index, temp_data['bytes_in'], label='Bytes In', marker='o', alpha=0.7)
        ax.plot(temp_data.index, temp_data['bytes_out'], label='Bytes Out', marker='o', alpha=0.7)
        ax.set_title('Web Traffic Analysis Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Bytes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.divider()
        
        # Network Graph
        st.subheader("Network Interaction Graph")
        if st.button("Generate Network Graph (may take time for large datasets)"):
            with st.spinner("Generating network graph..."):
                G = nx.Graph()
                sample_data = data.sample(min(100, len(data)))
                for idx, row in sample_data.iterrows():
                    G.add_edge(row['src_ip'], row['dst_ip'])
                
                fig, ax = plt.subplots(figsize=(14, 10))
                nx.draw_networkx(G, with_labels=True, node_size=20, font_size=6, 
                                node_color='skyblue', font_color='darkblue', ax=ax)
                ax.set_title('Network Interaction between Source and Destination IPs (Sample)')
                ax.axis('off')
                st.pyplot(fig)
    
    # Page: ML Models
    elif page == "🤖 ML Models":
        st.markdown('<p class="sub-header">Machine Learning Models</p>', unsafe_allow_html=True)
        
        model_type = st.selectbox("Select Model", [
            "Random Forest Classifier",
            "Neural Network (Basic)",
            "Neural Network (Advanced)",
            "Conv1D Neural Network"
        ])
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                
                # Prepare data
                X = transformed_df[['bytes_in', 'bytes_out', 'scaled_duration_seconds']].values
                y = transformed_df['is_suspicious'].values
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
                
                if model_type == "Random Forest Classifier":
                    # Random Forest
                    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_classifier.fit(X_train, y_train)
                    y_pred = rf_classifier.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    st.success(f"Model trained successfully!")
                    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
                    
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())
                    
                    # Feature Importance
                    st.subheader("Feature Importance")
                    importances = pd.DataFrame({
                        'Feature': ['bytes_in', 'bytes_out', 'scaled_duration_seconds'],
                        'Importance': rf_classifier.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    st.bar_chart(importances.set_index('Feature'))
                
                else:
                    # Neural Networks
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    if model_type == "Neural Network (Basic)":
                        model = Sequential([
                            Dense(8, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                            Dense(16, activation='relu'),
                            Dense(1, activation='sigmoid')
                        ])
                        epochs = 10
                        batch_size = 8
                    
                    elif model_type == "Neural Network (Advanced)":
                        model = Sequential([
                            Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                            Dropout(0.5),
                            Dense(128, activation='relu'),
                            Dropout(0.5),
                            Dense(1, activation='sigmoid')
                        ])
                        epochs = 10
                        batch_size = 32
                    
                    else:  # Conv1D
                        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
                        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
                        
                        model = Sequential([
                            Conv1D(32, kernel_size=1, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
                            Flatten(),
                            Dense(64, activation='relu'),
                            Dropout(0.5),
                            Dense(1, activation='sigmoid')
                        ])
                        epochs = 10
                        batch_size = 32
                    
                    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    history = model.fit(
                        X_train_scaled, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=0,
                        validation_split=0.2
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Training complete!")
                    
                    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                    
                    st.success(f"Model trained successfully!")
                    st.metric("Test Accuracy", f"{accuracy*100:.2f}%")
                    
                    # Plot training history
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(history.history['accuracy'], label='Training Accuracy')
                        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
                        ax.set_title('Model Accuracy')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Accuracy')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(history.history['loss'], label='Training Loss')
                        ax.plot(history.history['val_loss'], label='Validation Loss')
                        ax.set_title('Model Loss')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
    
    # Page: Reports
    elif page == "📋 Reports":
        st.markdown('<p class="sub-header">Analysis Reports</p>', unsafe_allow_html=True)
        
        st.subheader("Executive Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Threats Detected", transformed_df['is_suspicious'].sum())
        with col2:
            threat_rate = (transformed_df['is_suspicious'].sum() / len(transformed_df)) * 100
            st.metric("Threat Rate", f"{threat_rate:.2f}%")
        with col3:
            st.metric("Average Traffic (MB)", f"{data['bytes_in'].mean() / 1024 / 1024:.2f}")
        
        st.divider()
        
        st.subheader("Top Threat Sources")
        suspicious_data = transformed_df[transformed_df['is_suspicious'] == 1]
        top_threat_countries = suspicious_data['src_ip_country_code'].value_counts().head(10)
        st.bar_chart(top_threat_countries)
        
        st.divider()
        
        st.subheader("Download Report")
        csv = transformed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Analysis (CSV)",
            data=csv,
            file_name="web_threat_analysis_report.csv",
            mime="text/csv"
        )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <p>🛡️ Web Threat Analysis Dashboard | Built with Streamlit</p>
    <p>For security monitoring and threat detection</p>
</div>
""", unsafe_allow_html=True)
