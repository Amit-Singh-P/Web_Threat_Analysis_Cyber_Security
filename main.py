import streamlit as st
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Web Threat Analysis", layout="wide")

# Title
st.title("🔒 Web Threat Analysis Dashboard")

# ============= PASTE YOUR CSV FILE PATH HERE =============
CSV_FILE_PATH = "CloudWatch_Traffic_Web_Attack.csv"  # <- CHANGE THIS PATH
# =========================================================

@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and preprocess the data"""
    data = pd.read_csv(file_path)
    
    # Remove duplicate rows
    df_unique = data.drop_duplicates()
    
    # Convert time-related columns to datetime format
    df_unique['creation_time'] = pd.to_datetime(df_unique['creation_time'])
    df_unique['end_time'] = pd.to_datetime(df_unique['end_time'])
    df_unique['time'] = pd.to_datetime(df_unique['time'])
    
    # Standardize text data
    df_unique['src_ip_country_code'] = df_unique['src_ip_country_code'].str.upper()
    
    # Feature engineering: Calculate duration of connection
    df_unique['duration_seconds'] = (df_unique['end_time'] - df_unique['creation_time']).dt.total_seconds()
    
    # Preparing column transformations
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_unique[['bytes_in', 'bytes_out', 'duration_seconds']])
    
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df_unique[['src_ip_country_code']])
    
    # Combining transformed features
    scaled_columns = ['scaled_bytes_in', 'scaled_bytes_out', 'scaled_duration_seconds']
    encoded_columns = encoder.get_feature_names_out(['src_ip_country_code'])
    
    scaled_df = pd.DataFrame(scaled_features, columns=scaled_columns, index=df_unique.index)
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns, index=df_unique.index)
    
    transformed_df = pd.concat([df_unique, scaled_df, encoded_df], axis=1)
    
    return data, df_unique, transformed_df

# Load data
try:
    data, df_unique, transformed_df = load_and_preprocess_data(CSV_FILE_PATH)
    st.success("✅ Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# Sidebar
st.sidebar.header("📊 Analysis Options")
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Data Overview", "Correlation Analysis", "Detection Types Analysis", 
     "Traffic Over Time", "Network Graph", "Machine Learning Models"]
)

# Main content based on selection
if analysis_type == "Data Overview":
    st.header("📋 Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Unique Records", len(df_unique))
    with col3:
        st.metric("Detection Types", data['detection_types'].nunique())
    
    st.subheader("Dataset Sample")
    st.dataframe(df_unique.head(10))
    
    st.subheader("Dataset Information")
    buffer = df_unique.describe()
    st.dataframe(buffer)

elif analysis_type == "Correlation Analysis":
    st.header("🔗 Correlation Analysis")
    
    numeric_df = transformed_df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix Heatmap')
    st.pyplot(fig)

elif analysis_type == "Detection Types Analysis":
    st.header("🎯 Detection Types by Country")
    
    detection_types_by_country = pd.crosstab(
        transformed_df['src_ip_country_code'], 
        transformed_df['detection_types']
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    detection_types_by_country.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Detection Types by Country Code')
    ax.set_xlabel('Country Code')
    ax.set_ylabel('Frequency of Detection Types')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Detection Type')
    st.pyplot(fig)

elif analysis_type == "Traffic Over Time":
    st.header("📈 Web Traffic Analysis Over Time")
    
    data_time = data.copy()
    data_time['creation_time'] = pd.to_datetime(data_time['creation_time'])
    data_time.set_index('creation_time', inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_time.index, data_time['bytes_in'], label='Bytes In', marker='o')
    ax.plot(data_time.index, data_time['bytes_out'], label='Bytes Out', marker='o')
    ax.set_title('Web Traffic Analysis Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Bytes')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

elif analysis_type == "Network Graph":
    st.header("🌐 Network Interaction Graph")
    
    sample_size = st.sidebar.slider("Sample Size (for performance)", 10, min(500, len(data)), 100)
    
    G = nx.Graph()
    data_sample = data.head(sample_size)
    
    for idx, row in data_sample.iterrows():
        G.add_edge(row['src_ip'], row['dst_ip'])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx(G, with_labels=True, node_size=20, font_size=8, 
                     node_color='skyblue', font_color='darkblue', ax=ax)
    ax.set_title('Network Interaction between Source and Destination IPs')
    ax.axis('off')
    st.pyplot(fig)

elif analysis_type == "Machine Learning Models":
    st.header("🤖 Machine Learning Models")
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        ["Random Forest", "Neural Network (Simple)", "Neural Network (Advanced)", "CNN Model"]
    )
    
    # Prepare data
    transformed_df['is_suspicious'] = (transformed_df['detection_types'] == 'waf_rule').astype(int)
    
    if model_choice == "Random Forest":
        st.subheader("🌲 Random Forest Classifier")
        
        X = transformed_df[['bytes_in', 'bytes_out', 'scaled_duration_seconds']]
        y = transformed_df['is_suspicious']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        with st.spinner("Training Random Forest..."):
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train, y_train)
            y_pred = rf_classifier.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            classification = classification_report(y_test, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{accuracy:.2%}")
            
            st.text("Classification Report:")
            st.text(classification)
    
    else:  # Neural Network models
        data_ml = data.copy()
        data_ml['is_suspicious'] = (data_ml['detection_types'] == 'waf_rule').astype(int)
        
        X = data_ml[['bytes_in', 'bytes_out']].values
        y = data_ml['is_suspicious'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_choice == "Neural Network (Simple)":
            st.subheader("🧠 Simple Neural Network")
            
            model = Sequential([
                Dense(8, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            
            with st.spinner("Training model..."):
                history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=8, verbose=0)
                loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                
                st.metric("Test Accuracy", f"{accuracy*100:.2f}%")
        
        elif model_choice == "Neural Network (Advanced)":
            st.subheader("🧠 Advanced Neural Network with Dropout")
            
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            
            with st.spinner("Training model..."):
                history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, 
                                  verbose=0, validation_split=0.2)
                loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                
                st.metric("Test Accuracy", f"{accuracy*100:.2f}%")
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history.history['accuracy'], label='Training Accuracy')
                ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                
                ax2.plot(history.history['loss'], label='Training Loss')
                ax2.plot(history.history['val_loss'], label='Validation Loss')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                
                st.pyplot(fig)
        
        elif model_choice == "CNN Model":
            st.subheader("🔷 CNN Model")
            
            X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
            X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
            
            model = Sequential([
                Conv1D(32, kernel_size=1, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
                Flatten(),
                Dense(64, activation='relu'),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
            
            with st.spinner("Training CNN model..."):
                history = model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, 
                                  verbose=0, validation_split=0.2)
                loss, accuracy = model.evaluate(X_test_cnn, y_test, verbose=0)
                
                st.metric("Test Accuracy", f"{accuracy*100:.2f}%")
                
                # Plot training history
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history.history['accuracy'], label='Training Accuracy')
                ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                
                ax2.plot(history.history['loss'], label='Training Loss')
                ax2.plot(history.history['val_loss'], label='Validation Loss')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                
                st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("💡 Web Threat Analysis Dashboard - Analyzing CloudWatch Traffic Data")
