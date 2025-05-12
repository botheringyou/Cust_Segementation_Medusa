import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("Customer Segmentation Analysis")

# Sidebar for file uploads and parameters
with st.sidebar:
    st.header("Upload Data")
    train_file = st.file_uploader("Upload Train CSV", type=["csv"])
    test_file = st.file_uploader("Upload Test CSV", type=["csv"])
    
    st.header("Model Parameters")
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=4)
    random_state = st.number_input("Random State", min_value=0, value=42)
    
    st.header("Options")
    save_model = st.checkbox("Save Model", value=True)
    show_raw_data = st.checkbox("Show Raw Data", value=False)

# Function to load and preprocess data
def load_data(file):
    if file is not None:
        try:
            df = pd.read_csv(file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

# Function to preprocess data
def preprocess_data(df, numeric_cols=None):
    if df is None:
        return None
    
    # If numeric columns not specified, use all numeric columns
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Standardize numeric columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, scaler

# Function to train model
def train_model(df, n_clusters, random_state):
    if df is None:
        return None
    
    # Use all numeric columns for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X = df[numeric_cols]
    
    # Train KMeans model
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)
    
    # Add cluster labels to dataframe
    df['Cluster'] = model.labels_
    
    return model, df

# Function to visualize clusters
def visualize_clusters(df, cluster_col='Cluster'):
    if df is None:
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in numeric_cols:
        numeric_cols.remove(cluster_col)
    
    # Select two most important features (for demo purposes)
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[:2]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=col1, y=col2, hue=cluster_col, palette='viridis', ax=ax)
        ax.set_title("Customer Clusters")
        st.pyplot(fig)
        
        # Cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = df.groupby(cluster_col)[numeric_cols].mean()
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        
        # Cluster sizes
        st.subheader("Cluster Sizes")
        cluster_sizes = df[cluster_col].value_counts().sort_index()
        st.bar_chart(cluster_sizes)
    else:
        st.warning("Not enough numeric columns for visualization")

# Main app logic
def main():
    # Load data
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    
    if train_df is not None:
        if show_raw_data:
            st.subheader("Raw Training Data")
            st.dataframe(train_df.head())
        
        # Preprocess data
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        train_df_processed, scaler = preprocess_data(train_df.copy(), numeric_cols)
        
        # Train model
        model, clustered_df = train_model(train_df_processed, n_clusters, random_state)
        
        if model is not None:
            st.success("Model trained successfully!")
            
            # Visualize clusters
            st.header("Training Data Clusters")
            visualize_clusters(clustered_df)
            
            # Save model if requested
            if save_model:
                model_data = {
                    'model': model,
                    'scaler': scaler,
                    'numeric_cols': numeric_cols
                }
                with open('customer_segmentation_model.pkl', 'wb') as f:
                    pickle.dump(model_data, f)
                st.success("Model saved as 'customer_segmentation_model.pkl'")
    
    # Test data analysis
    if test_df is not None:
        if show_raw_data:
            st.subheader("Raw Test Data")
            st.dataframe(test_df.head())
        
        if 'model' in locals():
            # Preprocess test data using same scaler
            test_df_processed = test_df.copy()
            test_df_processed[numeric_cols] = scaler.transform(test_df_processed[numeric_cols])
            
            # Predict clusters
            test_clusters = model.predict(test_df_processed[numeric_cols])
            test_df_processed['Cluster'] = test_clusters
            
            # Visualize test clusters
            st.header("Test Data Clusters")
            visualize_clusters(test_df_processed)
        else:
            st.warning("Please train a model first to analyze test data")

if __name__ == "__main__":
    main()
