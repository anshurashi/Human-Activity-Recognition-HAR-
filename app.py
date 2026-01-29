import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Human Activity Recognition", layout="wide")

# 2. Load Models
@st.cache_resource
def load_models():
    # Make sure these files are in the same folder
    model = joblib.load('har_svm_model.pkl')
    scaler = joblib.load('har_scaler.pkl')
    pca = joblib.load('har_pca.pkl')
    le = joblib.load('har_label_encoder.pkl')
    return model, scaler, pca, le

try:
    model, scaler, pca, le = load_models()
    st.sidebar.success("Models Loaded Successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: .pkl files not found. Run the training script first.")

# 3. App Title
st.title("🏃 Human Activity Recognition System")
st.write("This app classifies human activities (Walking, Sitting, etc.) using sensor data.")

# --- NEW SECTION: CHOOSE INPUT METHOD ---
st.subheader("Step 1: Get Data")
col1, col2 = st.columns(2)

input_data = None

with col1:
    # Option A: Upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")

with col2:
    # Option B: Use Demo Button
    st.write("Don't have a file?")
    if st.button("Use Example Data (Load test.csv)"):
        try:
            # Load the first 10 rows of your local test.csv
            input_data = pd.read_csv("test.csv").head(10)
            st.success("Loaded example data from test.csv")
        except FileNotFoundError:
            st.error("Could not find 'test.csv' in the folder.")

# 4. Process & Predict (Only runs if data is loaded)
if input_data is not None:
    st.subheader("Step 2: Raw Data")
    st.dataframe(input_data.head())

    # PREPROCESSING
    try:
        # Clean columns
        features = input_data.copy()
        if 'Activity' in features.columns:
            features = features.drop(columns=['Activity'])
        if 'subject' in features.columns:
            features = features.drop(columns=['subject'])
            
        # Scale & PCA
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Predict
        predictions_encoded = model.predict(features_pca)
        predictions_labels = le.inverse_transform(predictions_encoded)
        
        # Results
        st.subheader("Step 3: Predictions")
        results_df = features.copy()
        results_df.insert(0, 'Predicted_Activity', predictions_labels) # Put prediction first
        
        # Show specific columns to make it readable
        st.dataframe(results_df[['Predicted_Activity']].style.applymap(
            lambda x: 'background-color: #90ee90' if 'WALK' in x else 'background-color: #add8e6'
        ))
        
        # Download
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "predictions.csv", "text/csv")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")