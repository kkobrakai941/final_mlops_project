import streamlit as st
import requests
import pandas as pd
import numpy as np
import psycopg2

API_URL = "http://ml_service:8000"

DB_HOST = "db"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "postgres"

st.title("Iris Species Prediction Service")
st.markdown("This UI allows you to interact with the deployed ML model, submit feature values for prediction, and explore recent prediction history and metrics.")

# Sidebar for navigation
mode = st.sidebar.radio("Select view", ["Predict", "History & Analytics", "Model Management"])

# Helper function to connect to database
def get_db_connection():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)

# Predict mode
if mode == "Predict":
    st.header("Make a prediction")
    # Input fields for iris features
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1)
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2)

    if st.button("Predict species"):
        payload = {"features": [sepal_length, sepal_width, petal_length, petal_width]}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.success(f"Prediction: **{result['prediction']}** (class {result['predicted_class']})")
            st.write(f"Probabilities: {result['probabilities']}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# History and analytics mode
elif mode == "History & Analytics":
    st.header("Prediction History & Analytics")
    # Connect to DB and fetch last 200 predictions
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT id, timestamp, features, prediction FROM predictions ORDER BY timestamp DESC LIMIT 200", conn)
        conn.close()
        if df.empty:
            st.info("No prediction history yet.")
        else:
            # Convert JSONB features to separate columns
            feature_df = df["features"].apply(pd.Series)
            feature_df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
            combined_df = pd.concat([df[["id", "timestamp", "prediction"]], feature_df], axis=1)
            st.subheader("Recent Predictions")
            st.dataframe(combined_df)
            # Analytics: count predictions per class
            st.subheader("Prediction Counts by Class")
            counts = combined_df["prediction"].value_counts().reset_index()
            counts.columns = ["class", "count"]
            st.bar_chart(counts.set_index("class"))
    except Exception as e:
        st.error(f"Failed to load history: {e}")

# Model management mode
else:
    st.header("Model Management")
    st.write("Reload the model if it has been retrained and the file has changed.")
    if st.button("Reload model"):
        try:
            response = requests.post(f"{API_URL}/reload_model", timeout=10)
            response.raise_for_status()
            st.success("Model reloaded successfully.")
        except Exception as e:
            st.error(f"Failed to reload model: {e}")

    st.write("To update the model weights, train and replace the model file in the ml_service container. This page allows you to trigger the service to reload the new model without downtime.")
