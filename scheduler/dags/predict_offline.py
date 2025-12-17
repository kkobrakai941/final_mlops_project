"""
Airflow DAG that periodically generates random Iris dataset samples and
sends them to the inference API endpoint. This simulates offline batch
prediction jobs and allows monitoring of end-to-end system health.
"""

from datetime import datetime, timedelta
import json
import random
import urllib.request
import urllib.error

from airflow import DAG
from airflow.operators.python import PythonOperator

API_URL = "http://ml_service:8000/predict"

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def generate_and_predict(**context):
    """Generate random Iris-like features and send to the ML service for prediction."""
    # Iris feature ranges based on the dataset (approximate)
    sepal_length = random.uniform(4.0, 7.5)
    sepal_width = random.uniform(2.0, 4.5)
    petal_length = random.uniform(1.0, 6.5)
    petal_width = random.uniform(0.1, 2.5)
    payload = {
        "features": [sepal_length, sepal_width, petal_length, petal_width]
    }
    try:
        # Prepare data for HTTP POST
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
        # Log the prediction result to Airflow task log
        print(f"Sent features {payload['features']}, received prediction: {result}")
    except urllib.error.URLError as e:
        print(f"Error sending prediction request: {e}")
        raise

# Define the DAG
with DAG(
    dag_id="offline_prediction_dag",
    default_args=default_args,
    description="Periodically send random data to the prediction API",
    schedule_interval=timedelta(minutes=30),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["mlops", "inference", "batch"],
) as dag:

    run_offline_prediction = PythonOperator(
        task_id="run_offline_prediction",
        python_callable=generate_and_predict,
        provide_context=True,
    )
