from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os
import psycopg2
from psycopg2.extras import Json
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import logging
from sklearn.metrics import precision_score

# Initialize FastAPI app
app = FastAPI(title="ML Inference Service", version="1.0")

MODEL_PATH = "model.joblib"

# Prometheus metrics
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_TIME = Histogram("prediction_time_seconds", "Time spent making predictions")
MODEL_ACCURACY = Gauge("model_accuracy", "Accuracy of the model on holdout set")
MODEL_PRECISION = Gauge("model_precision", "Precision of the model on holdout set")
DB_INSERT_ERRORS = Counter("db_insert_errors_total", "Number of errors inserting into DB")

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", filename="ml_service.log")

# Pydantic model for input features
class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


def get_db_connection():
    """Create a new database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops"),
        password=os.getenv("POSTGRES_PASSWORD", "mlops")
    )


def train_and_save():
    """
    Train a logistic regression model on the Iris dataset and save it to disk.
    Also update accuracy and precision Prometheus gauges.
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    # Evaluate metrics
    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, model.predict(X_test), average='weighted')
    MODEL_ACCURACY.set(accuracy)
    MODEL_PRECISION.set(precision)
    # Persist model
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Trained model saved to {MODEL_PATH} with accuracy={accuracy}, precision={precision}")
    return model


def load_model():
    """Load the model from disk or train a new one if it doesn't exist."""
    if os.path.exists(MODEL_PATH):
        logging.info("Loading existing model from disk")
        return joblib.load(MODEL_PATH)
    logging.info("Model file not found; training new model")
    return train_and_save()

# Load model at startup
model = load_model()

@app.post("/predict")
def predict(features: Features):
    """Make a prediction given iris features and store the result in the database."""
    PREDICTION_REQUESTS.inc()
    start = time.time()
    try:
        X = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        pred = model.predict(X)[0]
        duration = time.time() - start
        PREDICTION_TIME.observe(duration)
        # Save prediction to database
        try:
            conn = get_db_connection()
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO predictions (features, prediction) VALUES (%s, %s)",
                    (Json(features.dict()), int(pred))
                )
            conn.close()
        except Exception as db_err:
            DB_INSERT_ERRORS.inc()
            logging.error(f"DB insert error: {db_err}")
        return {"prediction": int(pred)}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/reload_model")
def reload_model():
    """Reload the ML model from disk without restarting the service."""
    global model
    model = load_model()
    return {"status": "model reloaded"}
