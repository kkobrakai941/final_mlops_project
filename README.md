# final_mlops_project

## Overview

This repository contains a full MLOps demonstration project that takes a small machine‑learning model from experimentation to a production‑ready service.  The goal is to showcase how to build a **reliable, reproducible and observable ML system** using open‑source tools such as Docker, FastAPI, RabbitMQ, PostgreSQL, Airflow, Streamlit, Prometheus and Grafana.  Although the example model uses the Iris dataset for simplicity, the architecture is generic and can be adapted to real business problems like demand forecasting, anomaly detection or recommendation.

The project satisfies all mandatory requirements (business value, reproducibility, ML service, database, scheduler, broker, UI and monitoring) and includes advanced features such as batch/off‑line inference via a scheduler, model reloading, extended metrics and a simple analytics dashboard to target the 8–10 point evaluation.

## Components

| Component | Description |
|-----------|-------------|
| **ml_service** | A FastAPI app that loads a trained scikit‑learn model and exposes a `/predict` endpoint for online inference.  Every request and prediction is logged to PostgreSQL.  The service also exposes Prometheus metrics (`/metrics`) such as request count, latency, accuracy and precision.  A `/reload_model` endpoint allows hot‑reloading a new model without downtime. |
| **db** | A PostgreSQL database used to store input features, prediction results and metadata.  The `db/init.sql` script defines the `predictions` table. |
| **message broker** | RabbitMQ is used for asynchronous processing.  A consumer script (`ml_service/consumer.py`) listens on the `prediction_requests` queue, performs inference using the same model and writes results back to a `prediction_results` exchange. |
| **scheduler** | An Airflow container orchestrates ETL tasks and batch predictions.  The DAG in `scheduler/dags/predict_offline.py` periodically generates random Iris‑like feature vectors and calls the ML service API, simulating offline/batch inference and exercising the entire pipeline. |
| **ui** | A Streamlit application (`ui/app.py`) that allows a human user to enter feature values, call the prediction API and view the returned class.  It also displays a history of predictions from the database and a simple bar chart of class frequencies.  There is a button to trigger model reloading. |
| **monitoring** | Prometheus scrapes metrics from the ML service according to `monitoring/prometheus.yml`.  A Grafana instance (included in `docker‑compose.yml`) is pre‑configured to connect to Prometheus; you can create dashboards to observe request throughput, latency, accuracy and class distribution. |
| **docker‑compose** | The `docker‑compose.yml` file defines all services—PostgreSQL, RabbitMQ, ML service, UI, scheduler, Prometheus and Grafana—and brings them up with a single command. |

## Architecture

```
User (UI / Batch)  --->  ML Service  --->  Database
         |                ^                 |
         |                |                 |
         v                |                 v
   RabbitMQ (asynchronous queue)     Monitoring (Prometheus/Grafana)
         ^
         |
    Scheduler (Airflow)
```

Requests may arrive either from the Streamlit UI (synchronous) or via RabbitMQ (asynchronous).  All predictions are stored in PostgreSQL.  Prometheus scrapes metrics from the ML service; Grafana dashboards visualise them.  Airflow periodically generates synthetic data and sends it through the inference API.

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and `docker-compose` installed locally.
- At least 4 GB of RAM free to run the containers.

### Clone and run the project

```bash
git clone https://github.com/kkobrakai941/final_mlops_project.git
cd final_mlops_project
# Build and start all services
docker-compose up --build
```

On first start, the database will be initialised automatically.  Docker Compose will launch:

- **PostgreSQL** on port `5432`
- **RabbitMQ** on port `5672` (and management UI on `15672`)
- **ML service** on `http://localhost:8000`
- **Streamlit UI** on `http://localhost:8501`
- **Airflow** on `http://localhost:8080` (default login: `airflow` / `airflow`)
- **Prometheus** on `http://localhost:9090`
- **Grafana** on `http://localhost:3000` (default login: `admin` / `admin`)

### Using the API

Send a POST request to `/predict` with a JSON payload containing a `features` array of four floats:

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

The response will include the predicted class and a timestamp.  All requests and predictions are saved to the database.

### Using the Streamlit UI

Navigate to `http://localhost:8501` to open the UI.  Enter feature values and click **Predict** to see the result.  Use the **History & Analytics** section to view a table of recent predictions and a bar chart of class distributions.  The **Model Management** section provides a button to reload the model from disk after re‑training.

### Batch / Offline Predictions

Airflow is configured with a DAG (`predict_offline.py`) that every 30 minutes generates random feature values and calls the ML service.  This simulates offline batch predictions and helps monitor the health of the service.  You can access the Airflow UI at `http://localhost:8080` (login: `airflow` / `airflow`) to trigger the DAG manually or inspect logs.

### Monitoring

Prometheus scrapes the ML service on `/metrics` every 15 seconds.  After starting the stack you can browse metrics at `http://localhost:9090`.  Grafana is available at `http://localhost:3000` with a Prometheus datasource configured.  Create dashboards to visualise:

- Total requests and error rate
- Request latency histogram
- Class distribution of predictions
- Model accuracy/precision over time

### Reloading the Model

To update the model without downtime:

1. Train a new model offline (e.g. modify `ml_service/main.py` or add a training script).
2. Replace or update `ml_service/model.joblib` in the container image.
3. Call the `/reload_model` endpoint:
   ```bash
   curl -X POST http://localhost:8000/reload_model
   ```
   The service will reload `model.joblib` and update metrics accordingly.

## Directory Structure

```
.
├── db/                  # Database initialisation scripts
│   └── init.sql
├── ml_service/          # ML inference service code
│   ├── main.py
│   ├── consumer.py
│   ├── requirements.txt
│   └── Dockerfile
├── ui/                  # Streamlit application
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
├── scheduler/           # Airflow DAGs
│   └── dags/
│       └── predict_offline.py
├── monitoring/          # Prometheus configuration
│   └── prometheus.yml
├── one_pager.md         # Business one‑pager summarising the project
├── docker-compose.yml   # Orchestration of all services
└── README.md            # Project documentation (this file)
```

## Development Notes

- The current model is a simple logistic regression trained on the Iris dataset and stored as `model.joblib`.  You can train a new model using scikit‑learn and replace this file.
- The consumer service demonstrates how to perform inference off the API path; in a real application it might enrich events, join features and publish results to downstream systems.
- Airflow DAGs can be extended to perform feature engineering, model retraining or ETL jobs.  Additional DAGs can be added under `scheduler/dags`.
- Grafana dashboards are not pre‑provisioned; create them via the UI to visualise the metrics that matter for your business.

---

This README provides a comprehensive guide to running and extending the final MLOps project.
