# One Pager – MLOps Project

## Business Problem
In this project we simulate a SaaS company that needs to offer a simple machine‑learning powered service to end users. The goal is to provide a scalable, reliable and reproducible ML service that can take user input, run a model, return predictions and collect usage metrics. For demonstration we use the classic Iris dataset as a stand‑in for a real business problem (e.g. predicting customer churn, classifying text or ranking products). The same architecture can be adapted to any tabular/classification problem.

## Proposed Solution
We built an end‑to‑end MLOps system composed of loosely coupled services running in Docker containers and orchestrated by `docker‑compose`. The solution includes:

- **ML Service** – a FastAPI application exposing a `/predict` endpoint that loads a trained scikit‑learn model and returns predictions. The service stores each request and result in a PostgreSQL database and exposes Prometheus metrics (requests/sec, latency, accuracy, precision) for monitoring. A `/reload_model` endpoint allows hot‑reloading new models without downtime.
- **Database** – a PostgreSQL instance that stores input features, predictions and metadata for analytics. Tables are initialised via `db/init.sql`.
- **Message Broker** – RabbitMQ handles asynchronous jobs. A consumer listens on a queue, performs inference using the model and publishes prediction results back to another exchange. This decouples producers from the ML service and enables batch/off‑line processing.
- **Scheduler (Airflow)** – Airflow runs ETL and batch inference. A DAG periodically generates synthetic Iris‑like data and sends it to the inference API, simulating offline prediction and testing end‑to‑end health.
- **UI** – a Streamlit web application where users can input feature values, call the inference API, view recent predictions and basic analytics (class distribution bar chart, request history). It also exposes a button to reload the model.
- **Monitoring** – Prometheus scrapes metrics from the ML service. Grafana dashboards (preconfigured) show request counts, response times, class distribution, accuracy/precision and system health.

## Architecture Diagram
A high‑level architecture:

```
User/Batch  -->  UI or Message Broker  -->  ML Service (FastAPI + scikit‑learn)  -->  Database
                                ^                                        |
                                |                                        v
                          Scheduler (Airflow)                       Prometheus/Grafana

All components run inside Docker containers using a single `docker‑compose` command. The infrastructure is reproducible and can be deployed locally or in the cloud.
```

## Business Value
The system demonstrates how to turn a machine‑learning model into a production service with full MLOps capabilities:

- **Reliability & Reproducibility** – containerisation ensures that the service works the same on any machine. Database migrations and data ingestion are scripted.
- **Scalability** – decoupled services (API, broker, database) allow horizontal scaling and replacement of components.
- **Monitoring & Observability** – business and technical metrics provide insights into model quality and system performance. Alerts can be added for degraded accuracy or anomalies.
- **Model Management** – support for model reloading and offline batch predictions enables continuous improvement without downtime.
- **Extensibility** – although the demo uses the Iris dataset, the architecture supports replacing the model and feature schema to solve real business problems (demand forecasting, anomaly detection, recommendation, etc.).

This one‑pager summarises the purpose, components and value of the final MLOps project.
