import asyncio
import json
import os
import numpy as np
import joblib
import psycopg2
from psycopg2.extras import Json
import aio_pika

MODEL_PATH = "model.joblib"

# Load model once
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found")

model = load_model()


def get_db_connection():
    """Return a new database connection using environment variables."""
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        dbname=os.getenv("POSTGRES_DB", "mlops"),
        user=os.getenv("POSTGRES_USER", "mlops"),
        password=os.getenv("POSTGRES_PASSWORD", "mlops")
    )


async def process_message(message: aio_pika.IncomingMessage, result_exchange: aio_pika.Exchange):
    async with message.process():
        try:
            payload = json.loads(message.body)
            features = payload.get("features")
            if not features:
                print("Invalid message: no features field")
                return
            # Extract feature values
            X = np.array([[
                features["sepal_length"],
                features["sepal_width"],
                features["petal_length"],
                features["petal_width"]
            ]])
            prediction = int(model.predict(X)[0])
            # Save to database
            try:
                conn = get_db_connection()
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO predictions (features, prediction) VALUES (%s, %s)",
                        (Json(features), prediction)
                    )
                conn.close()
            except Exception as db_err:
                print(f"Database insert error: {db_err}")
            # Publish result message
            result_msg = {"prediction": prediction, "features": features}
            await result_exchange.publish(
                aio_pika.Message(body=json.dumps(result_msg).encode()),
                routing_key=""
            )
            print(f"Processed message, prediction={prediction}")
        except Exception as e:
            print(f"Error processing message: {e}")


async def main():
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@message_broker:5672/")
    connection = await aio_pika.connect_robust(rabbitmq_url)
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("prediction_requests", durable=True)
        result_exchange = await channel.declare_exchange(
            "prediction_results", aio_pika.ExchangeType.FANOUT, durable=True
        )
        print("Consumer started, waiting for messages...")
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_message(message, result_exchange)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Consumer stopped")
