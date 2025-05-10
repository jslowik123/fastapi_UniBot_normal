from celery import Celery
import os

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

celery = Celery("myapp", broker=redis_url, backend=redis_url)

@celery.task
def test_task():
    return "Worker is running!"
