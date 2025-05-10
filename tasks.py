from celery_app import celery
import time
from redis import Redis
import os

r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

@celery.task(name="tasks.process_file")
def process_file(job_id: str):
    r.set(f"job:{job_id}", "started")
    time.sleep(2)  # Simuliere Arbeit
    r.set(f"job:{job_id}", "50%")
    time.sleep(2)
    r.set(f"job:{job_id}", "done")
