from celery import Celery
import os



redis_url = "rediss://:p277c32da1640e7109e3e7fd52cd4c73d5b85ac54e4972db72473a2fa0b02742d@ec2-54-171-238-172.eu-west-1.compute.amazonaws.com:31030"

celery = Celery("myapp", broker=redis_url, backend=redis_url)
celery.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    broker_use_ssl={
        'ssl_cert_reqs': None
    },
    redis_backend_use_ssl={
        'ssl_cert_reqs': None
    }
)

@celery.task
def test_task():
    return "Worker is running!"
