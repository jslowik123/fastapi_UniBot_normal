from celery import Celery
import os
import ssl



redis_url = "rediss://:p277c32da1640e7109e3e7fd52cd4c73d5b85ac54e4972db72473a2fa0b02742d@ec2-54-171-238-172.eu-west-1.compute.amazonaws.com:31030"

celery = Celery("myapp",
                broker=redis_url,
                backend=redis_url,
                include=['tasks'])

celery.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    broker_use_ssl={
        'ssl_cert_reqs': ssl.CERT_NONE,
        'ssl_ca_certs': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_check_hostname': False
    },
    redis_backend_use_ssl={
        'ssl_cert_reqs': ssl.CERT_NONE,
        'ssl_ca_certs': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_check_hostname': False
    },
    broker_connection_retry_on_startup=True,
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

@celery.task
def test_task():
    return "Worker is running!"
