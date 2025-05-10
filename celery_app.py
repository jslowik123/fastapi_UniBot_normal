from celery import Celery
import os



redis_url = "rediss://:p277c32da1640e7109e3e7fd52cd4c73d5b85ac54e4972db72473a2fa0b02742d@ec2-54-171-238-172.eu-west-1.compute.amazonaws.com:31030?ssl_cert_reqs=CERT_NONE"

celery = Celery("myapp", broker=redis_url, backend=redis_url)
celery.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    broker_use_ssl={
        'ssl_cert_reqs': None,
        'ssl_ca_certs': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_check_hostname': False
    },
    redis_backend_use_ssl={
        'ssl_cert_reqs': None,
        'ssl_ca_certs': None,
        'ssl_certfile': None,
        'ssl_keyfile': None,
        'ssl_check_hostname': False
    }
)

@celery.task
def test_task():
    return "Worker is running!"
