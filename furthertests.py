import os

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
from redis import Redis
r = Redis.from_url(redis_url, ssl_cert_reqs=None)
r.ping()  # Should return True