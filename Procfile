web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}
worker: celery -A celery_app.celery worker --loglevel=info