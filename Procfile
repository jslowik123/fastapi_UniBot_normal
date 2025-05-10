web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000} --timeout-keep-alive 75 --timeout-graceful-shutdown 30
worker: celery -A celery_app.celery worker --loglevel=info --pool=solo