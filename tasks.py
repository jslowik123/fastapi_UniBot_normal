from celery_app import celery
import time
from redis import Redis
import os
from doc_processor import DocProcessor
from dotenv import load_dotenv

load_dotenv()

r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
doc_processor = DocProcessor(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@celery.task(name="tasks.process_file")
def process_file(job_id: str):
    r.set(f"job:{job_id}", "started")
    time.sleep(2)  # Simuliere Arbeit
    r.set(f"job:{job_id}", "50%")
    time.sleep(2)
    r.set(f"job:{job_id}", "done")

@celery.task(bind=True, name="tasks.process_document")
def process_document(self, file_path: str, namespace: str, fileID: str):
    try:
        self.update_state(state='PROCESSING', meta={
            'status': 'Processing document',
            'current': 0,
            'total': 100,
            'file': file_path
        })
        
        result = doc_processor.process_pdf(file_path, namespace, fileID)
        
        if result['status'] == 'success':
            return {
                'status': result['status'],
                'message': result['message'],
                'chunks': result.get('chunks', 0),
                'pinecone_result': result.get('pinecone_result', {}),
                'firebase_result': result.get('firebase_result', {}),
                'file': file_path
            }
        else:
            raise Exception(result['message'])
            
    except Exception as e:
        self.update_state(state='FAILURE', meta={
            'error': str(e),
            'file': file_path
        })
        raise
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)
