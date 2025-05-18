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
def process_document(self, file_content: bytes, namespace: str, fileID: str, filename: str):
    try:
        import io
        from PyPDF2 import PdfReader
        
        if doc_processor._firebase_available:
            doc_processor._firebase.update_document_status(namespace, fileID, {
                'processing': True,
                'progress': 0,
                'status': 'Starting document processing'
            })
        
        self.update_state(
            state='STARTED',
            meta={
                'status': 'Starting document processing',
                'current': 0,
                'total': 100,
                'file': filename
            }
        )
        
        if doc_processor._firebase_available:
            doc_processor._firebase.update_document_status(namespace, fileID, {
                'progress': 25,
                'status': 'Reading document'
            })
        
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Reading document',
                'current': 25,
                'total': 100,
                'file': filename
            }
        )
        
        pdf_file = io.BytesIO(file_content)
        result = doc_processor.process_pdf_bytes(pdf_file, namespace, fileID, filename)
        
        if doc_processor._firebase_available:
            doc_processor._firebase.update_document_status(namespace, fileID, {
                'progress': 75,
                'status': 'Finalizing processing'
            })
        
        self.update_state(
            state='PROCESSING',
            meta={
                'status': 'Finalizing processing',
                'current': 75,
                'total': 100,
                'file': filename
            }
        )
        
        if result['status'] == 'success':
            if doc_processor._firebase_available:
                doc_processor._firebase.update_document_status(namespace, fileID, {
                    'processing': False,
                    'progress': 100,
                    'status': 'Complete'
                })
            return {
                'status': 'success',
                'message': result['message'],
                'chunks': result.get('chunks', 0),
                'pinecone_result': result.get('pinecone_result', {}),
                'firebase_result': result.get('firebase_result', {}),
                'file': filename,
                'current': 100,
                'total': 100
            }
        else:
            if doc_processor._firebase_available:
                doc_processor._firebase.update_document_status(namespace, fileID, {
                    'processing': False,
                    'progress': 0,
                    'status': f"Processing failed: {result['message']}"
                })
            raise Exception(f"Processing failed: {result['message']}")
            
    except Exception as e:
        if doc_processor._firebase_available:
            doc_processor._firebase.update_document_status(namespace, fileID, {
                'processing': False,
                'progress': 0,
                'status': f"Failed: {str(e)}"
            })
        
        self.update_state(
            state='FAILURE',
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'status': 'Failed',
                'error': f"{type(e).__name__}: {str(e)}",
                'file': filename,
                'current': 0,
                'total': 100
            }
        )
        raise e
