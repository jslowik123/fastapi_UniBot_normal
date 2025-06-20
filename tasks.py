from celery_app import celery
import time
from redis import Redis
import os
from doc_processor import DocProcessor
from dotenv import load_dotenv

load_dotenv()

# Initialize Redis and DocProcessor
r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
doc_processor = DocProcessor(
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

@celery.task(bind=True, name="tasks.process_document")
def process_document(self, file_content: bytes, namespace: str, fileID: str, filename: str):
    """
    Process a document asynchronously using Celery.
    
    Handles PDF parsing, text extraction, embedding generation, and storage
    in both Pinecone and Firebase with progress tracking.
    
    Args:
        self: Celery task instance (bound task)
        file_content: Raw PDF file content as bytes
        namespace: Pinecone namespace for organization
        fileID: Unique document identifier
        filename: Original filename
        
    Returns:
        Dict containing processing results and status
        
    Raises:
        Exception: If processing fails at any stage
    """
    try:
        import io
        from PyPDF2 import PdfReader
        
        # Update initial status
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
        
        # Update status: Reading document
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
        
        # Process the PDF file
        pdf_file = io.BytesIO(file_content)
        result = doc_processor.process_pdf_bytes(pdf_file, namespace, fileID, filename)
        
        # Update status: Finalizing processing
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
            # Update final status
            if doc_processor._firebase_available:
                doc_processor._firebase.update_document_status(namespace, fileID, {
                    'processing': False,
                    'progress': 100,
                    'status': 'Complete'
                })
            
            # Trigger the global summary generation task
            generate_namespace_summary.delay(namespace)
            
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
            # Handle processing failure
            if doc_processor._firebase_available:
                doc_processor._firebase.update_document_status(namespace, fileID, {
                    'processing': False,
                    'progress': 0,
                    'status': f"Processing failed: {result['message']}"
                })
            raise Exception(f"Processing failed: {result['message']}")
            
    except Exception as e:
        # Update error status
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


@celery.task(name="tasks.generate_namespace_summary")
def generate_namespace_summary(namespace: str):
    """
    Generate and store a global summary for all documents in a namespace.
    
    This task is triggered after successful document processing to maintain
    an up-to-date overview of all documents in the namespace.
    
    Args:
        namespace: The namespace to generate a summary for
        
    Returns:
        Dict containing operation status and results
    """
    try:
        result = doc_processor.generate_global_summary(namespace)
        return {
            'status': 'success',
            'message': f"Global summary generated for namespace: {namespace}",
            'summary_result': result
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Failed to generate global summary: {str(e)}"
        }