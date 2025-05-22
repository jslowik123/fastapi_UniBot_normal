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

@celery.task(name="tasks.generate_namespace_summary")
def generate_namespace_summary(namespace: str):
    """
    Generates and stores a global summary for all documents in a namespace.
    """
    try:
        print(f"[TASK_generate_summary] Starting for namespace: {namespace}")
        
        if not doc_processor._firebase_available:
            print(f"[TASK_generate_summary] Firebase not available. Skipping for {namespace}.")
            return {"status": "skipped", "message": "Firebase not available"}

        # 1. Fetch all document data from the namespace using DocProcessor's method
        print(f"[TASK_generate_summary] Fetching namespace data for: {namespace}")
        
        # DocProcessor.get_namespace_data directly returns a List[Dict[str, Any]] or []
        # It handles internal errors or empty data by returning an empty list or might raise an exception
        # if FirebaseConnection itself raises a critical one that DocProcessor doesn't catch.
        try:
            actual_documents_list = doc_processor.get_namespace_data(namespace)
        except Exception as e:
            # Catching potential errors during the call to get_namespace_data itself
            print(f"[TASK_generate_summary] Error calling doc_processor.get_namespace_data for {namespace}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error_fetching_data", "message": f"Failed to fetch data for namespace {namespace}: {str(e)}", "namespace": namespace}
        
        # No need to check for response.get('status') as actual_documents_list is the list itself.
        if not actual_documents_list:
            print(f"[TASK_generate_summary] No processable document data found to summarize in namespace {namespace}. (Result from get_namespace_data was empty)")
            return {"status": "skipped", "message": "No processable document data to summarize"}

        # 2. Generate and store the global summary
        print(f"[TASK_generate_summary] Generating summary from {len(actual_documents_list)} documents in {namespace}.")
        summary_result = doc_processor.generate_global_summary(namespace=namespace, documents_data=actual_documents_list)

        # 3. Log and return the result from generate_global_summary
        if summary_result.get("status") == "success":
            print(f"[TASK_generate_summary] Success for {namespace}: {summary_result.get('message')}")
        else:
            error_message = summary_result.get('message', 'Unknown error or non-success status from DocProcessor.generate_global_summary.')
            print(f"[TASK_generate_summary] Non-success status for {namespace} from DocProcessor: {error_message} (Status: {summary_result.get('status')})")
        
        return summary_result
            
    except Exception as e:
        print(f"[TASK_generate_summary] Unhandled task-level exception for {namespace}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Unhandled task exception: {str(e)}", "namespace": namespace}
