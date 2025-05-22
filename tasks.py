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
        print(f"Starting global summary generation for namespace: {namespace}")
        
        # Ensure DocProcessor instance is available (it's global in this file)
        # global doc_processor # Not strictly needed if doc_processor is already in the global scope of the module

        if not doc_processor._firebase_available: # Assuming doc_processor is the global instance
            print(f"Firebase not available. Skipping global summary for {namespace}.")
            return {"status": "skipped", "message": "Firebase not available"}

        # 1. Fetch all document data from the namespace using DocProcessor's method
        # This method should return List[Dict[str, Any]] suitable for generate_global_summary
        documents_data_list = doc_processor.get_namespace_data(namespace)
        
        if not documents_data_list: # get_namespace_data returns a list
            print(f"No data found for namespace {namespace} or error fetching data. Skipping summary.")
            return {"status": "skipped", "message": "No documents or error fetching data from get_namespace_data"}

        # Filter out any potential non-document entries if get_namespace_data doesn't already do that
        # The current get_namespace_data in DocProcessor already filters and returns a list of dicts
        # with 'summary', 'keywords' etc.
        
        if not documents_data_list:
            print(f"No document content found to summarize in namespace {namespace}.")
            return {"status": "skipped", "message": "No document content to summarize"}

        # 2. Generate and store the global summary using the modified DocProcessor method
        print(f"Generating global summary from {len(documents_data_list)} documents in namespace {namespace}.")
        # The generate_global_summary method now takes namespace and documents_data
        # and handles storing to Firebase itself.
        summary_result = doc_processor.generate_global_summary(namespace=namespace, documents_data=documents_data_list)

        # 3. Log the result from generate_global_summary
        if summary_result.get("status") == "success":
            print(f"Global summary for namespace {namespace} processed successfully by DocProcessor: {summary_result.get('message')}")
            return summary_result # Return the detailed result from DocProcessor
        elif summary_result.get("status") == "success_firebase_unavailable":
            print(f"Global summary generated but Firebase was unavailable for storage for namespace {namespace}: {summary_result.get('message')}")
            return summary_result
        elif summary_result.get("status") == "success_no_points":
            print(f"Global summary generation for {namespace} resulted in no bullet points: {summary_result.get('message')}")
            return summary_result
        else:
            # Handles error_openai, error_firebase_storage, error_firebase_method, or other errors from DocProcessor
            error_message = summary_result.get('message', 'Unknown error during summary generation in DocProcessor.')
            print(f"Failed to generate/store global summary for {namespace} via DocProcessor: {error_message}")
            # Optionally, update a general error status in Firebase if needed,
            # but generate_global_summary might already log specifics.
            # For example, if you had a general status field for the namespace:
            # if doc_processor._firebase_available:
            #     doc_processor._firebase.set_data(f"{namespace}/summary_status", {"error": error_message, "timestamp": time.time()})
            return summary_result # Return the detailed error result from DocProcessor
            
    except Exception as e:
        print(f"Unhandled exception during global summary generation for {namespace}: {str(e)}")
        # Optionally, update Firebase with an error status for the global summary
        # This is a catch-all for unexpected errors in the task itself.
        # if doc_processor._firebase_available: # Check again in case it became unavailable
        #     try:
        #         # A generic error logging for the namespace if specific methods aren't available
        #         doc_processor._firebase.set_data(f"{namespace}/summary_status", {"error": f"Task level exception: {str(e)}", "timestamp": time.time()})
        #     except Exception as fb_error:
        #         print(f"Failed to log task-level exception to Firebase: {fb_error}")
        return {"status": "error", "message": f"Unhandled task exception: {str(e)}"}
