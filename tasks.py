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
        if not doc_processor._firebase_available:
            print(f"Firebase not available. Skipping global summary for {namespace}.")
            return {"status": "skipped", "message": "Firebase not available"}

        # 1. Fetch all document data from the namespace
        namespace_data_response = doc_processor._firebase.get_namespace_data(namespace)
        
        if namespace_data_response.get('status') != 'success' or not namespace_data_response.get('data'):
            print(f"No data found for namespace {namespace} or error fetching data. Skipping summary.")
            return {"status": "skipped", "message": "No documents or error fetching data"}

        documents = []
        # The actual documents are typically nested under their fileIDs
        for file_id, doc_content in namespace_data_response['data'].items():
            if file_id == "_global_summary": # Skip the global summary itself
                continue
            if isinstance(doc_content, dict):
                 # We need the summary or text of each document
                documents.append(doc_content) 

        if not documents:
            print(f"No document content found to summarize in namespace {namespace}.")
            return {"status": "skipped", "message": "No document content to summarize"}

        # 2. Generate the global summary
        print(f"Generating global summary from {len(documents)} documents in namespace {namespace}.")
        global_summary_text = doc_processor.generate_global_summary(documents)

        if global_summary_text == "Error generating global summary." or global_summary_text == "No content available from documents to generate a summary." or global_summary_text == "No documents found to summarize.":
            print(f"Failed to generate global summary for {namespace}: {global_summary_text}")
            # Optionally, update Firebase with an error status for the global summary
            doc_processor._firebase.update_namespace_summary(namespace, f"Failed to generate summary: {global_summary_text}")
            return {"status": "error", "message": global_summary_text}

        # 3. Store the global summary in Firebase
        print(f"Storing global summary for namespace {namespace}.")
        store_result = doc_processor._firebase.update_namespace_summary(namespace, global_summary_text)
        
        if store_result.get('status') == 'success':
            print(f"Global summary for namespace {namespace} stored successfully.")
            return {"status": "success", "message": "Global summary generated and stored.", "summary": global_summary_text}
        else:
            print(f"Failed to store global summary for {namespace}: {store_result.get('message')}")
            return {"status": "error", "message": f"Failed to store global summary: {store_result.get('message')}"}
            
    except Exception as e:
        print(f"Exception during global summary generation for {namespace}: {str(e)}")
        # Optionally, update Firebase with an error status for the global summary
        if doc_processor._firebase_available:
            doc_processor._firebase.update_namespace_summary(namespace, f"Exception during summary generation: {str(e)}")
        return {"status": "error", "message": f"Exception: {str(e)}"}
