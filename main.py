from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pinecone
from dotenv import load_dotenv
import os
import uvicorn
from pinecone_connection import PineconeCon
from chatbot import get_bot, message_bot, message_bot_stream
from doc_processor import DocProcessor
from firebase_connection import FirebaseConnection
from celery_app import test_task, celery
from tasks import process_document
from redis import Redis
import json
import asyncio
from celery.exceptions import Ignore
import traceback

# Load environment variables
load_dotenv()

# Constants
API_VERSION = "1.0.0"
DEFAULT_DIMENSION = 1536
DEFAULT_NUM_RESULTS = 5
STREAM_DELAY = 0.01

# Initialize environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize FastAPI app
app = FastAPI(
    title="Uni Chatbot API",
    description="API for university document processing and chatbot interactions",
    version=API_VERSION
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize connections
pc = pinecone.Pinecone(api_key=pinecone_api_key)
con = PineconeCon("pdfs-index")
doc_processor = DocProcessor(pinecone_api_key, openai_api_key)


class ChatState:
    """
    Manages the state of the chatbot conversation.
    
    Stores the conversation chain and chat history for maintaining
    context across multiple interactions.
    """
    
    def __init__(self):
        self.chain = None
        self.chat_history = []
    
    def reset(self):
        """Reset the chat state to initial values."""
        self.chain = None
        self.chat_history = []


chat_state = ChatState()


@app.get("/")
async def root():
    """
    Root endpoint providing API information and health status.
    
    Returns:
        Dict containing welcome message, status, and version information
    """
    return {
        "message": "Welcome to the Uni Chatbot API", 
        "status": "online", 
        "version": API_VERSION
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...), fileID: str = Form(...), additionalInfo: str = Form(...)):
    """
    Upload and process a PDF document asynchronously.
    
    Args:
        file: PDF file to upload and process
        namespace: Namespace for organizing documents  
        fileID: Unique identifier for the document
        
    Returns:
        Dict containing upload status and task information
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            return {
                "status": "error",
                "message": "Only PDF files are supported",
                "filename": file.filename
                
            }
            
        content = await file.read()
        task = process_document.delay(content, namespace, fileID, file.filename)
        
        return {
            "status": "success",
            "message": "File upload started",
            "task_id": task.id,
            "filename": file.filename
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error processing file: {str(e)}", 
            "filename": file.filename
        }


@app.post("/delete")
async def delete_file(file_name: str = Form(...), namespace: str = Form(...), 
                     fileID: str = Form(...), just_firebase: str = Form(...)):
    """
    Delete a document from Pinecone and/or Firebase.
    
    Args:
        file_name: Name of the file to delete
        namespace: Namespace containing the document
        fileID: Document identifier
        just_firebase: If "true", delete only from Pinecone, otherwise delete from both
        
    Returns:
        Dict containing deletion status from both services
    """
    try:
        firebase = FirebaseConnection()
        
        if just_firebase.lower() == "true":
            # Delete only Pinecone embeddings
            pinecone_result = con.delete_embeddings(file_name, namespace)
            firebase_result = firebase.delete_document_metadata(namespace, fileID)
            
            return {
                "status": "success", 
                "message": f"File {file_name} deleted successfully",
                "pinecone_status": pinecone_result.get("status", "unknown"),
                "firebase_status": firebase_result["status"],
                "firebase_message": firebase_result["message"]
            }
        else:
            # Delete from both services
            firebase_result = firebase.delete_document_metadata(namespace, fileID)
            pinecone_result = con.delete_embeddings(file_name, namespace)

            return {
                "status": "success", 
                "message": f"File {file_name} deleted successfully",
                "pinecone_status": pinecone_result.get("status", "unknown"),
                "pinecone_message": pinecone_result.get("message", ""),
                "firebase_status": firebase_result["status"],
                "firebase_message": firebase_result["message"]
            }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error deleting file: {str(e)}"
        }


@app.post("/start_bot")
async def start_bot():
    """
    Initialize the chatbot and reset conversation state.
    
    Returns:
        Dict containing initialization status
    """
    try:
        chat_state.chain = get_bot()
        chat_state.chat_history = []
        return {
            "status": "success", 
            "message": "Bot started successfully"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error starting bot: {str(e)}"
        }

def _get_relevant_context(user_input: str, namespace: str, history: list) -> tuple:
    """
    Get relevant context for a user query from document database.
    
    Args:
        user_input: User's question or message
        namespace: Namespace to search within
        history: Chat history for context
        
    Returns:
        Tuple containing (context_text, database_overview, document_id, error_message)
        If successful: (context_string, database_data, document_id, None)
        If failed: (None, None, None, error_message)
    """
    try:
        # BULLETPROOF: Sanitize inputs - never throw errors for empty strings
        if not user_input or not isinstance(user_input, str):
            user_input = "Bitte stellen Sie eine Frage"
        user_input = user_input.strip() if user_input.strip() else "Bitte stellen Sie eine Frage"
        
        if not namespace or not isinstance(namespace, str):
            namespace = "default"
        namespace = namespace.strip() if namespace.strip() else "default"
        
        if not isinstance(history, list):
            history = []
        
        # Get namespace overview
        try:
            database_overview = doc_processor.get_namespace_data(namespace)
        except Exception as e:
            return "", [], "", None  # Return empty but valid values instead of error
        
        # BULLETPROOF: Check if we have valid database overview
        if not database_overview or not isinstance(database_overview, list):
            return "", [], "", None  # Return empty but valid values instead of error
            
        # Find appropriate document
        try:
            appropriate_document = doc_processor.appropriate_document_search(
                namespace=namespace, extracted_data=database_overview, user_query=user_input, history=history,
            )
        except Exception as e:
            return "", database_overview, "", None  # Return empty but valid values instead of error
        
        # BULLETPROOF: Validate document selection result
        if not appropriate_document or not isinstance(appropriate_document, dict) or "id" not in appropriate_document:
            return "", database_overview, "", None  # Return empty but valid values instead of error
        
        document_id = appropriate_document["id"]
        if not document_id or not isinstance(document_id, str):
            document_id = ""

        # Generate optimized query for vector search
        optimized_query = user_input  # Default fallback
        try:
            if document_id and document_id != "no_document_found":
                # Find the selected document's metadata
                selected_document = next((doc for doc in database_overview if doc.get("id") == document_id), None)
                if selected_document:
                    optimized_query = doc_processor.generate_search_query(
                        user_input=user_input,
                        document_metadata=selected_document,
                        history=history
                    )
        except Exception as e:
            # If query generation fails, use original user_input
            optimized_query = user_input

        # Query vector database
        try:
            results = con.query(
                query=optimized_query, 
                namespace=namespace, 
                fileID=document_id, 
                num_results=DEFAULT_NUM_RESULTS,
            )
            
            # STRUKTURIERTE AUSGABE - Vektor Query Ergebnisse
            print("\n" + "="*80)
            print("VEKTOR QUERY ERGEBNISSE:")
            print("-" * 40)
            print(f"Original Query: '{user_input}'")
            print(f"Optimized Query: '{optimized_query}'")
            print(f"Namespace: {namespace}")
            print(f"Document ID: {document_id}")
            print(f"Gefundene Matches: {len(results.matches) if results.matches else 0}")
            
            if results.matches:
                for i, match in enumerate(results.matches):
                    print(f"\nMatch {i+1}:")
                    print(f"  Score: {match.score:.4f}")
                    print(f"  ID: {match.id}")
                    if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                        text_preview = match.metadata['text'][:150] + "..." if len(match.metadata['text']) > 150 else match.metadata['text']
                        print(f"  Text: {text_preview}")
            print("="*80 + "\n")
            
        except Exception as e:
            return "", database_overview, document_id, None  # Return empty but valid values instead of error
        
        # BULLETPROOF: Extract context from results with comprehensive validation
        context_parts = []
        if results and hasattr(results, 'matches') and results.matches:
            for match in results.matches:
                # Multiple layers of validation
                if (hasattr(match, 'metadata') and 
                    match.metadata and 
                    isinstance(match.metadata, dict) and 
                    'text' in match.metadata and 
                    match.metadata['text'] and 
                    isinstance(match.metadata['text'], str) and 
                    match.metadata['text'].strip()):
                    
                    context_parts.append(match.metadata['text'].strip())
        
        # BULLETPROOF: Always return valid context, even if empty
        try:
            context = "\n".join(context_parts) if context_parts else ""
            return context, database_overview, document_id, None
            
        except Exception as e:
            return "", database_overview, document_id, None  # Return empty but valid values instead of error
            
    except Exception as e:
        return "", [], "", None  # Return empty but valid values instead of error



@app.post("/create_namespace")
async def create_namespace(namespace: str = Form(...), dimension: int = Form(DEFAULT_DIMENSION)):
    """
    Create a new namespace in the Pinecone index.
    
    Args:
        namespace: Name of the namespace to create
        dimension: Vector dimension (default: 1536 for OpenAI embeddings)
        
    Returns:
        Dict containing creation status
    """
    try:
        pc = PineconeCon("userfiles")
        result = pc.create_namespace_with_dummy(namespace, dimension)
        return result
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error creating namespace: {str(e)}"
        }


@app.post("/delete_namespace")
async def delete_namespace(namespace: str = Form(...)):
    """
    Delete a namespace from Pinecone index and Firebase metadata.
    
    Args:
        namespace: Name of the namespace to delete
        
    Returns:
        Dict containing deletion status from both services
    """
    try:
        pc = PineconeCon("userfiles")
        pinecone_result = pc.delete_namespace(namespace)
        
        firebase = FirebaseConnection()
        firebase_result = firebase.delete_namespace_metadata(namespace)
        
        return {
            "status": "success", 
            "message": f"Namespace {namespace} deleted successfully",
            "pinecone_status": pinecone_result.get("status", "unknown"),
            "pinecone_message": pinecone_result.get("message", ""),
            "firebase_status": firebase_result["status"],
            "firebase_message": firebase_result["message"]
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Error deleting namespace: {str(e)}"
        }


@app.get("/test_worker")
async def test_worker():
    """
    Test the Celery worker connectivity.
    
    Returns:
        Dict containing test task information
    """
    try:
        result = test_task.delay()
        return {
            "status": "success", 
            "task_id": result.id, 
            "message": "Test task sent to worker"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error testing worker: {str(e)}"
        }


def _handle_task_state(task) -> dict:
    """
    Handle different Celery task states and format response accordingly.
    
    Args:
        task: Celery AsyncResult object
        
    Returns:
        Dict containing formatted task status information
    """
    if task.state == 'PENDING':
        return {
            'state': task.state,
            'status': 'PENDING',
            'message': 'Task is waiting for execution',
            'progress': 0
        }
    elif task.state in ['STARTED', 'PROCESSING']:
        meta = task.info if isinstance(task.info, dict) else {}
        return {
            'state': task.state,
            'status': 'PROCESSING',
            'message': meta.get('status', 'Processing'),
            'current': meta.get('current', 0),
            'total': meta.get('total', 100),
            'progress': meta.get('current', 0),
            'file': meta.get('file', '')
        }
    elif task.state in ['FAILURE', 'REVOKED']:
        # Handle failure states
        if isinstance(task.info, Exception):
            error_info = {
                'type': type(task.info).__name__,
                'message': str(task.info),
                'details': 'Task failed with an exception'
            }
        else:
            meta = task.info if isinstance(task.info, dict) else {}
            error_info = {
                'type': meta.get('exc_type', type(task.result).__name__ if task.result else 'Unknown'),
                'message': meta.get('exc_message', str(task.result) if task.result else 'Unknown error'),
                'details': meta.get('error', 'No additional details available')
            }
        
        raise HTTPException(
            status_code=500,
            detail={
                'state': task.state,
                'status': 'FAILURE',
                'message': 'Task processing failed',
                'error': error_info,
                'progress': 0
            }
        )
    elif task.state == 'SUCCESS':
        result = task.result if isinstance(task.result, dict) else {}
        
        if not result:
            return {
                'state': task.state,
                'status': 'SUCCESS',
                'message': 'Task completed but no result available',
                'progress': 100,
                'result': {
                    'message': 'No result data available',
                    'chunks': 0,
                    'pinecone_status': 'unknown',
                    'firebase_status': 'unknown',
                    'file': ''
                }
            }
        else:
            return {
                'state': task.state,
                'status': 'SUCCESS',
                'message': 'Completed successfully',
                'progress': 100,
                'result': {
                    'message': result.get('message', 'Task completed'),
                    'chunks': result.get('chunks', 0),
                    'pinecone_status': result.get('pinecone_result', {}).get('status', 'unknown'),
                    'firebase_status': result.get('firebase_result', {}).get('status', 'unknown'),
                    'file': result.get('file', '')
                }
            }
    else:
        return {
            'state': task.state,
            'status': 'UNKNOWN',
            'message': f'Unknown state: {task.state}',
            'info': str(task.info) if task.info else 'No info available',
            'progress': 0
        }


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get the status of an asynchronous task.
    
    Args:
        task_id: Unique identifier of the task to check
        
    Returns:
        Dict containing task status, progress, and results
        
    Raises:
        HTTPException: If task failed or status check encountered an error
    """
    try:
        task = celery.AsyncResult(task_id)
        print(f"Task state: {task.state}")
        print(f"Task info: {task.info}")
        print(f"Task result: {task.result}")
        
        response = _handle_task_state(task)
        print(f"Sending response: {response}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in task status: {str(e)}")
        error_detail = {
            'state': 'ERROR',
            'status': 'ERROR',
            'message': 'Error checking task status',
            'error': {
                'type': type(e).__name__,
                'message': str(e),
                'details': 'Error occurred while checking task status'
            },
            'progress': 0
        }
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/send_message")
async def send_message(user_input: str = Form(...), namespace: str = Form(...)):
    """
    Send a message to the bot and get a structured response.
    
    Args:
        user_input: User's question or message  
        namespace: Namespace to search for relevant documents
        
    Returns:
        JSON response with the bot's answer
    """
    # BULLETPROOF: Sanitize inputs - never throw errors for empty strings
    if not user_input or not isinstance(user_input, str):
        user_input = "Bitte stellen Sie eine Frage"
    user_input = user_input.strip() if user_input.strip() else "Bitte stellen Sie eine Frage"
    
    if not namespace or not isinstance(namespace, str):
        namespace = "default"
    namespace = namespace.strip() if namespace.strip() else "default"

    if not chat_state.chain:
        raise HTTPException(
            status_code=400,
            detail="Bot not started. Please call /start_bot first."
        )

    try:
        # BULLETPROOF: Get chat history safely
        history = chat_state.chat_history if chat_state.chat_history else []
        
        context, database_overview, document_id, error = _get_relevant_context(user_input, namespace, history)
        
        # BULLETPROOF: Always continue, even if context retrieval had issues
        if context is None:
            context = ""
        
        # BULLETPROOF: Validate all parameters before calling message_bot
        if not isinstance(context, str):
            context = str(context) if context else ""
        
        if not isinstance(document_id, str):
            document_id = str(document_id) if document_id else ""
        
        # Get the AI response
        response = message_bot(
            user_input, 
            context, 
            "", 
            document_id, 
            history
        )
        
        # BULLETPROOF: Validate response
        if not response or not isinstance(response, str):
            response = "Entschuldigung, ich konnte keine Antwort generieren."
        
        # BULLETPROOF: Update chat history safely
        try:
            if not chat_state.chat_history:
                chat_state.chat_history = []
            
            chat_state.chat_history.append({"role": "user", "content": user_input})
            chat_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            # Don't fail the request if chat history update fails
            pass
        
        final_response = {
            "status": "success",
            "response": response
        }
        return final_response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Return a valid response even if something goes wrong
        return {
            "status": "success",
            "response": "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."
        }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )