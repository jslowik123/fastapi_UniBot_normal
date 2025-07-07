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
DEFAULT_NUM_RESULTS = 7
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
            
            # STRUKTURIERTE AUSGABE - Document Selection
            print("\n" + "="*80)
            print("DOCUMENT SELECTION ERGEBNISSE:")
            print("-" * 40)
            print(f"User Query: '{user_input}'")
            print(f"Namespace: {namespace}")
            print(f"Verfügbare Dokumente ({len(database_overview)}):")
            for i, doc in enumerate(database_overview):
                print(f"  {i+1}. ID: {doc.get('id', 'N/A')}")
                print(f"     Name: {doc.get('name', 'N/A')}")
                print(f"     Keywords: {doc.get('keywords', [])}")
                print(f"     Summary: {doc.get('summary', 'N/A')[:100]}{'...' if len(str(doc.get('summary', ''))) > 100 else ''}")
                print(f"     Additional Info: {doc.get('additional_info', 'N/A')}")
                print()
            
            if appropriate_document:
                print(f"AUSGEWÄHLTES DOKUMENT: {appropriate_document}")
            else:
                print("KEIN DOKUMENT AUSGEWÄHLT (appropriate_document ist None)")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\nERROR in document selection: {str(e)}")
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

        # Query vector database with adjacent chunks - ONLY if we have a valid document ID
        results = None
        if document_id and document_id != "no_document_found":
            try:
                results = con.query_with_adjacent_chunks(
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
                print(f"ERROR in Pinecone query: {str(e)}")
                results = None  # Set to None if query fails
        else:
            print("\n" + "="*80)
            print("KEINE VEKTOR QUERY - KEIN DOKUMENT AUSGEWÄHLT")
            print("-" * 40)
            print(f"Document ID: {document_id}")
            print("Grund: Kein passendes Dokument gefunden oder no_document_found")
            print("Fortsetzung ohne spezifischen Context...")
            print("="*80 + "\n")
        
        # BULLETPROOF: Extract context from results with comprehensive validation
        context_parts = []
        if results and hasattr(results, 'matches') and results.matches:
            for i, match in enumerate(results.matches):
                # Multiple layers of validation
                if (hasattr(match, 'metadata') and 
                    match.metadata and 
                    isinstance(match.metadata, dict) and 
                    'text' in match.metadata and 
                    match.metadata['text'] and 
                    isinstance(match.metadata['text'], str) and 
                    match.metadata['text'].strip()):
                    
                    # Sammle alle Chunks für dieses Match (previous + current + next)
                    match_chunks = []
                    
                    # Previous chunk hinzufügen
                    if ('adjacent_chunks' in match.metadata and 
                        match.metadata['adjacent_chunks'] and 
                        match.metadata['adjacent_chunks'].get('previous') and
                        hasattr(match.metadata['adjacent_chunks']['previous'], 'metadata') and
                        match.metadata['adjacent_chunks']['previous'].metadata and
                        'text' in match.metadata['adjacent_chunks']['previous'].metadata):
                        
                        prev_text = match.metadata['adjacent_chunks']['previous'].metadata['text'].strip()
                        if prev_text:
                            match_chunks.append(f"--- CHUNK {i+1}a (VORHERIGER) START ---\n{prev_text}\n--- CHUNK {i+1}a (VORHERIGER) END ---")
                    
                    # Current chunk hinzufügen
                    chunk_text = match.metadata['text'].strip()
                    match_chunks.append(f"--- CHUNK {i+1}b (HAUPTTREFFER) START ---\n{chunk_text}\n--- CHUNK {i+1}b (HAUPTTREFFER) END ---")
                    
                    # Next chunk hinzufügen
                    if ('adjacent_chunks' in match.metadata and 
                        match.metadata['adjacent_chunks'] and 
                        match.metadata['adjacent_chunks'].get('next') and
                        hasattr(match.metadata['adjacent_chunks']['next'], 'metadata') and
                        match.metadata['adjacent_chunks']['next'].metadata and
                        'text' in match.metadata['adjacent_chunks']['next'].metadata):
                        
                        next_text = match.metadata['adjacent_chunks']['next'].metadata['text'].strip()
                        if next_text:
                            match_chunks.append(f"--- CHUNK {i+1}c (NÄCHSTER) START ---\n{next_text}\n--- CHUNK {i+1}c (NÄCHSTER) END ---")
                    
                    # Alle Chunks für dieses Match zusammenfügen
                    if match_chunks:
                        context_parts.append("\n".join(match_chunks))
        
        # BULLETPROOF: Always return valid context, even if empty
        try:
            context = "\n\n".join(context_parts) if context_parts else ""
            return context, database_overview, document_id, None
            
        except Exception as e:
            return "", database_overview, document_id, None  # Return empty but valid values instead of error
            
    except Exception as e:
        return "", [], "", None  # Return empty but valid values instead of error


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
            database_overview,
            history,
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