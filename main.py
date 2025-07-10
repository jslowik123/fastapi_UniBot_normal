from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pinecone
from dotenv import load_dotenv
import os
import uvicorn
from pinecone_connection import PineconeCon
from chatbot import get_bot, message_bot
from doc_processor import DocProcessor
import logging


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
    
    Stores the chat history for maintaining context across multiple interactions.
    No longer stores LangChain chains since we use direct OpenAI API.
    """
    
    def __init__(self):
        self.bot_initialized = False  # Simple boolean instead of chain
        self.chat_history = []
    
    def reset(self):
        """Reset the chat state to initial values."""
        self.bot_initialized = False
        self.chat_history = []


chat_state = ChatState()


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


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
        success = get_bot()  # Returns True if OpenAI client can be created
        if success:
            chat_state.bot_initialized = True
            chat_state.chat_history = []
            return {
                "status": "success", 
                "message": "Bot started successfully"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to initialize OpenAI client"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error starting bot: {str(e)}"
        }

def _sanitize_inputs(user_input: str, namespace: str, history: list) -> tuple:
    """
    Sanitize and validate input parameters.
    
    Args:
        user_input: User's question or message
        namespace: Namespace to search within
        history: Chat history for context
        
    Returns:
        Tuple of (sanitized_user_input, sanitized_namespace, sanitized_history)
    """
    # Sanitize user input
    if not user_input or not isinstance(user_input, str):
        user_input = "Bitte stellen Sie eine Frage"
    user_input = user_input.strip() if user_input.strip() else "Bitte stellen Sie eine Frage"
    
    # Sanitize namespace
    if not namespace or not isinstance(namespace, str):
        namespace = "default"
    namespace = namespace.strip() if namespace.strip() else "default"
    
    # Sanitize history
    if not isinstance(history, list):
        history = []
    
    return user_input, namespace, history


def _get_database_overview(namespace: str) -> tuple:
    """
    Get namespace overview from document processor.
    
    Args:
        namespace: Namespace to get overview for
        
    Returns:
        Tuple of (database_overview, error_occurred)
        If successful: (overview_list, False)
        If failed: ([], True)
    """
    try:
        database_overview = doc_processor.get_namespace_data(namespace)
        if not database_overview or not isinstance(database_overview, list):
            return [], True
        return database_overview, False
    except Exception as e:
        return [], True


def _select_appropriate_document(namespace: str, database_overview: list, 
                               user_input: str, history: list) -> tuple:
    """
    Select appropriate document for the user query.
    
    Args:
        namespace: Namespace to search within
        database_overview: List of available documents
        user_input: User's question
        history: Chat history
        
    Returns:
        Tuple of (selected_document_id, selected_document_name, error_occurred)
    """
    try:
        appropriate_document = doc_processor.appropriate_document_search(
            namespace=namespace,
            extracted_data=database_overview,
            user_query=user_input,
            history=history,
        )
        
        if not appropriate_document or not isinstance(appropriate_document, dict):
            return "", True
        
        # Get single document ID
        document_id = appropriate_document.get("id", "")
        document_name = appropriate_document.get("name", "")
        if document_id and isinstance(document_id, str) and document_id != "no_document_found":
            return document_id, document_name, False
        
        return "", True
        
    except Exception as e:
        return "", True


def _generate_optimized_query(user_input: str, selected_document_id: str, 
                            database_overview: list, history: list) -> str:
    """
    Generate optimized search query for the selected document.
    
    Args:
        user_input: Original user input
        selected_document_id: Selected document ID
        database_overview: Database overview with document metadata
        history: Chat history
        
    Returns:
        Optimized search query string
    """
    try:
        selected_document = next((doc for doc in database_overview if doc.get("id") == selected_document_id), None)
        if selected_document:
            return doc_processor.generate_search_query(
                user_input=user_input,
                document_metadata=selected_document,
                history=history
            )
        return user_input
    except Exception as e:
        return user_input


def _extract_chunks_from_match(match, doc_index: int, match_index: int) -> list:
    """
    Extract all chunks (previous, current, next) from a single match.
    
    Args:
        match: Pinecone match object
        doc_index: Document index for labeling
        match_index: Match index for labeling
        
    Returns:
        List of formatted chunk strings
    """
    match_chunks = []
    
    # Previous chunk
    if ('adjacent_chunks' in match.metadata and 
        match.metadata['adjacent_chunks'] and 
        match.metadata['adjacent_chunks'].get('previous') and
        hasattr(match.metadata['adjacent_chunks']['previous'], 'metadata') and
        match.metadata['adjacent_chunks']['previous'].metadata and
        'text' in match.metadata['adjacent_chunks']['previous'].metadata):
        
        prev_text = match.metadata['adjacent_chunks']['previous'].metadata['text'].strip()
        if prev_text:
            match_chunks.append(f"--- DOK{doc_index+1} CHUNK {match_index+1}a (VORHERIGER) START ---\n{prev_text}\n--- DOK{doc_index+1} CHUNK {match_index+1}a (VORHERIGER) END ---")
    
    # Current chunk
    chunk_text = match.metadata['text'].strip()
    match_chunks.append(f"--- DOK{doc_index+1} CHUNK {match_index+1}b (HAUPTTREFFER) START ---\n{chunk_text}\n--- DOK{doc_index+1} CHUNK {match_index+1}b (HAUPTTREFFER) END ---")
    
    # Next chunk
    if ('adjacent_chunks' in match.metadata and 
        match.metadata['adjacent_chunks'] and 
        match.metadata['adjacent_chunks'].get('next') and
        hasattr(match.metadata['adjacent_chunks']['next'], 'metadata') and
        match.metadata['adjacent_chunks']['next'].metadata and
        'text' in match.metadata['adjacent_chunks']['next'].metadata):
        
        next_text = match.metadata['adjacent_chunks']['next'].metadata['text'].strip()
        if next_text:
            match_chunks.append(f"--- DOK{doc_index+1} CHUNK {match_index+1}c (NÄCHSTER) START ---\n{next_text}\n--- DOK{doc_index+1} CHUNK {match_index+1}c (NÄCHSTER) END ---")
    
    return match_chunks


def _query_document(document_id: str, optimized_query: str, 
                   namespace: str, database_overview: list) -> str:
    """
    Query a document and extract context.
    
    Args:
        document_id: Document ID to query
        optimized_query: Optimized search query
        namespace: Namespace to search in
        database_overview: Database overview with metadata
        
    Returns:
        Formatted context string
    """
    try:
        # Query vector database
        results = con.query_with_adjacent_chunks(
            query=optimized_query,
            namespace=namespace,
            fileID=document_id,
            num_results=DEFAULT_NUM_RESULTS,
        )
        
        # Extract context from results
        document_context_parts = []
        
        if results and hasattr(results, 'matches') and results.matches:
            for i, match in enumerate(results.matches):
                # Validate match has required metadata
                if (hasattr(match, 'metadata') and 
                    match.metadata and 
                    isinstance(match.metadata, dict) and 
                    'text' in match.metadata and 
                    match.metadata['text'] and 
                    isinstance(match.metadata['text'], str) and 
                    match.metadata['text'].strip()):
                    
                    # Extract all chunks for this match
                    match_chunks = _extract_chunks_from_match(match, 0, i)
                    if match_chunks:
                        document_context_parts.append("\n".join(match_chunks))
        
        # Format document context with header and footer
        context = ""
        if document_context_parts:
            doc_name = next((doc.get('name', 'Dokument') for doc in database_overview if doc.get("id") == document_id), 'Dokument')
            context = f"\n\n=== INFORMATIONEN AUS DOKUMENT: {doc_name} (ID: {document_id}) ===\n" + "\n\n".join(document_context_parts) + f"\n=== ENDE DOKUMENT: {doc_name} ===\n\n"
        
        return context
        
    except Exception as e:
        logger.error(f"ERROR in Pinecone query for document {document_id}: {str(e)}")
        return ""


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
        If failed: ("", [], "", error_message)
    """
    try:
        print(f"[CONTEXT-DEBUG] Starting context retrieval for: '{user_input}' in namespace: '{namespace}'")
        
        # Step 1: Sanitize inputs
        user_input, namespace, history = _sanitize_inputs(user_input, namespace, history)
        print(f"[CONTEXT-DEBUG] After sanitization - user_input: '{user_input}', namespace: '{namespace}'")
        
        # Step 2: Get database overview
        database_overview, overview_error = _get_database_overview(namespace)
        if overview_error:
            print(f"[CONTEXT-DEBUG] Database overview error - returning empty context")
            return "", [], "", None
        print(f"[CONTEXT-DEBUG] Database overview found: {len(database_overview)} documents")
        
        # Step 3: Select appropriate document
        selected_document_id, selected_document_name, selection_error = _select_appropriate_document(
            namespace, database_overview, user_input, history
        )
        print(f"[CONTEXT-DEBUG] Document selection - ID: '{selected_document_id}', Name: '{selected_document_name}', Error: {selection_error}")
        
        if selection_error or not selected_document_id:
            print(f"[CONTEXT-DEBUG] No document selected - returning empty context")
            return "", database_overview, "", None
        
        # Step 4: Generate optimized query for the document
        optimized_query = _generate_optimized_query(
            user_input, selected_document_id, database_overview, history
        )
        print(f"[CONTEXT-DEBUG] Optimized query: '{optimized_query}'")
        
        # Step 5: Query the document
        context = _query_document(
            selected_document_id, optimized_query, namespace, database_overview
        )
        print(f"[CONTEXT-DEBUG] Query result - Context length: {len(context)}")
        
        print(f"[CONTEXT-DEBUG] Final result - Context length: {len(context)}")
        return context, database_overview, selected_document_id, None
        
    except Exception as e:
        print(f"[CONTEXT-DEBUG] Exception in context retrieval: {e}")
        return "", [], "", None




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
    logger.info(f"/send_message called with user_input='{user_input}' and namespace='{namespace}'")
    # BULLETPROOF: Sanitize inputs - never throw errors for empty strings
    try:
        if not user_input or not isinstance(user_input, str):
            user_input = "Bitte stellen Sie eine Frage"
        user_input = user_input.strip() if user_input.strip() else "Bitte stellen Sie eine Frage"
    except Exception as e:
        logger.error(f"Error sanitizing user_input: {e}")
        user_input = "Bitte stellen Sie eine Frage"
    try:
        if not namespace or not isinstance(namespace, str):
            namespace = "default"
        namespace = namespace.strip() if namespace.strip() else "default"
    except Exception as e:
        logger.error(f"Error sanitizing namespace: {e}")
        namespace = "default"

    if not chat_state.bot_initialized:
        logger.error("Bot not started. Please call /start_bot first.")
        raise HTTPException(
            status_code=400,
            detail="Bot not started. Please call /start_bot first."
        )

    try:
        # BULLETPROOF: Get chat history safely
        history = chat_state.chat_history if chat_state.chat_history else []
        logger.info(f"Chat history loaded: {history}")
        context, database_overview, document_id, error = _get_relevant_context(user_input, namespace, history)
        # BULLETPROOF: Always continue, even if context retrieval had issues
        if context is None:
            logger.warning("Context is None, setting to empty string.")
            context = ""
        if not isinstance(context, str):
            logger.warning(f"Context is not a string: {context}")
            context = str(context) if context else ""
        if not isinstance(document_id, str):
            logger.warning(f"Document ID is not a string: {document_id}")
            document_id = str(document_id) if document_id else ""
        try:
            logger.info(f"Calling message_bot with user_input='{user_input}', context length={len(context)}, document_id='{document_id}', database_overview length={len(database_overview) if database_overview else 0}, history length={len(history)}")
            response = message_bot(
                user_input, 
                context, 
                document_id, 
                database_overview,
                history,
            )
            logger.info(f"message_bot response: {response}")
        except Exception as e:
            logger.error(f"Exception in message_bot: {e}")
            response = "Entschuldigung, es ist ein Fehler bei der AI-Verarbeitung aufgetreten."
        if not response or not isinstance(response, str):
            logger.warning(f"Invalid response from message_bot: {response}")
            response = "Entschuldigung, ich konnte keine Antwort generieren."
        try:
            if not chat_state.chat_history:
                chat_state.chat_history = []
            chat_state.chat_history.append({"role": "user", "content": user_input})
            chat_state.chat_history.append({"role": "assistant", "content": response})
            logger.info(f"Updated chat history: {chat_state.chat_history}")
        except Exception as e:
            logger.error(f"Error updating chat history: {e}")

        # NEU: Versuche, die Antwort als JSON zu parsen und Felder direkt zurückzugeben
        import json
        try:
            response_obj = json.loads(response)
            if not isinstance(response_obj, dict):
                response_obj = {"answer": str(response), "document_id": document_id, "source": ""}
        except Exception:
            response_obj = {"answer": str(response), "document_id": document_id, "source": ""}
        final_response = {"status": "success", **response_obj}
        logger.info(f"Returning final response: {final_response}")
        return final_response
    except HTTPException:
        logger.error("HTTPException raised, re-raising.")
        raise
    except Exception as e:
        logger.error(f"Exception in /send_message: {e}")
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


