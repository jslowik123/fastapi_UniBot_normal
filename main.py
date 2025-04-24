from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pinecone
import PyPDF2
from dotenv import load_dotenv
import os
import uvicorn
from pinecon_con import PineconeCon
from chatbot import get_bot, message_bot
from typing import Dict, List, Optional
from pydantic import BaseModel, constr

load_dotenv()

# Get Pinecone API key from environment
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

pc = pinecone.Pinecone(api_key=pinecone_api_key)

con = PineconeCon("quickstart")

# Chat state management
class ChatState:
    def __init__(self):
        self.chain = None
        self.chat_history: List[Dict[str, str]] = []

chat_state = ChatState()

class MessageResponse(BaseModel):
    status: str
    response: Optional[str] = None
    message: Optional[str] = None

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str

class QueryResponse(BaseModel):
    status: str
    results: List[Dict[str, str]]

class NamespaceResponse(BaseModel):
    status: str
    message: str
    namespace: str
    vector_id: Optional[str] = None
    dimension: Optional[int] = None

class DeleteAllResponse(BaseModel):
    status: str
    message: str

# Request Models
class UploadRequest(BaseModel):
    namespace: str

class DeleteRequest(BaseModel):
    file_name: str
    namespace: str

class QueryRequest(BaseModel):
    query: str

class MessageRequest(BaseModel):
    user_input: str

class NamespaceRequest(BaseModel):
    namespace: str
    dimension: int = 1024

class DeleteAllRequest(BaseModel):
    namespace: str = "ns1"

class DeleteNamespaceRequest(BaseModel):
    namespace: str

@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    """
    return {
        "message": "Welcome to the Uni Chatbot API",
        "status": "online",
        "version": "1.0.0"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile, request: UploadRequest):
    """
    Upload a PDF file and process its content into Pinecone.
    
    Args:
        file: The PDF file to upload
        request: UploadRequest containing the namespace
        
    Returns:
        UploadResponse with status and file information
    """
    try:
        pdf = PyPDF2.PdfReader(file.file)
        text = "".join(page.extract_text() for page in pdf.pages)
        data = [{
            'file': file.filename,
            'content': text
        }]

        embedding = con.create_embeddings(data)
        con.upload_embeddings(data, embedding, request.namespace)

        return UploadResponse(
            status="success",
            message=f"File {file.filename} uploaded and processed successfully",
            filename=file.filename
        )
    except Exception as e:
        return UploadResponse(
            status="error",
            message=f"Error processing file: {str(e)}",
            filename=file.filename
        )

@app.post("/delete")
async def delete_file(request: DeleteRequest):
    """
    Delete a file's embeddings from Pinecone.
    
    Args:
        request: DeleteRequest containing file_name and namespace
    """
    con.delete_embeddings(request.file_name, request.namespace)
    return {
        "status": "success",
        "message": f"File {request.file_name} deleted successfully",
        "filename": request.file_name
    }

@app.post("/query", response_model=QueryResponse)
async def search_query(request: QueryRequest):
    """
    Search the Pinecone index for relevant content.
    
    Args:
        request: QueryRequest containing the search query
        
    Returns:
        QueryResponse with matching results
    """
    results = con.query(request.query)
    
    return QueryResponse(
        status="success",
        results=[
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "file": match["metadata"]["file"]
            }
            for match in results
        ]
    )
    
@app.post("/start_bot")
async def start_bot():
    """
    Initialize the chatbot and reset chat history.
    """
    chat_state.chain = get_bot()
    chat_state.chat_history = []
    return {"status": "success", "message": "Bot started successfully"}

@app.post("/send_message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """
    Send a message to the chatbot and get a response.
    
    Args:
        request: MessageRequest containing the user's message
        
    Returns:
        MessageResponse with the bot's response or error message
    """
    if not chat_state.chain:
        return MessageResponse(
            status="error", 
            message="Bot not started. Please call /start_bot first"
        )
    
    # Get context from Pinecone
    results = con.query(request.user_input)
    context = "\n".join([match["text"] for match in results])
    
    # Get bot response
    response = message_bot(request.user_input, context, chat_state.chat_history)
    
    # Update chat history
    chat_state.chat_history.append({"role": "user", "content": request.user_input})
    chat_state.chat_history.append({"role": "assistant", "content": response})
    
    return MessageResponse(
        status="success",
        response=response
    )

@app.post("/create_namespace", response_model=NamespaceResponse)
async def create_namespace(request: NamespaceRequest):
    """
    Create a new namespace in Pinecone with a dummy vector.
    
    Args:
        request: NamespaceRequest containing namespace and dimension
        
    Returns:
        NamespaceResponse with creation status and details
    """
    result = con.create_namespace_with_dummy(request.namespace, request.dimension)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return NamespaceResponse(**result)

@app.post("/delete_all", response_model=DeleteAllResponse)
async def delete_all_vectors(request: DeleteAllRequest):
    """
    Delete all vectors from a namespace.
    
    Args:
        request: DeleteAllRequest containing the namespace
        
    Returns:
        DeleteAllResponse with operation status
    """
    result = con.delete_all(request.namespace)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return DeleteAllResponse(**result)

@app.post("/delete_namespace", response_model=DeleteAllResponse)
async def delete_namespace(request: DeleteNamespaceRequest):
    """
    Delete a namespace and all its associated files from Pinecone.
    
    Args:
        request: DeleteNamespaceRequest containing the namespace to delete
        
    Returns:
        DeleteAllResponse with operation status
    """
    try:
        # First delete all vectors in the namespace
        delete_result = con.delete_all(request.namespace)
        if delete_result["status"] == "error":
            raise HTTPException(status_code=400, detail=delete_result["message"])
            
        # Then delete the namespace itself
        namespace_result = con.delete_namespace(request.namespace)
        if namespace_result["status"] == "error":
            raise HTTPException(status_code=400, detail=namespace_result["message"])
            
        return DeleteAllResponse(
            status="success",
            message=f"Namespace {request.namespace} and all its files deleted successfully"
        )
    except Exception as e:
        return DeleteAllResponse(
            status="error",
            message=f"Error deleting namespace: {str(e)}"
        )

@app.get("/test")
def read_root():
    """
    Test endpoint for basic API functionality.
    """
    return {"message": "Hello, test"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)


    