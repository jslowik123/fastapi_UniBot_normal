from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pinecone
import PyPDF2
from dotenv import load_dotenv
import os
import uvicorn
from pinecone_connection import PineconeCon
from chatbot import get_bot, message_bot
from doc_processor import DocProcessor
import tempfile

load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pc = pinecone.Pinecone(api_key=pinecone_api_key)
con = PineconeCon("userfiles")
doc_processor = DocProcessor(pinecone_api_key, openai_api_key)



class ChatState:
    def __init__(self):
        self.chain = None
        self.chat_history = []

chat_state = ChatState()


@app.get("/")
async def root():
    return {"message": "Welcome to the Uni Chatbot API", "status": "online", "version": "1.0.0"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...)):
    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        
        result = doc_processor.process_pdf(temp_file_path, namespace)
        
        
        os.unlink(temp_file_path)
        
        return result
    except Exception as e:
        return {"status": "error", "message": f"Error processing file: {str(e)}", "filename": file.filename}

@app.post("/delete")
async def delete_file(file_name: str = Form(...), namespace: str = Form(...)):
    con.delete_embeddings(file_name, namespace)
    return {"status": "success", "message": f"File {file_name} deleted successfully"}


@app.post("/query")
async def search_query(query: str = Form(...), namespace: str = Form(...)):
    results = con.query(query, namespace=namespace)
    return {
        "status": "success",
        "results": results
    }
    

@app.post("/start_bot")
async def start_bot():
    chat_state.chain = get_bot()
    chat_state.chat_history = []
    return {"status": "success", "message": "Bot started successfully"}


@app.post("/send_message")
async def send_message(user_input: str = Form(...), namespace: str = Form(...)):
    if not chat_state.chain:
        return {"status": "error", "message": "Bot not started. Please call /start_bot first"}
    
    results = con.query(user_input, namespace=namespace)
    knowledge = con.query(user_input, namespace="knowledge")

    context = "\n".join([match["text"] for match in results])
    knowledge = "\n".join([match["text"] for match in knowledge])
    
    response = message_bot(user_input, context, knowledge, chat_state.chat_history)
    
    chat_state.chat_history.append({"role": "user", "content": user_input})
    chat_state.chat_history.append({"role": "assistant", "content": response})
    
    return {"status": "success", "response": response}


@app.post("/create_namespace")
async def create_namespace(namespace: str = Form(...), dimension: int = Form(1536)):
    """
    Create a new namespace in the Pinecone index.
    
    Args:
        namespace: Name of the namespace to create
        dimension: Dimension of the vectors (default: 1536 for OpenAI embeddings)
    """
    try:
        
        pc = PineconeCon("userfiles")
        
        
        pc.create_namespace(namespace)
        
        return {"status": "success", "message": f"Namespace {namespace} created successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/delete_all")
async def delete_all_vectors(namespace: str = Form(...)):
    result = con.delete_all(namespace)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/delete_namespace")
async def delete_namespace(namespace: str = Form(...)):
    """
    Delete a namespace from the Pinecone index.
    
    Args:
        namespace: Name of the namespace to delete
    """
    try:
        
        pc = PineconeCon("userfiles")
        
        
        pc.delete_namespace(namespace)
        
        return {"status": "success", "message": f"Namespace {namespace} deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )
    