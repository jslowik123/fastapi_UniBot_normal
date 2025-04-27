from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pinecone
import PyPDF2
from dotenv import load_dotenv
import os
import uvicorn
from pinecon_con import PineconeCon
from chatbot import get_bot, message_bot

load_dotenv()

# Load Pinecone API Key
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pc = pinecone.Pinecone(api_key=pinecone_api_key)
con = PineconeCon("quickstart")

# Chat State
class ChatState:
    def __init__(self):
        self.chain = None
        self.chat_history = []

chat_state = ChatState()

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Uni Chatbot API", "status": "online", "version": "1.0.0"}

# Upload PDF file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...)):
    try:
        pdf = PyPDF2.PdfReader(file.file)
        text = "".join(page.extract_text() for page in pdf.pages)
        data = [{
            'file': file.filename,
            'content': text
        }]
        embedding = con.create_embeddings(data)
        con.upload_embeddings(data, embedding, namespace=namespace)

        return {"status": "success", "message": f"File {file.filename} uploaded and processed successfully", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": f"Error processing file: {str(e)}", "filename": file.filename}

# Delete a specific file's embeddings
@app.post("/delete")
async def delete_file(file_name: str = Form(...), namespace: str = Form(...)):
    con.delete_embeddings(file_name, namespace)
    return {"status": "success", "message": f"File {file_name} deleted successfully"}

# Query Pinecone
@app.post("/query")
async def search_query(query: str = Form(...)):
    results = con.query(query)
    return {
        "status": "success",
        "results": [
            {"text": match["metadata"]["text"], "score": match["score"], "file": match["metadata"]["file"]}
            for match in results
        ]
    }

# Start the chatbot
@app.post("/start_bot")
async def start_bot():
    chat_state.chain = get_bot()
    chat_state.chat_history = []
    return {"status": "success", "message": "Bot started successfully"}

# Send a message to chatbot
@app.post("/send_message")
async def send_message(user_input: str = Form(...), namespace: str = Form(...)):
    if not chat_state.chain:
        return {"status": "error", "message": "Bot not started. Please call /start_bot first"}

    results = con.query(user_input, namespace=namespace)
    context = "\n".join([match["text"] for match in results])

    response = message_bot(user_input, context, chat_state.chat_history)

    chat_state.chat_history.append({"role": "user", "content": user_input})
    chat_state.chat_history.append({"role": "assistant", "content": response})

    return {"status": "success", "response": response}

# Create a namespace
@app.post("/create_namespace")
async def create_namespace(namespace: str = Form(...), dimension: int = Form(1024)):
    result = con.create_namespace_with_dummy(namespace, dimension)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

# Delete all vectors in a namespace
@app.post("/delete_all")
async def delete_all_vectors(namespace: str = Form(...)):
    result = con.delete_all(namespace)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

# Delete a namespace
@app.post("/delete_namespace")
async def delete_namespace(namespace: str = Form(...)):
    try:
        delete_result = con.delete_namespace(namespace)
        if delete_result["status"] == "error":
            raise HTTPException(status_code=400, detail=delete_result["message"])
        return {"status": "success", "message": f"Namespace {namespace} deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Error deleting namespace: {str(e)}"}

# Test endpoint
@app.get("/test")
def read_root():
    return {"message": "Hello, test"}

# Run local server
def run_locally():
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)

if __name__ == "__main__":
    run_locally()
