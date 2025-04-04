from fastapi import FastAPI, UploadFile
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import PyPDF2
from dotenv import load_dotenv
import os
from pinecon_con import PineconeCon
from chatbot import get_bot, message_bot
load_dotenv()

# Get Pinecone API key from environment
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

app = FastAPI()
pc = Pinecone(api_key=pinecone_api_key)
con = PineconeCon("quickstart")

# Global variables for bot state
chat_history = []
chain = None

@app.post("/upload")
async def upload_file(file: UploadFile):
    # Text extrahieren
    pdf = PyPDF2.PdfReader(file.file)
    text = "".join(page.extract_text() for page in pdf.pages)

    # Prepare data with filename
    data = [{
        'file': file.filename,
        'content': [text]
    }]

    embedding = con.create_embeddings(data)
    con.upload_embeddings(data, embedding)

    return {
        "status": "success",
        "message": f"File {file.filename} uploaded and processed successfully",
        "filename": file.filename
    }

@app.post("/delete")
async def delete_file(file_name: str):
    con.delete_embeddings(file_name)
    return {
        "status": "success",
        "message": f"File {file_name} deleted successfully",
        "filename": file_name
    }

@app.get("/query")
async def search_query(query: str):
    results = con.query(query)
    
    return {
        "status": "success",
        "results": [
            {
                "text": match["metadata"]["text"],
                "score": match["score"],
                "file": match["metadata"]["file"]
            }
            for match in results
        ]
    }
    
@app.post("/start_bot")
async def start_bot():
    global chain, chat_history
    chain = get_bot()
    chat_history = []
    return {"status": "success", "message": "Bot started successfully"}

@app.post("/send_message")
async def send_message(user_input: str):
    global chain, chat_history
    if not chain:
        return {
            "status": "error", 
            "message": "Bot not started. Please call /start_bot first"
        }
    
    # Get context from Pinecone
    con = PineconeCon("quickstart")
    results = con.query(user_input)
    context = "\n".join([match["text"] for match in results])
    
    # Get bot response
    response = message_bot(user_input, context, chat_history)
    
    # Update chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})
    
    return {
        "status": "success",
        "response": response,
    }