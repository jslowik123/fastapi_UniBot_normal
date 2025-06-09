from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pinecone
import PyPDF2
from dotenv import load_dotenv
import os
import uvicorn
from pinecone_connection import PineconeCon
from chatbot import get_bot, message_bot, message_bot_stream
from doc_processor import DocProcessor
import tempfile
from firebase_connection import FirebaseConnection
from celery_app import test_task, celery
from tasks import process_document
from redis import Redis
import time
import json
import asyncio
from celery.exceptions import Ignore

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
async def upload_file(file: UploadFile = File(...), namespace: str = Form(...), fileID: str = Form(...)):
    try:
        content = await file.read()
        task = process_document.delay(content, namespace, fileID, file.filename)
        
        return {
            "status": "success",
            "message": "File upload started",
            "task_id": task.id,
            "filename": file.filename
        }
    except Exception as e:
        return {"status": "error", "message": f"Error processing file: {str(e)}", "filename": file.filename}

@app.post("/delete")
async def delete_file(file_name: str = Form(...), namespace: str = Form(...), fileID: str = Form(...), just_firebase: str = Form(...)):
    try:
        if just_firebase.lower() == "true":
            con.delete_embeddings(file_name, namespace)
        
            firebase = FirebaseConnection()
            firebase_result = firebase.delete_document_metadata(namespace, fileID)
        
            return {
                "status": "success", 
                "message": f"File {file_name} deleted successfully",
                "pinecone_status": "success",
                "firebase_status": firebase_result["status"],
                "firebase_message": firebase_result["message"]
                }
        else:
            firebase = FirebaseConnection()
            firebase_result = firebase.delete_document_metadata(namespace, fileID)
            pinecone_result = con.delete_embeddings(file_name, namespace)

            return {
            "status": "success", 
            "message": f"File {file_name} deleted successfully",
            "pinecone_status": "success",
            "pinecone_message": pinecone_result["status"],
            "firebase_status": firebase_result["status"],
            "firebase_message": firebase_result["message"]
            }
    except Exception as e:
        return {"status": "error", "message": f"Error deleting file: {str(e)}"}



@app.post("/start_bot")
async def start_bot():
    chat_state.chain = get_bot()
    chat_state.chat_history = []
    return {"status": "success", "message": "Bot started successfully"}


@app.post("/send_message")
async def send_message(user_input: str = Form(...), namespace: str = Form(...)):
    if not chat_state.chain:
        return {"status": "error", "message": "Bot not started. Please call /start_bot first"}
    
    try:
        database_overview = doc_processor.get_namespace_data(namespace)
        if not database_overview:
            return {"status": "error", "message": f"No documents found in namespace {namespace}"}
            
        appropiate_document = doc_processor.appropiate_document_search(namespace, database_overview, user_input)
        
        if not appropiate_document or "id" not in appropiate_document:
            return {"status": "error", "message": "Could not find appropriate document for query"}

        results = con.query(
            query=user_input, 
            namespace=namespace, 
            fileID=appropiate_document["id"], 
            num_results=3
        )
        
        # Extract text from results
        context_parts = []
        for match in results.matches:
            if hasattr(match, 'metadata') and 'text' in match.metadata:
                context_parts.append(match.metadata['text'])
        
        if not context_parts:
            return {"status": "error", "message": "No relevant content found for query"}
            
        context = "\n".join(context_parts)
        
        response = message_bot(user_input, context, "",database_overview, appropiate_document["id"], chat_state.chat_history)
        
        chat_state.chat_history.append({"role": "user", "content": user_input})
        chat_state.chat_history.append({"role": "assistant", "content": response})
        
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"Error in send_message: {str(e)}")
        return {"status": "error", "message": f"Error processing message: {str(e)}"}


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



@app.post("/delete_namespace")
async def delete_namespace(namespace: str = Form(...)):
    """
    Delete a namespace from the Pinecone index and Firebase metadata.
    
    Args:
        namespace: Name of the namespace to delete
    """
    try:
        pc = PineconeCon("userfiles")
        pc.delete_namespace(namespace)
        
        firebase = FirebaseConnection()
        firebase_result = firebase.delete_namespace_metadata(namespace)
        
        return {
            "status": "success", 
            "message": f"Namespace {namespace} deleted successfully",
            "pinecone_status": "success",
            "firebase_status": firebase_result["status"],
            "firebase_message": firebase_result["message"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/test_worker")
async def test_worker():
    result = test_task.delay()
    return {"status": "success", "task_id": result.id, "message": "Test task sent to worker"}


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    try:
        task = celery.AsyncResult(task_id)
        print(f"Task state: {task.state}")
        print(f"Task info: {task.info}")
        print(f"Task result: {task.result}")
        
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'PENDING',
                'message': 'Task is waiting for execution',
                'progress': 0
            }
        elif task.state == 'STARTED' or task.state == 'PROCESSING':
            meta = task.info if isinstance(task.info, dict) else {}
            response = {
                'state': task.state,
                'status': 'PROCESSING',
                'message': meta.get('status', 'Processing'),
                'current': meta.get('current', 0),
                'total': meta.get('total', 100),
                'progress': meta.get('current', 0),
                'file': meta.get('file', '')
            }
        elif task.state == 'FAILURE' or task.state == 'REVOKED':
            # Handle the case where task.info might be an Exception
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
            # For completed tasks, we should look at task.result
            result = task.result if isinstance(task.result, dict) else {}
            
            # Check if we have a valid result
            if not result:
                response = {
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
                response = {
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
            response = {
                'state': task.state,
                'status': 'UNKNOWN',
                'message': f'Unknown state: {task.state}',
                'info': str(task.info) if task.info else 'No info available',
                'progress': 0
            }
        
        print(f"Sending response: {response}")
        return response
        
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


@app.post("/send_message_stream")
async def send_message_stream(user_input: str = Form(...), namespace: str = Form(...)):
    """
    Streaming version of send_message that sends Server-Sent Events with real-time AI streaming
    """
    if not chat_state.chain:
        # Send error event
        async def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Bot not started. Please call /start_bot first'})}\n\n"
        
        return StreamingResponse(
            error_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    async def generate_response():
        try:
            # Send initial processing event
            yield f"data: {json.dumps({'type': 'chunk', 'content': ''})}\n\n"
            await asyncio.sleep(0.1)
            
            # Get namespace data
            extracted_namespace_data = doc_processor.get_namespace_data(namespace)
            if not extracted_namespace_data:
                yield f"data: {json.dumps({'type': 'error', 'message': f'No documents found in namespace {namespace}'})}\n\n"
                return
            
            # Find appropriate document
            appropiate_document = doc_processor.appropiate_document_search(namespace, extracted_namespace_data, user_input)
            if not appropiate_document or "id" not in appropiate_document:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Could not find appropriate document for query'})}\n\n"
                return
            
            # Query Pinecone
            results = con.query(
                query=user_input, 
                namespace=namespace, 
                fileID=appropiate_document["id"], 
                num_results=3
            )
            
            # Extract context
            context_parts = []
            for match in results.matches:
                if hasattr(match, 'metadata') and 'text' in match.metadata:
                    context_parts.append(match.metadata['text'])
            
            if not context_parts:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant content found for query'})}\n\n"
                return
                
            context = "\n".join(context_parts)
            
            # Stream the response from AI in real-time
            accumulated_response = ""
            
            # Use the streaming function
            for chunk in message_bot_stream(user_input, context, "", extracted_namespace_data, appropiate_document["id"], chat_state.chat_history):
                accumulated_response += chunk
                
                # Send chunk event with only the new chunk (not accumulated)
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the client
            
            # Update chat history after streaming is complete
            chat_state.chat_history.append({"role": "user", "content": user_input})
            chat_state.chat_history.append({"role": "assistant", "content": accumulated_response})
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'fullResponse': accumulated_response})}\n\n"
            
        except Exception as e:
            print(f"Error in send_message_stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error processing message: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=120,
        timeout_graceful_shutdown=120
    )