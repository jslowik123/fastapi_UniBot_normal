import os
import json
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from openai import OpenAI
from dotenv import load_dotenv
from pinecone_connection import PineconeCon
from doc_processor import DocProcessor
from firebase_connection import FirebaseConnection

load_dotenv()


class UniversityAgent:
    """
    OpenAI Assistant für Universitäts-Dokument-Verarbeitung und Chat.
    
    Verwendet die OpenAI Assistants API mit Tools und Function Calling
    für erweiterte Funktionalität.
    """
    
    def __init__(self):
        """Initialize the agent with required connections and configurations."""
        # Initialize OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=openai_api_key)
        
        # Initialize connections
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
            
        self.pinecone_con = PineconeCon("userfiles")
        self.doc_processor = DocProcessor(pinecone_api_key, openai_api_key)
        
        # Agent configuration
        self.model = "gpt-4o-mini"
        self.default_num_results = 3
        
        # Initialize assistant
        self.assistant = None
        self.thread = None
        self._create_assistant()
        
    def _create_assistant(self):
        """Create the OpenAI Assistant with tools."""
        self.assistant = self.client.beta.assistants.create(
            name="University Document Assistant",
            instructions="""Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität. 
            
Deine Hauptaufgaben:
1. Beantworte Fragen zu Universitätsdokumenten (Modulhandbücher, Studienordnungen, etc.)
2. Verwende die verfügbaren Tools um relevante Informationen zu finden
3. Gib strukturierte, hilfreiche Antworten basierend auf den gefundenen Informationen

Wichtige Regeln:
- Verwende IMMER die Tools um relevante Dokumente zu finden
- Stütze deine Antworten auf die gefundenen Informationen
- Sei präzise und verständlich
- Gib die Quelle deiner Informationen an
- Frage nach, wenn die Anfrage unklar ist

Verwende die Tools in dieser Reihenfolge:
1. get_namespace_overview - um verfügbare Dokumente zu sehen
2. find_relevant_document - um das passende Dokument zu finden  
3. search_document_content - um spezifische Inhalte zu finden
4. Beantworte basierend auf den gefundenen Informationen""",
            model=self.model,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_namespace_overview",
                        "description": "Ruft eine Übersicht aller verfügbaren Dokumente in einem Namespace ab",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "Der Namespace für den die Dokumentübersicht abgerufen werden soll"
                                }
                            },
                            "required": ["namespace"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "find_relevant_document",
                        "description": "Findet das relevanteste Dokument für eine Benutzeranfrage",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "Der Namespace in dem gesucht werden soll"
                                },
                                "user_query": {
                                    "type": "string",
                                    "description": "Die Benutzeranfrage für die ein passendes Dokument gefunden werden soll"
                                }
                            },
                            "required": ["namespace", "user_query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_document_content",
                        "description": "Durchsucht den Inhalt eines spezifischen Dokuments nach relevanten Informationen",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "Der Namespace des Dokuments"
                                },
                                "document_id": {
                                    "type": "string",
                                    "description": "Die ID des zu durchsuchenden Dokuments"
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Die Suchanfrage für den Dokumentinhalt"
                                }
                            },
                            "required": ["namespace", "document_id", "query"]
                        }
                    }
                }
            ]
        )
        
    def _execute_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a function call and return the result."""
        try:
            if function_name == "get_namespace_overview":
                namespace = arguments.get("namespace")
                result = self.doc_processor.get_namespace_data(namespace)
                return json.dumps(result, ensure_ascii=False, indent=2)
                
            elif function_name == "find_relevant_document":
                namespace = arguments.get("namespace")
                user_query = arguments.get("user_query")
                
                # Get namespace overview first
                database_overview = self.doc_processor.get_namespace_data(namespace)
                if not database_overview:
                    return json.dumps({"error": f"No documents found in namespace {namespace}"})
                
                # Find relevant document
                result = self.doc_processor.appropriate_document_search(
                    namespace, database_overview, user_query
                )
                
                # Add document details
                if result and "id" in result:
                    for doc in database_overview:
                        if doc["id"] == result["id"]:
                            result.update({
                                "name": doc.get("name", ""),
                                "keywords": doc.get("keywords", []),
                                "summary": doc.get("summary", "")
                            })
                            break
                
                return json.dumps(result, ensure_ascii=False, indent=2)
                
            elif function_name == "search_document_content":
                namespace = arguments.get("namespace")
                document_id = arguments.get("document_id")
                query = arguments.get("query")
                
                # Query vector database
                results = self.pinecone_con.query(
                    query=query,
                    namespace=namespace,
                    fileID=document_id,
                    num_results=self.default_num_results
                )
                
                # Extract relevant content
                content_parts = []
                for match in results.matches:
                    if hasattr(match, 'metadata') and 'text' in match.metadata:
                        content_parts.append({
                            "text": match.metadata['text'],
                            "score": float(match.score),
                            "page": match.metadata.get('page', 'unknown')
                        })
                
                return json.dumps({
                    "document_id": document_id,
                    "query": query,
                    "results": content_parts,
                    "total_results": len(content_parts)
                }, ensure_ascii=False, indent=2)
                
            else:
                return json.dumps({"error": f"Unknown function: {function_name}"})
                
        except Exception as e:
            return json.dumps({"error": f"Error executing {function_name}: {str(e)}"})

    def create_thread(self) -> str:
        """Create a new conversation thread."""
        self.thread = self.client.beta.threads.create()
        return self.thread.id

    def add_message(self, content: str, namespace: str) -> str:
        """Add a message to the current thread."""
        if not self.thread:
            self.create_thread()
        
        # Add namespace context to the message
        enhanced_content = f"[Namespace: {namespace}]\n{content}"
        
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=enhanced_content
        )
        return message.id

    def run_assistant(self, namespace: str) -> Dict[str, Any]:
        """Run the assistant and handle function calls."""
        if not self.thread:
            return {"error": "No thread created"}
        
        # Create and run
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id
        )
        
        # Wait for completion and handle function calls
        while True:
            run_status = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )
            
            if run_status.status == "completed":
                break
            elif run_status.status == "requires_action":
                # Handle function calls
                tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"Executing function: {function_name} with args: {arguments}")
                    
                    # Execute the function
                    result = self._execute_function(function_name, arguments)
                    
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": result
                    })
                
                # Submit tool outputs
                self.client.beta.threads.runs.submit_tool_outputs(
                    thread_id=self.thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs
                )
                
            elif run_status.status in ["failed", "cancelled", "expired"]:
                return {"error": f"Run failed with status: {run_status.status}"}
            
            time.sleep(1)  # Wait before checking again
        
        # Get the assistant's response
        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
            order="desc",
            limit=1
        )
        
        if messages.data:
            response = messages.data[0].content[0].text.value
            return {
                "status": "success",
                "response": response,
                "thread_id": self.thread.id,
                "run_id": run.id
            }
        else:
            return {"error": "No response received"}

    async def run_assistant_stream(self, namespace: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the assistant with streaming support."""
        if not self.thread:
            yield {"type": "error", "message": "No thread created"}
            return
        
        try:
            # Create and run with streaming (if supported)
            run = self.client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id
            )
            
            # Monitor run status
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id,
                    run_id=run.id
                )
                
                if run_status.status == "completed":
                    # Get final response
                    messages = self.client.beta.threads.messages.list(
                        thread_id=self.thread.id,
                        order="desc",
                        limit=1
                    )
                    
                    if messages.data:
                        response = messages.data[0].content[0].text.value
                        yield {
                            "type": "complete",
                            "content": response,
                            "thread_id": self.thread.id,
                            "run_id": run.id
                        }
                    break
                    
                elif run_status.status == "requires_action":
                    yield {"type": "status", "message": "Executing tools..."}
                    
                    # Handle function calls
                    tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)
                        
                        yield {
                            "type": "tool_call",
                            "function": function_name,
                            "arguments": arguments
                        }
                        
                        result = self._execute_function(function_name, arguments)
                        
                        tool_outputs.append({
                            "tool_call_id": tool_call.id,
                            "output": result
                        })
                    
                    # Submit tool outputs
                    self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=self.thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
                    
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    yield {"type": "error", "message": f"Run failed with status: {run_status.status}"}
                    break
                
                await asyncio.sleep(1)  # Wait before checking again
                
        except Exception as e:
            yield {"type": "error", "message": f"Error in stream: {str(e)}"}

    def process_message(self, user_input: str, namespace: str) -> Dict[str, Any]:
        """Process a user message using the assistant."""
        try:
            # Add message to thread
            self.add_message(user_input, namespace)
            
            # Run assistant
            result = self.run_assistant(namespace)
            
            return result
            
        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing message: {str(e)}"
            }

    async def process_message_stream(self, user_input: str, namespace: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a user message using the assistant with streaming."""
        try:
            # Add message to thread
            self.add_message(user_input, namespace)
            
            # Run assistant with streaming
            async for chunk in self.run_assistant_stream(namespace):
                yield chunk
                
        except Exception as e:
            print(f"Error in process_message_stream: {str(e)}")
            yield {
                "type": "error",
                "message": f"Error processing message: {str(e)}"
            }

    def reset_conversation(self):
        """Reset the conversation by creating a new thread."""
        self.thread = None
        self.create_thread()

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history from the current thread."""
        if not self.thread:
            return []
        
        try:
            messages = self.client.beta.threads.messages.list(
                thread_id=self.thread.id,
                order="asc"
            )
            
            history = []
            for message in messages.data:
                history.append({
                    "role": message.role,
                    "content": message.content[0].text.value,
                    "created_at": message.created_at
                })
            
            return history
        except Exception as e:
            print(f"Error getting conversation history: {str(e)}")
            return []

    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return {
            "assistant_id": self.assistant.id if self.assistant else None,
            "thread_id": self.thread.id if self.thread else None,
            "model": self.model,
            "default_num_results": self.default_num_results,
            "tools_available": len(self.assistant.tools) if self.assistant else 0,
            "status": "ready"
        } 