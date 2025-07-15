from openai import OpenAI
import PyPDF2
from typing import Dict, Any, List, Tuple, Optional
from pinecone import Pinecone
import json
import os
import unicodedata
import re
from pinecone_connection import PineconeCon
from firebase_connection import FirebaseConnection

# Constants
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.3
EMBEDDING_MODEL = "text-embedding-3-small"


class DocProcessor:
    """
    Handles document processing, including PDF extraction, text cleaning, 
    chunking, and storage in vector database with metadata.
    """
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str):
        """
        Initialize DocProcessor with API keys and connections.
        
        Args:
            pinecone_api_key: API key for Pinecone vector database
            openai_api_key: API key for OpenAI services
            
        Note:
            Firebase connection is configured via environment variables:
            - FIREBASE_DATABASE_URL: URL of Firebase Realtime Database
            - FIREBASE_CREDENTIALS_PATH: Path to credentials file (optional)
            - FIREBASE_CREDENTIALS_JSON: JSON string with credentials (optional, for Heroku)
            
        Raises:
            ValueError: If required API keys are missing
        """
        if not pinecone_api_key or not openai_api_key:
            raise ValueError("Both Pinecone and OpenAI API keys are required")
            
        self._openai = OpenAI(api_key=openai_api_key)
        self._pinecone = Pinecone(api_key=pinecone_api_key)
        self._con = PineconeCon("pdfs-index")
        
        try:
            self._firebase = FirebaseConnection()
            self._firebase_available = True
        except ValueError as e:
            pass
            self._firebase_available = False

    
    def get_namespace_data(self, namespace: str) -> List[Dict[str, Any]]:
        """
        Retrieves and formats all document metadata for a given namespace.
        
        Args:
            namespace: Namespace to retrieve metadata from
            
        Returns:
            A clean list of document metadata dictionaries.
        """
        if not self._firebase_available:
            pass
            return []
            
        try:
            # Calls Firebase to get the raw namespace data
            namespace_data = self._firebase.get_namespace_data(namespace)
            
            extracted_data = []
            # Check if the call was successful and data exists
            if namespace_data.get('status') == 'success' and 'data' in namespace_data:
                # The data is a dictionary of documents, convert it to a list
                for doc_id, doc_data in namespace_data['data'].items():
                    if not isinstance(doc_data, dict):
                        continue
                    
                    # Create a clean dictionary for each document
                    doc_info = {
                        'id': doc_id,
                        'name': doc_data.get('name', 'Unknown'),
                        'keywords': doc_data.get('keywords', []),
                        'summary': doc_data.get('summary', ''),
                        'additional_info': doc_data.get('additional_info', '')
                    }
                    extracted_data.append(doc_info)
                    
            return extracted_data
        except Exception as e:
            pass
            return []

    def generate_search_query(self, user_input: str, document_metadata: Dict[str, Any], history: list) -> str:
        """
        Generate an optimized search query for vector database retrieval.
        
        Analyzes user input and document context to create a better query
        for finding relevant document chunks.
        
        Args:
            user_input: User's original question
            document_metadata: Metadata of the selected document
            history: Chat history for context
            
        Returns:
            Optimized query string for vector search
        """
        try:
            prompt = {
                "role": "system",
                "content": """Du bist ein Experte für Informationssuche. Deine Aufgabe ist es, basierend auf einer Nutzerfrage und dem Kontext eines Dokuments, eine optimierte Suchanfrage zu erstellen, die die relevantesten Textabschnitte in einer Vektordatenbank findet.

                    Wichtig:
                    - Ich möchte dass du nur 1-2 Stichwörter aus der Frage extrahierst und diese in die Suchanfrage einsetzt.
                    - Wichtig Zahlen etc. übernehmen
                    z.B. Frage: "Was ist das Modul Programmierung 1?"
                    Antwort: "Programmierung 1"
                    z.B. Frage: "Wie lange dauert der PO-Wechsel?"
                    Antwort: "PO-Wechsel Dauer"
                    z.B. Frage: "Wer ist der Modulverantwortliche für Analysis"
                    Antwort: "Analysis"
                    """
            }
            
            formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-3:]]) if history else "Keine"
            
            user_message = {
                "role": "user", 
                "content": f"""Nutzerfrage: {user_input}

Dokumentkontext:
- Keywords: {document_metadata.get('keywords', 'Keine')}
- Zusammenfassung: {document_metadata.get('summary', 'Keine')}
- Zusätzliche Infos: {document_metadata.get('additional_info', 'Keine')}

Letzte Chat-Nachrichten: {formatted_history}

Erstelle eine optimierte Suchanfrage:"""
            }
            
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[prompt, user_message],
                temperature=0.1,
                max_tokens=100
            )
            
            optimized_query = response.choices[0].message.content.strip()
            
            # Fallback to original query if generation fails or returns empty
            if not optimized_query or len(optimized_query) < 3:
                return user_input
                
            return optimized_query
            
        except Exception as e:
            # Return original query if generation fails
            return user_input



    def appropriate_document_search(self, namespace: str, extracted_data: List[Dict[str, Any]], user_query: str, history: list) -> Optional[Dict[str, Any]]:
        """
        Find the most appropriate document for a user query using AI.
        
        Uses AI to analyze document metadata (keywords, summaries) and match
        them against the user's question to find the most relevant document.
        
        Args:
            namespace: The namespace being searched
            extracted_data: List of document metadata dictionaries
            user_query: User's question or search query
            history: Chat history for context
            
        Returns:
            Dict containing the ID of the best document, or None
            Format: {"id": "document_id"} or {"id": "no_document_found"}
        """
        if not extracted_data:
            return None
            
        if len(extracted_data) == 1:
            return {"id": extracted_data[0]["id"], "name": extracted_data[0]["name"]}

        try:
            prompt = {
                "role": "system", 
                "content": """Du bist ein Assistent, der verschiedene Informationen über Dokumente bekommt. Du sollst entscheiden welches Dokument am besten passt um eine Frage des Nutzers zu beantworten. 

Antworte im JSON-Format mit einem dieser Schemas:
- Für ein Dokument: {"id": "document_id", "name": "document_name"}
- Wenn kein Dokument passt: {"id": "no_document_found", "name": "no_document_found"}

Analysiere die Keywords und Zusammenfassungen der Dokumente und wähle das relevanteste für die Nutzeranfrage aus. 
Beachte dabei die vom Nutzer zu den jeweiligen Dokumenten hinzugefügten Infos.
Beachte die beigefügte Chat History des Nutzers, wenn deiner Meinung nach keine weiteren Informationen benötigt werden aus den Dokumenten, sondern einfach nur weiterführende Fragen gestellt wurden,
dann antworte mit {"id": "no_document_found"}."""
            }
                
            formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) if history else "Keine"
            
            user_message = {
                "role": "user",
                "content": f"Hier sind die verfügbaren Dokumente:\n\n{json.dumps(extracted_data, indent=2, ensure_ascii=False)}\n\nDie Frage des Users lautet: {user_query}\n\nDie Chat History des Users lautet: {formatted_history}\n\nWelches Dokument ist am besten geeignet? \n\n"
            }
            
            # STRUKTURIERTE AUSGABE - Document Selection Debugging
                
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=0.1,  # Low temperature for consistent selection
            )
                
            response_content = response.choices[0].message.content
            
            result = json.loads(response_content)
        
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: return first document
            return {"id": extracted_data[0]["id"]}

    def appropriate_document_search_for_multiple_documents(self, namespace: str, extracted_data: List[Dict[str, Any]], user_query: str, history: list) -> Optional[Dict[str, Any]]:
        """
        Find the most appropriate document(s) for a user query using AI.
        
        Uses AI to analyze document metadata (keywords, summaries) and match
        them against the user's question to find the most relevant document(s).
        Can now return multiple documents when needed.
        
        Args:
            namespace: The namespace being searched
            extracted_data: List of document metadata dictionaries
            user_query: User's question or search query
            history: Chat history for context
            
        Returns:
            Dict containing the ID(s) of the best document(s) and reasoning, or None
            Format: {"id": "single_document_id"} or {"ids": ["doc1", "doc2", ...]} or {"id": "no_document_found"}
        """
        if not extracted_data:
            return None
            
        if len(extracted_data) == 1:
            return {"id": extracted_data[0]["id"]}

        try:
            prompt = {
                "role": "system", 
                "content": """Du bist ein Assistent, der verschiedene Informationen über Dokumente bekommt. Du sollst entscheiden welche Dokumente am besten passen um eine Frage des Nutzers zu beantworten. 

WICHTIG: Du kannst jetzt auch MEHRERE Dokumente auswählen, wenn die Frage Informationen aus verschiedenen Dokumenten benötigt.

Antworte im JSON-Format mit einem dieser Schemas:
- Für EIN Dokument: {"id": "document_id"}
- Für MEHRERE Dokumente: {"ids": ["document_id1", "document_id2", ...]}
- Wenn kein Dokument passt: {"id": "no_document_found"}

WANN MEHRERE DOKUMENTE WÄHLEN:
- Wenn die Frage verschiedene Themenbereiche umfasst
- Wenn Informationen aus verschiedenen Dokumenten kombiniert werden müssen  
- Wenn eine umfassende Antwort mehrere Quellen benötigt
- Maximal 3 Dokumente gleichzeitig auswählen

WANN EIN DOKUMENT WÄHLEN:
- Wenn die Frage spezifisch zu einem Themenbereich gehört
- Wenn ein Dokument alle nötigen Informationen enthält

Analysiere die Keywords und Zusammenfassungen der Dokumente und wähle das/die relevanteste(n) für die Nutzeranfrage aus. 
Beachte dabei die vom Nutzer zu den jeweiligen Dokumenten hinzugefügten Infos.
Beachte die beigefügte Chat History des Nutzers, wenn deiner Meinung nach keine weiteren Informationen benötigt werden aus den Dokumenten, sondern einfach nur weiterführende Fragen gestellt wurden,
dann antworte mit {"id": "no_document_found"}."""
            }
                
            formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history]) if history else "Keine"
            
            user_message = {
                "role": "user",
                "content": f"Hier sind die verfügbaren Dokumente:\n\n{json.dumps(extracted_data, indent=2, ensure_ascii=False)}\n\nDie Frage des Users lautet: {user_query}\n\nDie Chat History des Users lautet: {formatted_history}\n\nWelche(s) Dokument(e) ist/sind am besten geeignet? \n\n"
            }
            
            # STRUKTURIERTE AUSGABE - Document Selection Debugging
           
                
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=0.1,  # Low temperature for consistent selection
            )
                
            response_content = response.choices[0].message.content
            
            result = json.loads(response_content)
        
            
            return result
            
        except (json.JSONDecodeError, Exception) as e:
            # Fallback: return first document
            return {"id": extracted_data[0]["id"]}

    