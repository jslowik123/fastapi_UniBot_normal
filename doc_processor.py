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
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MODEL = "gpt-4.1-nano"
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
        self._con = PineconeCon("userfiles")
        
        try:
            self._firebase = FirebaseConnection()
            self._firebase_available = True
        except ValueError as e:
            print(f"Firebase nicht verfügbar: {e}")
            self._firebase_available = False

    def process_pdf(self, file_path: str, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Process a PDF file from disk and store its content in vector database.
        
        Args:
            file_path: Path to the PDF file on disk
            namespace: Pinecone namespace for organizing documents
            fileID: Unique identifier for the document
            
        Returns:
            Dict containing processing results with status, message, and metadata
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: For various processing errors
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = self._extract_text_from_pdf_reader(pdf_reader)

            file_name = os.path.basename(file_path)
            return self._process_extracted_text(text, namespace, fileID, file_name)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing PDF file: {str(e)}"
            }
    
    def process_pdf_bytes(self, pdf_file, namespace: str, fileID: str, file_name: str) -> Dict[str, Any]:
        """
        Process a PDF file from bytes (e.g., uploaded file) and store content in vector database.
        
        Args:
            pdf_file: PDF file as bytes or file-like object
            namespace: Pinecone namespace for organizing documents
            fileID: Unique identifier for the document
            file_name: Original filename of the document
            
        Returns:
            Dict containing processing results with status, message, and metadata
        """
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = self._extract_text_from_pdf_reader(pdf_reader)
            
            return self._process_extracted_text(text, namespace, fileID, file_name)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing PDF bytes: {str(e)}"
            }
    
    def _extract_text_from_pdf_reader(self, pdf_reader: PyPDF2.PdfReader) -> str:
        """
        Extract text from a PDF reader object.
        
        Args:
            pdf_reader: PyPDF2 PdfReader instance
            
        Returns:
            str: Extracted text from all pages
        """
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def _process_extracted_text(self, text: str, namespace: str, fileID: str, file_name: str) -> Dict[str, Any]:
        """
        Process extracted text by cleaning, chunking, and storing in databases.
        
        Args:
            text: Raw text extracted from PDF
            namespace: Namespace for organizing documents
            fileID: Unique identifier for the document  
            file_name: Original filename
            
        Returns:
            Dict containing processing results
        """
        try:
            cleaned_text, keywords, summary, structured_sections = self._recondition_text(text)
            chunks = self._split_text(cleaned_text)
            
            pinecone_result = self._con.upload(chunks, namespace, file_name, fileID=fileID)

            firebase_result = self._store_metadata(namespace, fileID, len(chunks), keywords, summary, structured_sections)
            
            return {
                "status": "success",
                "message": f"File {file_name} processed successfully",
                "chunks": len(chunks),
                "pinecone_result": pinecone_result,
                "firebase_result": firebase_result,
                "original_file": file_name,
                "structured_sections": structured_sections
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in text processing pipeline: {str(e)}"
            }
    
    def _store_metadata(self, namespace: str, fileID: str, chunk_count: int, 
                       keywords: List[str], summary: str, structured_sections: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Store document metadata in Firebase if available.
        
        Args:
            namespace: Document namespace
            fileID: Document identifier  
            chunk_count: Number of text chunks created
            keywords: Extracted keywords
            summary: Document summary
            structured_sections: List of structured sections
            
        Returns:
            Dict with storage result status
        """
        if self._firebase_available:
            return self._firebase.append_metadata(
                namespace=namespace,
                fileID=fileID,
                chunk_count=chunk_count,
                keywords=keywords,
                summary=summary,
                structured_sections=structured_sections
            )
        else:
            return {
                'status': 'error',
                'message': 'Firebase nicht verfügbar'
            }
    
    def _recondition_text(self, text: str) -> Tuple[str, List[str], str, List[Dict[str, str]]]:
        """
        Clean text and extract keywords and summary using OpenAI.
        
        This method uses AI to:
        1. Clean and format the text (fix line breaks, formatting errors)
        2. Extract relevant keywords and topics (max 3 words each)
        3. Generate a concise summary (3-5 sentences)
        
        Args:
            text: Raw text extracted from PDF
            
        Returns:
            Tuple containing (cleaned_text, keywords_list, summary)
            
        Raises:
            Exception: If OpenAI API call fails or returns invalid JSON
        """
        if not text or not text.strip():
            return "", [], ""
            
        prompt = {
            "role": "system", 
            "content": """Du bist ein Assistent zur Textbereinigung und Strukturierung. Du erhältst einen Text aus einem PDF und sollst vier Aufgaben erfüllen:

1) Den Text bereinigen und gut formatieren (überflüssige Zeilenumbrüche entfernen, Formatierungsfehler korrigieren)
2) Den Text in logische Abschnitte unterteilen und passende Überschriften für jeden Abschnitt erstellen
3) Die wichtigsten Schlagwörter und Themen aus dem gesamten Text extrahieren (max. 3 Wörter pro Keyword)
4) Eine kurze Zusammenfassung des gesamten Dokuments in 3-5 Sätzen erstellen

Strukturiere den Text in Abschnitte mit aussagekräftigen Überschriften. Jeder Abschnitt sollte thematisch zusammengehörigen Inhalt haben.
Dabei soll zu jeder Überschrift auch der gesamte Inhalt 1zu1 zurück gegeben werden. Der Gesamte Input text soll also wieder zurück kommen.
Gib das Ergebnis als JSON zurück mit folgender Struktur:
{
  "structured_sections": [
    {
      "title": "Überschrift des Abschnitts",
      "content": "Vollständiger Inhalt des Abschnitts"
    }
  ],
  "keywords": ["keyword1", "keyword2", ...],
  "summary": "Kurze Zusammenfassung des gesamten Dokuments"
}"""
        }
        
        user_message = {
            "role": "user",
            "content": f"Hier ist der Text aus einem PDF. Bitte bereinige den Text, extrahiere die Schlagwörter/Themen und erstelle eine kurze Zusammenfassung:\n\n{text}"
        }
        
        try:
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=DEFAULT_TEMPERATURE,
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extract structured sections and combine them back to text for compatibility
            structured_sections = result.get("structured_sections", [])
            if structured_sections:
                # Combine all sections into one text for backward compatibility
                cleaned_text_parts = []
                for section in structured_sections:
                    title = section.get("title", "")
                    content = section.get("content", "")
                    if title and content:
                        cleaned_text_parts.append(f"## {title}\n\n{content}")
                    elif content:
                        cleaned_text_parts.append(content)
                
                cleaned_text = "\n\n".join(cleaned_text_parts)
            else:
                cleaned_text = result.get("cleaned_text", text)
            
            return (
                cleaned_text,
                result.get("keywords", []), 
                result.get("summary", ""),
                structured_sections  # Return structured sections as additional data
            )
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error in text reconditioning: {str(e)}")
            # Fallback: return original text with empty metadata
            return text, [], "", []

    def _split_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """
        Split text into manageable chunks for vector embedding.
        
        Splits text by sentences to maintain semantic coherence within chunks.
        Each chunk aims for approximately chunk_size characters.
        
        Args:
            text: Text to split into chunks
            chunk_size: Target size for each chunk in characters
            
        Returns:
            List of text chunks, each roughly chunk_size characters
        """
        if not text or not text.strip():
            return []
            
        chunks = []
        current_chunk = ""
        
        # Split by sentences to maintain semantic boundaries
        sentences = text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def get_namespace_data(self, namespace: str) -> List[Dict[str, Any]]:
        """
        Retrieve document metadata for all documents in a namespace.
        
        Args:
            namespace: The namespace to query
            
        Returns:
            List of dictionaries containing document metadata (id, name, keywords, summary, chunk_count)
        """
        if not self._firebase_available:
            print("Firebase not available for namespace data retrieval")
            return []
            
        try:
            namespace_data = self._firebase.get_namespace_data(namespace)
            
            extracted_data = []
            if namespace_data.get('status') == 'success' and 'data' in namespace_data:
                for doc_id, doc_data in namespace_data['data'].items():
                    # Skip non-document entries
                    if not isinstance(doc_data, dict) or 'keywords' not in doc_data:
                        continue
                    
                    doc_info = {
                        'id': doc_id,
                        'name': doc_data.get('name', 'Unknown'),
                        'keywords': doc_data.get('keywords', []),
                        'summary': doc_data.get('summary', ''),
                        'chunk_count': doc_data.get('chunk_count', 0)
                    }
                    extracted_data.append(doc_info)
                    
            return extracted_data
        except Exception as e:
            print(f"Error retrieving namespace data: {str(e)}")
            return []

    def appropriate_document_search(self, namespace: str, extracted_data: List[Dict[str, Any]], 
                                  user_query: str) -> Dict[str, Any]:
        """
        Find the most appropriate document to answer a user's query.
        
        Uses AI to analyze document metadata (keywords, summaries) and match
        them against the user's question to find the most relevant document.
        
        Args:
            namespace: The namespace being searched
            extracted_data: List of document metadata dictionaries
            user_query: User's question or search query
            
        Returns:
            Dict containing the ID of the most appropriate document
        """
        if not extracted_data:
            return {"id": "default"}
            
        if len(extracted_data) == 1:
            return {"id": extracted_data[0]["id"]}

        try:
            prompt = {
                "role": "system", 
                "content": """Du bist ein Assistent, der verschiedene Informationen über Dokumente bekommt. Du sollst entscheiden welches Dokument am besten passt um eine Frage des Nutzers zu beantworten. 

Antworte im JSON-Format mit genau diesem Schema: {"id": "document_id"}. 
Verwende keine anderen Felder und füge keine Erklärungen hinzu.

Analysiere die Keywords und Zusammenfassungen der Dokumente und wähle das relevanteste für die Nutzeranfrage aus."""
            }
                
            user_message = {
                "role": "user",
                "content": f"Hier sind die verfügbaren Dokumente:\n\n{json.dumps(extracted_data, indent=2, ensure_ascii=False)}\n\nDie Frage des Users lautet: {user_query}\n\nWelches Dokument ist am besten geeignet?"
            }
                
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=0.1,  # Low temperature for consistent selection
            )
                
            response_content = response.choices[0].message.content
            return json.loads(response_content)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error in document selection: {str(e)}")
            print(f"Response content: {response_content if 'response_content' in locals() else 'No response'}")
            # Fallback: return first document
            return {"id": extracted_data[0]["id"]}

    def generate_global_summary(self, namespace: str) -> Dict[str, Any]:
        """
        Generate a global summary for all documents in a namespace.
        
        Args:
            namespace: The namespace to summarize
            
        Returns:
            Dict containing the operation status and generated summary
        """
        try:
            namespace_data = self.get_namespace_data(namespace)
            if not namespace_data:
                return {
                    "status": "error",
                    "message": f"No documents found in namespace {namespace}"
                }
            
            # Extract all summaries and keywords
            all_summaries = []
            all_keywords = []
            
            for doc in namespace_data:
                if doc.get('summary'):
                    all_summaries.append(f"Document {doc['name']}: {doc['summary']}")
                if doc.get('keywords'):
                    all_keywords.extend(doc['keywords'])
            
            if not all_summaries:
                return {
                    "status": "error", 
                    "message": "No document summaries available for global summary generation"
                }
            
            # Generate global summary using AI
            prompt = {
                "role": "system",
                "content": """Du erstellst eine globale Zusammenfassung für eine Sammlung von Dokumenten. 
                Erstelle eine kohärente Übersicht über alle Dokumente und extrahiere die wichtigsten Themenbereiche 
                als Bullet Points. Antworte im JSON-Format mit den Feldern 'global_summary' und 'main_topics'."""
            }
            
            user_message = {
                "role": "user", 
                "content": f"Erstelle eine globale Zusammenfassung für diese Dokumente:\n\n{chr(10).join(all_summaries)}\n\nWichtige Keywords: {', '.join(set(all_keywords))}"
            }
            
            response = self._openai.chat.completions.create(
                model=DEFAULT_MODEL,
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=DEFAULT_TEMPERATURE,
            )
            
            result = json.loads(response.choices[0].message.content)
            global_summary = result.get("global_summary", "")
            main_topics = result.get("main_topics", [])
            
            # Store in Firebase if available
            if self._firebase_available:
                self._firebase.update_namespace_summary(namespace, main_topics)
            
            return {
                "status": "success",
                "global_summary": global_summary,
                "main_topics": main_topics,
                "message": f"Global summary generated for namespace {namespace}"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error generating global summary: {str(e)}"
            }
        
    