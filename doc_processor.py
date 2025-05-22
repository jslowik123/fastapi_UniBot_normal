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

class DocProcessor:
    def __init__(self, pinecone_api_key: str, openai_api_key: str):
        """
        Initialize DocProcessor with API keys.
        
        Args:
            pinecone_api_key: API key for Pinecone
            openai_api_key: API key for OpenAI
            
        Hinweis: Firebase-Verbindung wird über Umgebungsvariablen konfiguriert:
            - FIREBASE_DATABASE_URL: URL der Firebase Realtime Database
            - FIREBASE_CREDENTIALS_PATH: Pfad zur Credentials-Datei (optional)
            - FIREBASE_CREDENTIALS_JSON: JSON-String mit Credentials (optional, für Heroku)
        """
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
        Process a PDF file and store its content in Pinecone.
        
        Args:
            file_path: Path to the PDF file
            namespace: Pinecone namespace to use
            fileID: ID to use for storing document metadata
            
        Returns:
            Dict containing processing results
        """
        try:
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

            file_name = os.path.basename(file_path)
            cleaned_text, keywords, summary = self._recondition_text(text)
            
            chunks = self._split_text(cleaned_text)
            
            pinecone_result = self._con.upload(chunks, namespace, file_name, fileID=fileID)

            firebase_result = None
            if self._firebase_available:
                firebase_result = self._firebase.append_metadata(
                    namespace=namespace,
                    fileID=fileID,
                    chunk_count=len(chunks),
                    keywords=keywords,
                    summary=summary
                )
            else:
                firebase_result = {
                    'status': 'error',
                    'message': 'Firebase nicht verfügbar'
                }
            
            return {
                "status": "success",
                "message": f"File {file_name} processed successfully",
                "chunks": len(chunks),
                "pinecone_result": pinecone_result,
                "firebase_result": firebase_result,
                "original_file": file_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def process_pdf_bytes(self, pdf_file, namespace: str, fileID: str, file_name: str) -> Dict[str, Any]:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            cleaned_text, keywords, summary = self._recondition_text(text)
            
            chunks = self._split_text(cleaned_text)
            
            pinecone_result = self._con.upload(chunks, namespace, file_name, fileID=fileID)

            firebase_result = None
            if self._firebase_available:
                firebase_result = self._firebase.append_metadata(
                    namespace=namespace,
                    fileID=fileID,
                    chunk_count=len(chunks),
                    keywords=keywords,
                    summary=summary
                )
            else:
                firebase_result = {
                    'status': 'error',
                    'message': 'Firebase nicht verfügbar'
                }
            
            return {
                "status": "success",
                "message": f"File {file_name} processed successfully",
                "chunks": len(chunks),
                "pinecone_result": pinecone_result,
                "firebase_result": firebase_result,
                "original_file": file_name
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _recondition_text(self, text: str) -> Tuple[str, List[str], str]:
        """
        Use OpenAI to clean the text and extract keywords/topics.
        
        Args:
            text: The raw text extracted from the PDF
            
        Returns:
            Tuple containing (cleaned text, list of keywords/topics, summary)
        """
        prompt = {
            "role": "system", 
            "content": "Du bist ein Assistent zur Textbereinigung und Inhaltsanalyse. Du erhältst einen Text aus einem PDF und sollst drei Aufgaben erfüllen: 1) Den Text bereinigen und gut formatieren (Absätze erhalten, überflüssige Zeilenumbrüche entfernen, Formatierungsfehler korrigieren), 2) Die wichtigsten Schlagwörter und Themen aus dem Text extrahieren. Die Keywords sollen nicht mehr als 3 Wörter lang sein. 3) Eine kurze Zusammenfassung des Dokuments in 3-5 Sätzen erstellen. Gib das Ergebnis als JSON mit den Feldern 'cleaned_text', 'keywords' und 'summary' zurück."
        }
        
        user_message = {
            "role": "user",
            "content": f"Hier ist der Text aus einem PDF. Bitte bereinige den Text, extrahiere die Schlagwörter/Themen und erstelle eine kurze Zusammenfassung:\n\n{text}"
        }
        
        response = self._openai.chat.completions.create(
            model="gpt-4.1-nano",
            response_format={"type": "json_object"},
            messages=[prompt, user_message],
            temperature=0.3,
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("cleaned_text", ""), result.get("keywords", []), result.get("summary", "")

    def _split_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters.
        
        Args:
            text: Text to split
            chunk_size: Target size for each chunk
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        sentences = text.replace('\n', ' ').split('. ')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def get_namespace_data(self, namespace: str) -> str:
        namespace_data = self._firebase.get_namespace_data(namespace)
        
        extracted_data = []
        if namespace_data['status'] == 'success' and 'data' in namespace_data:
            for doc_id, doc_data in namespace_data['data'].items():
            # Überspringe nicht-Dokument Einträge
                if not isinstance(doc_data, dict) or 'keywords' not in doc_data:
                    continue
                
            # Extrahiere die gewünschten Informationen
                doc_info = {
                    'id': doc_id,
                    'name': doc_data.get('name', 'Unbekannt'),
                    'keywords': doc_data.get('keywords', []),
                    'summary': doc_data.get('summary', ''),
                    'chunk_count': doc_data.get('chunk_count', 0)
                }
                extracted_data.append(doc_info)
        return extracted_data

    def appropiate_document_search(self, namespace: str, extracted_data: str, user_query: str) -> Dict:
    

        prompt = {
                "role": "system", 
                "content": "Du bist ein Assistent, der verschiedene Informationen über Dokumente bekommt. Du sollst entscheiden welches Dokument am besten passt um eine Frage des Nutzers zu beantworten. Antworte im JSON-Format mit genau diesem Schema: {\"id\": \"document_id\"}. Verwende keine anderen Felder und füge keine Erklärungen hinzu."
            }
            
        user_message = {
                "role": "user",
                "content": f"Hier sind die Themen der Dokumente:\n\n{extracted_data}. Bitte antworte im JSON-Format, indem du nur die ID des geeigneten Dokuments zurückgibst. Die Frage des Users lautet: {user_query}"
            }
            
        response = self._openai.chat.completions.create(
                model="gpt-4.1-nano",
                response_format={"type": "json_object"},
                messages=[prompt, user_message],
                temperature=0.1,  # Lower temperature for more predictable output
            )
            
        response_content = response.choices[0].message.content
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Original content: {response_content}")
            return {"id": extracted_data[0]["id"] if extracted_data else "default", "chunk_count": 5}
        
    def generate_global_summary(self, documents_data: List[Dict[str, Any]]) -> str:
        """
        Generates a global summary from a list of document data (summaries or texts).
        
        Args:
            documents_data: A list of dictionaries, where each dictionary represents a document
                            and should contain a 'summary' or 'text' key.
                            
        Returns:
            A string containing the global summary in German bullet points.
        """
        if not documents_data:
            return "Keine Dokumente zum Zusammenfassen gefunden."

        combined_text = ""
        for doc in documents_data:
            content = doc.get('summary')
            if not content:
                content = doc.get('cleaned_text') 
            if not content:
                 content = doc.get('text') 
            
            if content:
                combined_text += content + "\n\n---\n\n" # Add a clear separator
        
        if not combined_text.strip():
            return "Kein Inhalt aus Dokumenten verfügbar, um eine Zusammenfassung zu erstellen."

        prompt = {
            "role": "system", 
            "content": "Du bist ein Assistent, der eine prägnante globale Zusammenfassung auf Deutsch und in Stichpunkten erstellen soll. Du erhältst eine Sammlung von Texten oder Zusammenfassungen aus mehreren Dokumenten derselben Kategorie oder desselben Namespace. Dein Ziel ist es, diese Informationen zu einer einzigen, kohärenten Übersicht zusammenzufassen, die die Hauptthemen und Schlüsselinformationen aller Dokumente erfasst. Präsentiere diese Übersicht als eine Reihe von Stichpunkten. Jeder Stichpunkt sollte prägnant und informativ sein. Konzentriere dich auf Gemeinsamkeiten und wichtige Unterschiede, die in den bereitgestellten Texten zu finden sind. Antworte ausschließlich auf Deutsch."
        }
        
        user_message = {
            "role": "user",
            "content": f"Bitte erstelle eine globale Zusammenfassung auf Deutsch und in Stichpunkten basierend auf den folgenden Dokumentinhalten. Stelle sicher, dass jeder Hauptpunkt ein separater Stichpunkt ist:\n\n{combined_text}"
        }
        
        try:
            response = self._openai.chat.completions.create(
                model="gpt-4.1-nano", 
                messages=[prompt, user_message],
                temperature=0.3, # Lower temperature for more factual, structured output
            )
            global_summary = response.choices[0].message.content
            return global_summary.strip()
        except Exception as e:
            print(f"Error generating global summary: {str(e)}")
            return "Fehler beim Erstellen der globalen Zusammenfassung."

