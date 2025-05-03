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
            cleaned_text, keywords, summary = self._clean_text(text)
            
            chunks = self._split_text(cleaned_text)
            
            pinecone_result = self._con.upload(chunks, namespace, file_name, fileID=fileID)

            firebase_result = self._firebase.append_metadata(
                    namespace=namespace,
                    fileID=fileID,
                    chunk_count=len(chunks),
                    keywords=keywords,
                    summary=summary
                )
            
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
    
    def _clean_text(self, text: str) -> Tuple[str, List[str], str]:
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

