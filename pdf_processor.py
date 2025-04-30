import os
import tempfile
import PyPDF2
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import unicodedata
import re

class PDFProcessor:
    def __init__(self, pinecone_api_key: str, openai_api_key: str):
        """
        Initialize PDFProcessor with API keys.
        
        Args:
            pinecone_api_key: API key for Pinecone
            openai_api_key: API key for OpenAI
        """
        self._openai = OpenAI(api_key=openai_api_key)
        self._pinecone = Pinecone(api_key=pinecone_api_key)
        self._index = self._pinecone.Index("userfiles")

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text by:
        1. Normalizing unicode characters
        2. Replacing multiple spaces with single space
        3. Removing special characters while preserving umlauts
        4. Normalizing line endings
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep umlauts and basic punctuation
        text = re.sub(r'[^\w\s.,;:!?äöüÄÖÜß-]', '', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()

    def process_pdf(self, file_path: str, namespace: str = "ns1") -> Dict[str, Any]:
        """
        Process a PDF file and store its content in Pinecone.
        
        Args:
            file_path: Path to the PDF file
            namespace: Pinecone namespace to use
            
        Returns:
            Dict containing processing results
        """
        try:
            # Read PDF content
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # Clean the text
            text = self._clean_text(text)

            # Split text into chunks
            chunks = self._split_text(text)
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                # Create embedding
                response = self._openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embedding = response.data[0].embedding

                # Store in Pinecone
                self._index.upsert(
                    namespace=namespace,
                    vectors=[{
                        "id": f"{os.path.basename(file_path)}_{i}",
                        "values": embedding,
                        "metadata": {
                            "text": chunk,
                            "file": os.path.basename(file_path),
                            "chunk_number": i
                        }
                    }]
                )

            return {
                "status": "success",
                "message": f"File {os.path.basename(file_path)} processed successfully",
                "chunks": len(chunks)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

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
        
        # Split by sentences (rough approximation)
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