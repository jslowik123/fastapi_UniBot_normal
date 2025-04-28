import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI

class PineconeCon:
    """
    A class to handle interactions with Pinecone vector database.
    
    This class provides methods for creating embeddings, uploading vectors,
    querying the database, and managing the index.
    """
    
    def __init__(self, index_name: str):
        """
        Initialize the Pinecone connection.
        
        Args:
            index_name: Name of the Pinecone index to use
        """
        load_dotenv(dotenv_path=".env")
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
            
        self._pc = Pinecone(api_key=api_key)
        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        while not self._pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        self._index = self._pc.Index(index_name)
        self._index_name = index_name

    def _chunk_text(self, text: Union[str, List[str]], max_bytes: int = 40960*0.5) -> List[str]:
        """
        Split text into chunks of maximum size in bytes.
        
        Args:
            text: The text to split (string or list of strings)
            max_bytes: Maximum size of each chunk in bytes (default: 20480)
            
        Returns:
            List of text chunks
        """
        # Convert list to string if necessary
        if isinstance(text, list):
            text = ' '.join(text)
            
        chunks = []
        current_chunk = ""
        current_size = 0
        
        # Split text into sentences (rough approximation)
        sentences = text.split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence.encode('utf-8'))
            
            if current_size + sentence_size > max_bytes:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_size = sentence_size
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size + 2  # +2 for ". "
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def create_index(self, new_index_name: str, dims: int) -> None:
        """
        Create a new Pinecone index.
        
        Args:
            new_index_name: Name of the new index
            dims: Dimension of the vectors
        """
        self._pc.create_index(
            name=new_index_name,
            dimension=dims,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    def create_embeddings(self, data: List[Dict[str, str]]) -> List[Any]:
        """
        Create embeddings for the given data using OpenAI's embedding model.
        
        Args:
            data: List of dictionaries containing file and content information
            
        Returns:
            List of embeddings for each document
        """
        list_embeddings = []
        for doc in data:
            # Split text into chunks
            chunks = self._chunk_text(doc["content"])
            # Number the chunks
            numbered_chunks = [f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)]
            
            # Create embeddings using OpenAI
            embeddings = []
            for chunk in numbered_chunks:
                response = self._openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk
                )
                embeddings.append(response.data[0].embedding)
            list_embeddings.append(embeddings)
        return list_embeddings

    def upload_embeddings(self, data: List[Dict[str, str]], embeddings: List[Any], namespace: str = "ns1") -> None:
        """
        Upload embeddings to Pinecone.
        
        Args:
            data: List of dictionaries containing file and content information
            embeddings: List of embeddings to upload
            namespace: Pinecone namespace to use (default: "ns1")
        """
        vectors = []
        for d, e_list in zip(data, embeddings):
            for i, e in enumerate(e_list):
                # Create unique ID for each chunk
                patch_id = f"{d['file']}_patch{i}"
                # Get the chunk text from the original chunks
                chunks = self._chunk_text(d["content"])
                chunk_text = chunks[i] if i < len(chunks) else ""
                
                vectors.append({
                    "id": patch_id,
                    "values": e,
                    "metadata": {
                        'text': chunk_text,
                        'file': d['file'],
                        'chunk_number': i+1
                    }
                })
        self._index.upsert(
            vectors=vectors,
            namespace=namespace,
        )

    def delete_embeddings(self, file_name: str, namespace: str = "ns1") -> None:
        """
        Delete all vectors associated with a specific file from Pinecone.
        
        Args:
            file_name: The name of the file whose vectors should be deleted
            namespace: Pinecone namespace to use (default: "ns1")
        """
        self._index.delete(
            namespace=namespace,
            filter={"file": file_name}
        )

    def query(self, query: str, num_results: int = 3, namespace: str = "ns1") -> List[Dict[str, Any]]:
        """
        Query the Pinecone index and return the top results.
        
        Args:
            query: The query text
            num_results: Number of results to return (default: 3)
            namespace: Pinecone namespace to use (default: "ns1")
            
        Returns:
            List of top results with their metadata
        """
        # Create embedding using OpenAI
        response = self._openai.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        embedding = response.data[0].embedding

        results = self._index.query(
            namespace=namespace,
            vector=embedding,
            top_k=num_results,
            include_values=False,
            include_metadata=True
        )

        # Return all matches that have a score above threshold, numbered
        numbered_results = []
        for i, match in enumerate(results["matches"]):
            if match["score"] >= 0.8:
                numbered_results.append({
                    "text": f"{i+1}. {match['metadata']['text']}",
                    "score": match["score"],
                })
        return numbered_results

    def delete_namespace(self, namespace: str = "ns1") -> None:
        """
        Delete all vectors from the specified namespace.
        
        Args:
            namespace: Pinecone namespace to use (default: "ns1")
        """
        self._index.delete(namespace=namespace, delete_all=True)

    def create_namespace_with_dummy(self, namespace: str, dimension: int = 1024) -> Dict[str, Any]:
        """
        Create a new namespace and upload a dummy vector.
        
        Args:
            namespace: Name of the new namespace to create
            dimension: Dimension of the vector (default: 1024)
            
        Returns:
            Dictionary containing the operation status and details
        """
        # Generate a random vector
        dummy_vector = np.random.rand(dimension).tolist()
        
        # Create and upload vector
        vector = {
            "id": "dummy_vector_1",
            "values": dummy_vector,
            "metadata": {
                "type": "dummy",
                "description": "Initial dummy vector"
            }
        }
        
        self._index.upsert(
            vectors=[vector],
            namespace=namespace
        )
        
        return {
            "status": "success",
            "message": f"Namespace '{namespace}' created with dummy vector",
            "namespace": namespace,
            "vector_id": "dummy_vector_1",
            "dimension": dimension
        }


def delete_all():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("PINECOIN_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("quickstart")

    # Alle Vektoren löschen
    index.delete(namespace="ns1", delete_all=True)

