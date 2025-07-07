import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DIMENSION = 1536
MAX_RETRIES = 10
RETRY_DELAY = 1


class PineconeCon:
    """
    Handles connections and operations with Pinecone vector database.
    
    Manages vector embeddings, uploads, queries, and namespace operations
    for document storage and retrieval.
    """
    
    def __init__(self, index_name: str):
        """
        Initialize Pinecone connection and OpenAI client.
        
        Args:
            index_name: Name of the Pinecone index to connect to
            
        Raises:
            ValueError: If required API keys are missing
            ConnectionError: If unable to connect to Pinecone index
        """
        load_dotenv(dotenv_path=".env")
        
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self._pc = Pinecone(api_key=pinecone_key)
        self._openai = OpenAI(api_key=openai_key)
        self._index_name = index_name

        # Wait for index to be ready with timeout
        retries = 0
        while retries < MAX_RETRIES:
            try:
                index_status = self._pc.describe_index(index_name)
                if index_status.status['ready']:
                    break
            except Exception as e:
                print(f"Error checking index status: {e}")
            
            retries += 1
            time.sleep(RETRY_DELAY)
            
        if retries >= MAX_RETRIES:
            raise ConnectionError(f"Unable to connect to Pinecone index '{index_name}' after {MAX_RETRIES} retries")

        self._index = self._pc.Index(index_name)


    def query(self, query: str, namespace: str, fileID: str, num_results: int = 3) -> Any:
        """
        Search for similar content using semantic vector search.
        
        Converts the query to an embedding and finds the most similar
        stored content within the specified document.
        
        Args:
            query: Text query to search for
            namespace: Namespace to search within  
            fileID: Specific document ID to search within
            num_results: Maximum number of results to return
            
        Returns:
            Pinecone query results with matches and metadata
            
        Raises:
            Exception: If embedding generation or query fails
        """
        # BULLETPROOF: Sanitize query - never throw error for empty string
        if not query or not query.strip():
            query = "Bitte stellen Sie eine Frage"
            
        try:
            # Generate embedding for the query
            response = self._openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=query
            )
            embedding = response.data[0].embedding

            # Filter to search only within the specified document
            query_filter = {"pdf_id": fileID}

            print(f"PINECONE QUERY DETAILS:")
            print(f"  Query: '{query}'")
            print(f"  Namespace: '{namespace}'")
            print(f"  Filter: {query_filter}")
            print(f"  Top K: {num_results}")
            print(f"  Embedding model: {EMBEDDING_MODEL}")
            print(f"  Embedding length: {len(embedding)}")

            results = self._index.query(
                namespace=namespace,
                vector=embedding,
                top_k=num_results,
                include_values=False,
                include_metadata=True,
                filter=query_filter
            )
            
            print(f"  Raw Pinecone Results: {results}")
            print(f"  Matches found: {len(results.matches) if hasattr(results, 'matches') and results.matches else 0}")
            
            return results
            
        except Exception as e:
            print(f"ERROR in Pinecone query: {str(e)}")
            raise

    def get_adjacent_chunks(self, chunk_id: str, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Retrieve adjacent chunks (previous and next) for a given chunk ID.
        
        Args:
            chunk_id: ID of the current chunk
            namespace: Namespace to search within
            fileID: Document ID
            
        Returns:
            Dict containing previous and next chunks if they exist
        """
        try:
            # Extract chunk number from ID (assuming format like "fileID_chunk_0", "fileID_chunk_1", etc.)
            chunk_parts = chunk_id.split('_')
            if len(chunk_parts) < 3 or chunk_parts[-2] != 'chunk':
                return {"previous": None, "next": None}
            
            try:
                chunk_number = int(chunk_parts[-1])
            except ValueError:
                return {"previous": None, "next": None}
            
            # Build IDs for adjacent chunks
            base_id = '_'.join(chunk_parts[:-1])  # Everything except the number
            prev_id = f"{base_id}_{chunk_number - 1}" if chunk_number > 0 else None
            next_id = f"{base_id}_{chunk_number + 1}"
            
            result = {"previous": None, "next": None}
            
            # Query for previous chunk
            if prev_id:
                try:
                    prev_result = self._index.fetch(ids=[prev_id], namespace=namespace)
                    if prev_result.vectors and prev_id in prev_result.vectors:
                        result["previous"] = prev_result.vectors[prev_id]
                except Exception:
                    pass  # Previous chunk doesn't exist
            
            # Query for next chunk
            try:
                next_result = self._index.fetch(ids=[next_id], namespace=namespace)
                if next_result.vectors and next_id in next_result.vectors:
                    result["next"] = next_result.vectors[next_id]
            except Exception:
                pass  # Next chunk doesn't exist
            
            return result
            
        except Exception as e:
            return {"previous": None, "next": None}

    def query_with_adjacent_chunks(self, query: str, namespace: str, fileID: str, num_results: int = 3) -> Any:
        """
        Search for similar content and include adjacent chunks for each result.
        
        Args:
            query: Text query to search for
            namespace: Namespace to search within  
            fileID: Specific document ID to search within
            num_results: Maximum number of results to return
            
        Returns:
            Enhanced Pinecone query results with adjacent chunks included
        """
        # BULLETPROOF: Sanitize query - never throw error for empty string
        if not query or not query.strip():
            query = "Bitte stellen Sie eine Frage"
            
        try:
            print("\n" + "="*80)
            print("PINECONE QUERY WITH ADJACENT CHUNKS DEBUGGING:")
            print("-" * 40)
            print(f"Query: '{query}'")
            print(f"Namespace: '{namespace}'")
            print(f"FileID: '{fileID}'")
            print(f"Num Results: {num_results}")
            
            # Get regular query results first
            results = self.query(query, namespace, fileID, num_results)
            
            print(f"Initial Query Results: {results}")
            if hasattr(results, 'matches'):
                print(f"Number of matches: {len(results.matches) if results.matches else 0}")
                if results.matches:
                    for i, match in enumerate(results.matches):
                        print(f"  Match {i+1}: ID={match.id}, Score={match.score:.4f}")
            else:
                print("No matches attribute found in results")
            
            # For each match, try to get adjacent chunks
            if results and hasattr(results, 'matches') and results.matches:
                for match in results.matches:
                    if hasattr(match, 'id') and match.id:
                        print(f"Getting adjacent chunks for: {match.id}")
                        adjacent = self.get_adjacent_chunks(match.id, namespace, fileID)
                        print(f"Adjacent chunks result: {adjacent}")
                        # Add adjacent chunks to match metadata
                        if not hasattr(match, 'metadata'):
                            match.metadata = {}
                        if not match.metadata:
                            match.metadata = {}
                        match.metadata['adjacent_chunks'] = adjacent
            
            print("="*80 + "\n")
            return results
            
        except Exception as e:
            print(f"ERROR in query_with_adjacent_chunks: {str(e)}")
            print("="*80 + "\n")
            raise
