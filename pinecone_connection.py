import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional, Union
import numpy as np
from openai import OpenAI

class PineconeCon:
    def __init__(self, index_name: str):
        load_dotenv(dotenv_path=".env")          
        self._pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self._openai = OpenAI()

        while not self._pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        self._index = self._pc.Index(index_name)
        self._index_name = index_name

    def upload(self, chunks, namespace: str, file_name: str, fileID: str) -> List[Any]:
        for i, chunk in enumerate(chunks):
                response = self._openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embedding = response.data[0].embedding

                self._index.upsert(
                    namespace=namespace,
                    vectors=[{
                        "id": fileID,
                        "values": embedding,
                        "metadata": {
                            "id": fileID,
                            "text": chunk,
                            "file": file_name,
                            "chunk_number": i
                        }
                    }]
                )
        return {
                "status": "success",
                "message": f"File {fileID} processed successfully",
                "chunks": len(chunks),
                "original_file": file_name
        }

    def delete_embeddings(self, file_name: str, namespace: str = "ns1") -> None:
        self._index.delete(
            namespace=namespace,
            filter={"file": file_name}
        )

    def query(self, query: str, namespace: str, fileID: str, chunk_count: int,  num_results: int = 3) -> List[Dict[str, Any]]:
        response = self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        embedding = response.data[0].embedding


        list_ids = [f"{fileID}_{i}" for i in range(chunk_count)]
        print(list_ids)
        filter={"file": "tmpc6i3re44.pdf"}

        results = self._index.query(
            namespace=namespace,
            vector=embedding,
            top_k=num_results,
            include_values=False,
            include_metadata=True,
            filter=filter
        )

        numbered_results = []
        # for i, match in enumerate(results["matches"]):
        #     if match["score"] >= 0.8:
        #         numbered_results.append({
        #             "text": f"{i+1}. {match['metadata']['text']}",
        #             "score": match["score"],
        #         })
        return results


    def delete_namespace(self, namespace: str = "ns1") -> None:
        self._index.delete(namespace=namespace, delete_all=True)

    def create_namespace_with_dummy(self, namespace: str, dimension: int = 1536) -> Dict[str, Any]:
        dummy_vector = np.random.rand(dimension).tolist()
        
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
        
    def delete_all(self, namespace: str) -> Dict[str, Any]:
        """
        Löscht alle Vektoren in einem Namespace.
        
        Args:
            namespace: Der Namespace, in dem alle Vektoren gelöscht werden sollen
            
        Returns:
            Dict mit Statusinformationen
        """
        try:
            self._index.delete(namespace=namespace, delete_all=True)
            
            return {
                "status": "success",
                "message": f"Alle Vektoren im Namespace '{namespace}' erfolgreich gelöscht"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Fehler beim Löschen aller Vektoren: {str(e)}"
            }

    def query_by_id_prefix(self, query: str, namespace: str, id_prefix: str, chunk_count: int, num_results: int = 3) -> List[Dict[str, Any]]:
        """
        Führt eine Vektorsuche durch und filtert nach Vektoren, deren IDs mit dem gegebenen Präfix beginnen.
        
        Args:
            query: Die Suchanfrage
            namespace: Der Namespace, in dem gesucht werden soll
            id_prefix: Das Präfix der Vektor-IDs, nach denen gefiltert werden soll
            num_results: Anzahl der zurückzugebenden Ergebnisse
            
        Returns:
            Liste mit den ähnlichsten Vektoren und deren Metadaten
        """
        response = self._openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        embedding = response.data[0].embedding

        list_ids = [f"{id_prefix}_{i}" for i in range(chunk_count)]
        print(list_ids)
        filter={"id": {"$in": list_ids}}
        results = self._index.query(
            namespace=namespace,
            vector=embedding,
            top_k=num_results,
            include_values=False,
            include_metadata=True,
            filter=filter
        )
        print(results)
        numbered_results = []
        for i, match in enumerate(results["matches"]):
            # if match["score"] >= 0.5:
                numbered_results.append({
                    "text": f"{i+1}. {match['metadata']['text']}",
                    "score": match["score"],
                    "file": match["metadata"].get("file", ""),
                    "chunk_number": match["metadata"].get("chunk_number", 0),
                    "id": match.get("id", "")
                })
        return numbered_results


def delete_all():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("userfiles")
    
    try:
        index.delete(delete_all=True)
        return {
            "status": "success",
            "message": "Alle Vektoren im Index erfolgreich gelöscht"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Fehler beim Löschen aller Vektoren: {str(e)}"
        }


def test_query():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("userfiles")
    pc = PineconeCon("userfiles")
    results = pc.query(query="Was versteht man unter der Makroökonomie?", namespace="neuertest", fileID="-OPWqLaaM3Lsclxr-5fM", chunk_count=12, num_results=3)
    print(results)

if __name__ == "__main__":
    test_query()