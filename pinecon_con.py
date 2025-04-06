import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


class PineconeCon:
    def __init__(self, index_name):
        load_dotenv(dotenv_path=".env")
        api_key = os.getenv("PINECOIN_API_KEY")
        self._pc = Pinecone(api_key=api_key)

        while not self._pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        self._index = self._pc.Index(index_name)
        self._index_name = index_name

    def create_index(self, new_index_name, dims):
        self._pc.create_index(
            name=new_index_name,
            dimension=dims, # Replace with your model dimensions
            metric="cosine", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    def create_embeddings(self, data):
        list_embeddings = []
        for doc in data:
            # Split text into chunks and number them
            chunks = [f"{i+1}. {text}" for i, text in enumerate(doc["content"])]
            embeddings = self._pc.inference.embed(
                model="multilingual-e5-large",
                inputs=chunks,
                parameters={"input_type": "passage", "truncate": "END"}
            )
            list_embeddings.append(embeddings)
        return list_embeddings

    def upload_embeddings(self, data, embeddings):
        vectors = []
        for d, e_list in zip(data, embeddings):
            for i, e in enumerate(e_list):
                # Eindeutige ID pro Patch
                patch_id = f"{d['file']}_patch{i}"
                # Number the text chunk
                numbered_text = f"{i+1}. {d['content'][i]}"
                vectors.append({
                    "id": patch_id,
                    "values": e['values'],
                    "metadata": {
                        'text': numbered_text,
                        'file': d['file'],
                        'chunk_number': i+1
                    }
                })
        self._index.upsert(
            vectors=vectors,
            namespace="ns1"
        )

    def delete_embeddings(self, file_name):
        """
        Delete all vectors associated with a specific file from Pinecone.
        
        Args:
            file_name (str): The name of the file whose vectors should be deleted
        """
        # Delete all vectors with IDs starting with the file name
        self._index.delete(
            namespace="ns1",
            filter={"file": file_name}
        )

    def query(self, query, num_results=3):
        """
        Query the Pinecone index and return the top results.
        
        Args:
            query (str): The query text
            num_results (int): Number of results to return (default: 3)
            
        Returns:
            list: List of top results with their metadata
        """
        embedding = self._pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query],
            parameters={
                "input_type": "query"
            }
        )

        results = self._index.query(
            namespace="ns1",
            vector=embedding[0].values,
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


def delete_all():
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("PINECOIN_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("quickstart")

    # Alle Vektoren löschen
    index.delete(namespace="ns1", delete_all=True)

