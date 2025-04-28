import os
from typing import List, Union
from openai import OpenAI
from dotenv import load_dotenv

def chunk_text(text: Union[str, List[str]], max_bytes: int = 40960*0.5) -> List[str]:
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

def test_embeddings():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Test data
    test_data = [
        {
            "file": "test_document_1",
            "content": "This is a test document. It contains multiple sentences. Each sentence should be properly chunked."
        },
        {
            "file": "test_document_2",
            "content": "Another test document with different content. This one is longer and should test the chunking functionality more thoroughly."
        }
    ]
    
    # Process embeddings
    list_embeddings = []
    for doc in test_data:
        # Split text into chunks
        chunks = chunk_text(doc["content"])
        # Number the chunks
        numbered_chunks = [f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)]
        
        # Create embeddings using OpenAI
        embeddings = []
        for chunk in numbered_chunks:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk
            )
            embeddings.append(response.data[0].embedding)
        list_embeddings.append(embeddings)
    
    # Print results
    print(f"Number of documents processed: {len(list_embeddings)}")
    for i, doc_embeddings in enumerate(list_embeddings):
        print(f"Document {i+1}:")
        print(f"  Number of chunks: {len(doc_embeddings)}")
        print(f"  Embedding dimensions: {len(doc_embeddings[0])}")
        print(f"  First embedding sample: {doc_embeddings[0][:5]}...")  # Print first 5 values

if __name__ == "__main__":
    # test_embeddings()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print(api_key)