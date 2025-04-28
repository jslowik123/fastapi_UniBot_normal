import requests

def upload_pdf(pdf_path: str, namespace: str = "WING"):
    """
    Upload a local PDF file directly to the server.
    
    Args:
        pdf_path: Path to the PDF file
        namespace: Pinecone namespace to use
        
    Returns:
        str: The message from the response
    """
    with open(pdf_path, "rb") as f:
        response = requests.post(
            "https://uni-chatbot-e2bc39ffc8de.herokuapp.com/upload",
            files={"file": (pdf_path, f, "application/pdf")},
            data={"namespace": namespace}
        )
    return response  # Return the response as JSON

if __name__ == "__main__":
    response = requests.post(
            "https://uni-chatbot-e2bc39ffc8de.herokuapp.com/create_namespace",
            data={"namespace": "new namespace for test", "dimension":1536}
        )
    print(response.json())  # Print the full response content