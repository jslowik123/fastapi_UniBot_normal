import requests

def upload_pdf(pdf_path: str, namespace: str = "was geht"):
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
    response = upload_pdf("test.pdf")
    print(response.json())