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
    return response.text  # Return the response message as text

if __name__ == "__main__":
    result = upload_pdf("tipps_fuer_studierende.pdf")
    print(result)