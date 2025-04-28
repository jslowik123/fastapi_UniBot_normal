import requests

def upload_pdf(pdf_path: str, namespace: str = "WING"):
    """
    Upload a local PDF file directly to the server.
    
    Args:
        pdf_path: Path to the PDF file
        namespace: Pinecone namespace to use
    """
    with open(pdf_path, "rb") as f:
        response = requests.post(
            "https://uni-chatbot-e2bc39ffc8de.herokuapp.com/upload",
            files={"file": (pdf_path, f, "application/pdf")},
            data={"namespace": namespace}
        )
    return response.json()

if __name__ == "__main__":
    result = upload_pdf("test.pdf")
    print(result)