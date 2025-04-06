import requests

def send_pdf_post(url, pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        files = {'file': (pdf_path.split('/')[-1], pdf_file, 'application/pdf')}
        response = requests.post(url, files=files)
        
        if response.status_code == 200:
            print("Erfolg! Server-Antwort:", response.json())
        else:
            print(f"Fehler {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_url = "http://localhost:8000/upload"
    test_pdf = "Innovative Ideas Seminar - Expose - Jasper Slowik Kopie 2.pdf"
    send_pdf_post(test_url, test_pdf)