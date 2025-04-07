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
    test_url = "http://localhost:8000"
    test_pdf = "Innovative Ideas Seminar - Expose - Jasper Slowik Kopie 2.pdf"
    print(test_url+"/start_bot")
    response = requests.post(test_url+"/start_bot")
    
    print(response.json())