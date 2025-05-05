import requests
import os
from dotenv import load_dotenv
from firebase_connection import FirebaseConnection
from openai import OpenAI
import json
from pinecone_connection import PineconeCon

def test_bot():
    # Load environment variables
    load_dotenv()
    
    # Server URL
    base_url = "https://uni-chatbot-e2bc39ffc8de.herokuapp.com"
    
    try:
        # Start the bot
        start_response = requests.post(f"{base_url}/start_bot")
        print("\nStarting bot:", start_response.json())
        
        # Test message
        test_message = "Was ist ein Bachelor-Studium?"
        
        # Send message to bot
        message_response = requests.post(
            f"{base_url}/send_message",
            data={
                "user_input": test_message,
                "namespace": "ns1"  # Using default namespace
            }
        )
        
        response_data = message_response.json()
        print("\nTest Message:", test_message)
        print("Server Response:", response_data)
        
        if "status" in response_data and response_data["status"] == "success":
            print("Bot Response:", response_data.get("response", "No response content"))
        else:
            print("Error:", response_data.get("message", "Unknown error"))
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")

def appropiate_document_search():
    # Load environment variables
    load_dotenv()
    
    # Setze die notwendigen Umgebungsvariablen, falls sie in .env fehlen
    if not os.getenv('FIREBASE_DATABASE_URL'):
        os.environ['FIREBASE_DATABASE_URL'] = "https://chatbot-17dbe-default-rtdb.europe-west1.firebasedatabase.app/"
    
    # Verwende den Pfad zur Credentials-Datei statt des JSON-Strings
    if os.path.exists('firebase-credentials.json'):
        os.environ['FIREBASE_CREDENTIALS_PATH'] = 'firebase-credentials.json'
        
    fb = FirebaseConnection()
    openai = OpenAI()
    # Namespace-spezifische Daten abrufen
    namespace_data = fb.get_namespace_data("neuertest")
    print("Namespace-Daten:", json.dumps(namespace_data, indent=2, ensure_ascii=False))
    
    if namespace_data['status'] == 'success' and 'data' in namespace_data:
            for doc_id, doc_data in namespace_data['data'].items():
                print(doc_id)

                
if __name__ == "__main__":
    appropiate_document_search() 