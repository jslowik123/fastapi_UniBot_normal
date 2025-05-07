import requests
import os
from dotenv import load_dotenv
from firebase_connection import FirebaseConnection
from openai import OpenAI
import json
from pinecone_connection import PineconeCon
from doc_processor import DocProcessor
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

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    doc = DocProcessor(pinecone_api_key, openai_api_key)
    namespace_data = doc.get_namespace_data("neuertest")
    print(namespace_data)
    res = doc.appropiate_document_search(extracted_data=namespace_data, user_query="Was versteht man unter der Makroökonomie?", namespace="neuertest")
    print(res)

def query_by_id_prefix():
    con = PineconeCon("userfiles")
    results = con.query_by_id_prefix(query="Was versteht man unter der Makroökonomie?", id_prefix="-OPWqLaaM3Lsclxr-5fM", chunk_count=12, namespace="neuertest", num_results=1)
    print(results)
            

if __name__ == "__main__":
    query_by_id_prefix() 