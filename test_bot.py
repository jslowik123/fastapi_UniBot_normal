import requests
import os
from dotenv import load_dotenv

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

if __name__ == "__main__":
    test_bot() 