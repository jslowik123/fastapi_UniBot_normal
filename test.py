from doc_processor import DocProcessor
import os
import requests

if __name__ == "__main__":
    bot = requests.post("https://uni-chatbot-e2bc39ffc8de.herokuapp.com/start_bot")
    response = requests.post("https://uni-chatbot-e2bc39ffc8de.herokuapp.com/send_message", data={"user_input": "Was ist Python?", "namespace": "zwischenpraesentation"})
    print(f"Status: {response.status_code}, Response: {response.json()}")
    
