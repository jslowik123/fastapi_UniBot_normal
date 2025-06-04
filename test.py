from doc_processor import DocProcessor
import os
import requests
import time

if __name__ == "__main__":
    # Start bot first
    bot = requests.post("https://uni-chatbot-e2bc39ffc8de.herokuapp.com/start_bot")
    time.sleep(5)
    print("\n=== Testing streaming /send_message_stream ===")
    # Test streaming endpoint
    stream_response = requests.post("https://uni-chatbot-e2bc39ffc8de.herokuapp.com/send_message_stream", 
                                  data={"user_input": "Woraus setzten sich module zusammen?", "namespace": "zwischenpraesentation"}, 
                                  stream=True)
    
    print(f"Stream Status: {stream_response.status_code}")
    if stream_response.status_code == 200:
        for line in stream_response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    print(f"Stream chunk: {decoded_line}")
    else:
        print(f"Stream Error: {stream_response.text}")
    
