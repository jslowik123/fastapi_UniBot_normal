import requests

def send_message():
    # API endpoint
    url = "http://localhost:8000/send_message"
    
    # Start the bot first
    start_url = "http://localhost:8000/start_bot"
    start_response = requests.post(start_url)
    print("Start Bot Response:", start_response.json())
    
    # Prepare form data
    data = {
        'user_input': 'Was ist der Unterschied zwischen einem Bachelor und Master Studium?',
        'namespace': 'ns1'
    }
    
    try:
        # Send the request
        response = requests.post(url, data=data)
        
        # Print the response
        print("Status Code:", response.status_code)
        print("Response:", response.json())
        
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    send_message() 