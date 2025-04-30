from chatbot import message_bot
from pinecon_con import PineconeCon
import os
from dotenv import load_dotenv

def test_bot():
    # Load environment variables
    load_dotenv()
    
    # Initialize Pinecone connection
    pinecone = PineconeCon()
    
    # Test message
    test_message = "Was ist ein Bachelor-Studium?"
    
    # Get context from Pinecone (empty for this test)
    context = ""
    
    # Initialize empty chat history
    chat_history = []
    
    # Send message to bot
    response = message_bot(
        user_input=test_message,
        context=context,
        chat_history=chat_history
    )
    
    print("\nTest Message:", test_message)
    print("Bot Response:", response)

if __name__ == "__main__":
    test_bot() 