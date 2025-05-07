import requests
import os
from dotenv import load_dotenv
from firebase_connection import FirebaseConnection
from openai import OpenAI
import json
from pinecone_connection import PineconeCon
from doc_processor import DocProcessor
import main
import asyncio

async def test_bot():
    main.pinecone_api_key = os.getenv("PINECONE_API_KEY")
    main.openai_api_key = os.getenv("OPENAI_API_KEY")

    await main.start_bot()    
    response = await main.send_message("Was ist Makroökonomie?", "neuertest")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(test_bot()) 