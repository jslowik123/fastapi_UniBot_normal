import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import asyncio
from doc_processor import DocProcessor
from firebase_connection import FirebaseConnection

async def test_document_search():
    """Test the document search functionality and diagnose JSON parsing issues."""
    try:
        load_dotenv()
        print("Testing document search functionality...")
        
        # Initialize dependencies
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        doc_processor = DocProcessor(pinecone_api_key, openai_api_key)
        
        namespace = "neuertest"
        user_query = "Was ist Makroökonomie?"
        
        # Get namespace data
        print(f"Getting namespace data for {namespace}...")
        extracted_data = doc_processor.get_namespace_data(namespace)
        print(f"Found {len(extracted_data)} documents in namespace")
        
        # Try document search
        print(f"Searching for appropriate document for query: '{user_query}'")
        try:
            result = doc_processor.appropiate_document_search(namespace, extracted_data, user_query)
            print(f"Document search result: {result}")
        except Exception as e:
            print(f"Error during document search: {e}")
        
        # Test the JSON structure directly
        print("\nTesting direct JSON parsing to diagnose the issue:")
        openai = OpenAI(api_key=openai_api_key)
        
        prompt = {
            "role": "system", 
            "content": "Du bist ein Assistent, der verschiedene Informationen über Dokumente bekommt. Du sollst entscheiden welches Dokument am besten passt um eine Frage des Nutzers zu beantworten. Antworte im JSON-Format mit genau diesem Schema: {\"id\": \"document_id\", \"chunk_count\": number}. Verwende keine anderen Felder und füge keine Erklärungen hinzu."
        }
        
        user_message = {
            "role": "user",
            "content": f"Hier sind die Themen der Dokumente:\n\n{extracted_data}. Bitte antworte im JSON-Format, indem du nur die ID des geeigneten Dokuments zurückgibst, sowie die Anzahl der Chunks des Dokuments. Die Frage des Users lautet: {user_query}"
        }
        
        print("Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[prompt, user_message],
            temperature=0.1,
        )
        
        response_content = response.choices[0].message.content
        print(f"Raw response content:\n{response_content}")
        
        # Try parsing the JSON
        try:
            parsed = json.loads(response_content)
            print(f"Successfully parsed JSON: {parsed}")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Implement various cleaning attempts
            print("\nTrying to clean the response...")
            
            # Method 1: Strip whitespace
            try:
                cleaned = response_content.strip()
                print(f"Method 1 (strip): {cleaned}")
                parsed = json.loads(cleaned)
                print("Parsing succeeded after stripping!")
            except json.JSONDecodeError as e:
                print(f"Still failed: {e}")
            
            # Method 2: Extract JSON part
            try:
                import re
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    extracted_json = json_match.group(0)
                    print(f"Method 2 (regex): {extracted_json}")
                    parsed = json.loads(extracted_json)
                    print("Parsing succeeded after regex extraction!")
                else:
                    print("No JSON pattern found")
            except json.JSONDecodeError as e:
                print(f"Still failed: {e}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")

def parse_json_safely(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Try to extract just a JSON object with regex
        import re
        json_match = re.search(r'\{.*\}', json_string, re.DOTALL)
        if json_match:
            try:
                extracted_json = json_match.group(0)
                return json.loads(extracted_json)
            except:
                pass
        return None

if __name__ == "__main__":
    asyncio.run(test_document_search()) 