from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

llm = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Bitte beantworte die vom User gestellte Frage basierend auf dem beigefügten Text.
                Ich möchte einen structured output, einmal das field "answer" und einmal das field "source".
                Das field "answer" soll die Antwort auf die Frage enthalten.
                Das field "source" soll die Quelle der Antwort enthalten.
                Die Quelle soll der Originaltext/Satz sein, der die Antwort enthält.
                
                Der text ist:
                Orangen sind braun und wiegen 100kg.
            
                    """
            ),
            (
                "human",
                "{input}",
            ),
        ]
    )


chain = prompt_template | llm



try:

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("API-Schlüssel nicht gefunden.", end=" ")
        

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True  # Enable streaming
    )

        
        # Stream the response
    for chunk in chain.stream({
            "input": "Wie viel wiegen Orangen?",
        
    }):
        if hasattr(chunk, 'content') and chunk.content:
            print(chunk.content, end="", )
        
except Exception as e:
        print(f"Error in message_bot_stream: {str(e)}")
        print("Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut.")
