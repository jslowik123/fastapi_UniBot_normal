import os
import json
from dotenv import load_dotenv
from pinecone_connection import PineconeCon
from openai import OpenAI

# Load environment variables once at module level
load_dotenv()


def _get_openai_client(streaming: bool = False) -> OpenAI:
    """
    Creates and returns a configured OpenAI client.
    
    Args:
        streaming: Whether to enable streaming for real-time responses (unused now)
        
    Returns:
        OpenAI: Configured OpenAI client
        
    Raises:
        ValueError: If OpenAI API key is not found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return OpenAI(api_key=api_key)


def get_bot():
    """
    Validates OpenAI connection and returns a simple success indicator.
    
    Returns:
        bool: True if OpenAI client can be created successfully
        
    Raises:
        ValueError: If OpenAI client cannot be created
    """
    try:
        client = _get_openai_client()
        return True
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI client: {e}")


def _validate_inputs(user_input, context, database_overview, chat_history):
    """
    Validates and sanitizes all input parameters to prevent errors.
    
    Args:
        user_input: User's input message
        context: Document context
        database_overview: Database overview
        chat_history: Chat history
        
    Returns:
        Tuple of validated inputs
    """
    # Validate user input
    if not user_input or not isinstance(user_input, str):
        user_input = "Bitte stellen Sie eine Frage"
    user_input = user_input.strip() if user_input.strip() else "Bitte stellen Sie eine Frage"
    
    # Validate context
    if not isinstance(context, str):
        context = str(context) if context else ""
    
    # Validate database overview
    if not isinstance(database_overview, list):
        database_overview = []
    
    # Validate chat history
    if not isinstance(chat_history, list):
        chat_history = []
    
    return user_input, context, database_overview, chat_history


def _format_chat_history(chat_history):
    """
    Converts chat history to OpenAI message format.
    
    Args:
        chat_history: List of chat messages with role and content
        
    Returns:
        list: Formatted chat history for OpenAI API
    """
    if not isinstance(chat_history, list) or not chat_history:
        return []
    
    formatted_history = []
    for msg in chat_history:
        try:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                continue
            
            role = msg["role"]
            content = msg["content"]
            
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            
            if not role.strip() or not content.strip():
                continue
            
            if role.strip().lower() in ["user", "assistant"]:
                formatted_history.append({
                    "role": role.strip().lower(), 
                    "content": content.strip()
                })
                
        except Exception:
            continue
    
    return formatted_history



def message_bot(user_input, context, document_id, database_overview, chat_history):
    """
    Processes a user message and returns a response from the chatbot using direct OpenAI API.
    
    Args:
        user_input: The user's question or message
        context: Relevant document context from vector search
        document_id: ID of the document being referenced
        database_overview: Overview of available documents
        chat_history: Previous conversation history
        
    Returns:
        str: The chatbot's response
    """
    try:
        # Validate all inputs
        user_input, context, database_overview, chat_history = _validate_inputs(
            user_input, context, database_overview, chat_history
        )
        
        # Validate document_id
        if document_id is None:
            document_id = ""
        elif not isinstance(document_id, str):
            document_id = str(document_id)

        # Format chat history
        formatted_history = _format_chat_history(chat_history)

        # Create OpenAI client
        try:
            openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except Exception as e:
            return "Entschuldigung, es ist ein Fehler beim Erstellen des AI-Clients aufgetreten."

        # Print the full context being sent to LLM
        print("=" * 100)
        print("CONTEXT BEING SENT TO LLM:")
        print("=" * 100)
        print(context)
        print("=" * 100)

        # Create system message with context
        system_content = f"""Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität.

HOCHSCHULSPEZIFISCHE INFORMATIONEN:
{context}

{f'''DATABASE OVERVIEW (verfügbare Dokumente):
{str(database_overview)}''' if database_overview else ''}

[SYSTEM_INFO] FOUND_DOCUMENT_ID: {document_id}

VERHALTEN:
- Stütze deine Antworten auf die bereitgestellten Quellen
- Antworte natürlich und direkt, als würdest du mit Studierenden sprechen
- Gib ausführliche, aber präzise Antworten
- Verwende innerhalb der "answer" kein "" sondern nur ''

WICHTIG ZU SEITENZAHLEN:
- Jeder Textabschnitt in den HOCHSCHULSPEZIFISCHEN INFORMATIONEN ist mit seiner Seitenzahl markiert (z.B. "SEITE 5")
- Du MUSST die Seitenzahlen der Textabschnitte identifizieren, die du für deine Antwort verwendet hast
- Gib nur die Seitenzahlen der Textabschnitte an, die du tatsächlich zitiert hast

ANTWORTFORMAT:
{{
  "answer": "Deine ausführliche Antwort hier",
  "document_id": "{document_id}",
  "source": "Kopiere hier EXAKT und WÖRTLICH die spezifischen Sätze oder Textpassagen aus den HOCHSCHULSPEZIFISCHEN INFORMATIONEN, die du für deine Antwort verwendet hast. Gib nur die tatsächlichen Originalsätze wieder - keine Zusammenfassungen, keine Paraphrasierungen, keine eigenen Formulierungen. Wenn du mehrere Sätze verwendet hast, trenne sie mit ' | '. Beispiel: 'Die Anmeldung erfolgt über das Studentenportal. | Die Prüfung findet im Sommersemester statt.'",
  "pages": [hier die Seitenzahlen als Liste von Zahlen der Textabschnitte, die du für deine Antwort verwendet hast, z.B. [5, 12, 15]]
  }}"""

        # Create messages array
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input}
        ]

        # Add chat history to messages
        if formatted_history:
            for hist_msg in formatted_history:
                if isinstance(hist_msg, dict) and hist_msg.get("role") in ["user", "assistant"]:
                    messages.insert(-1, hist_msg)

        # Call OpenAI API directly
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                max_tokens=2000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return "Entschuldigung, es ist ein Fehler bei der AI-Verarbeitung aufgetreten."
        
    except Exception as e:
        return "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."

