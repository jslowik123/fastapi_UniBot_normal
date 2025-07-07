from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import json
from dotenv import load_dotenv
from pinecone_connection import PineconeCon

# Load environment variables once at module level
load_dotenv()


def _get_openai_client(streaming: bool = False) -> ChatOpenAI:
    """
    Creates and returns a configured OpenAI client.
    
    Args:
        streaming: Whether to enable streaming for real-time responses
        
    Returns:
        ChatOpenAI: Configured language model client
        
    Raises:
        ValueError: If OpenAI API key is not found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=api_key,
        streaming=streaming
    )

def _create_context_aware_prompt(has_chat_history=False):
    """
    Creates a context-aware prompt template that emphasizes chat history when present.
    
    Args:
        has_chat_history: Whether chat history is present
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    base_system_prompt = """Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität."""
    
    if has_chat_history:
        chat_history_emphasis = """

🔄 CHAT HISTORY BEACHTUNG - WICHTIG:
Du siehst eine Chat History mit vorherigen Nachrichten. BERÜCKSICHTIGE diese aktiv:
- Beziehe dich auf vorherige Fragen und Antworten
- Nutze den Kontext aus früheren Nachrichten
- Wenn der Nutzer "dazu", "darüber", "das" oder ähnliche Bezugswörter verwendet, beziehe dich auf vorherige Themen
- Beantworte Rückfragen oder Nachfragen basierend auf dem bisherigen Gesprächsverlauf
- Vermeide Wiederholungen bereits gegebener Antworten, es sei denn, es wird explizit verlangt
- Erkenne den Kontext der aktuellen Frage im Zusammenhang mit der Chat History"""
    else:
        chat_history_emphasis = """

📝 NEUE UNTERHALTUNG:
Dies ist der Beginn einer neuen Unterhaltung ohne vorherige Chat History."""

    sources_section = """

DEINE ANTWORTQUELLEN:
Diese Informationen stehen dir zur Verfügung:

DATABASE OVERVIEW (Hier sind alle Dokumente die du theoretisch zu Verfügung hast, damit kannst du User z.B. hinweisen über welche Themen du schreiben kannst und welche nicht.):
{database_overview}

HOCHSCHULSPEZIFISCHE INFORMATIONEN (Hier sind einzelnche Chunks extrahiert aus denen im DatabaseOverview enthaltenen Dokumenten):
{context}

{knowledge}

DOKUMENTEN-ID: {document_id}

VERHALTEN:
- Stütze deine Antworten auf die bereitgestellten Quellen
- Antworte natürlich und direkt, als würdest du mit Studierenden sprechen
- Bei Widersprüchen: Bevorzuge hochschulspezifische Informationen
- Bei fehlenden Informationen: Sage es klar und biete Hilfe an
- Gib ausführliche, aber präzise Antworten
- Verwende innerhalb der "answer" kein "" sondern nur ''

ANTWORTFORMAT:
{{
  "answer": "Deine ausführliche Antwort hier",
  "document_id": "{document_id}",
  "source": "Originaltext aus dem Kontext, der die Antwort stützt, korrekt formatiert."
}}"""

    full_system_prompt = base_system_prompt + chat_history_emphasis + sources_section

    return ChatPromptTemplate.from_messages(
        [
            ("system", full_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def get_bot():
    """
    Creates and configures the chatbot chain with prompt template.
    
    Returns:
        Chain: Configured LangChain pipeline for the university chatbot
    """
    llm = _get_openai_client()
    # Use default prompt (this will be overridden in message functions with context-aware prompt)
    prompt_template = _create_context_aware_prompt(has_chat_history=False)
    return prompt_template | llm


def _validate_inputs(user_input, context, knowledge, database_overview, chat_history):
    """
    Validates and sanitizes input parameters for message processing.
    
    Args:
        user_input: User's message
        context: Document context
        knowledge: General knowledge context
        database_overview: Overview of available documents
        chat_history: Previous conversation history
        
    Returns:
        tuple: Validated and sanitized inputs
    """
    # BULLETPROOF: Sanitize user input - never throw errors
    if not user_input or not isinstance(user_input, str):
        user_input = "Bitte stellen Sie eine Frage"
    user_input = user_input.strip() if user_input.strip() else "Bitte stellen Sie eine Frage"
    
    # BULLETPROOF: Validate and clean chat history
    if not isinstance(chat_history, list):
        chat_history = []
    
    # Clean chat history - remove invalid entries
    cleaned_history = []
    for msg in chat_history:
        if (isinstance(msg, dict) and 
            "role" in msg and 
            "content" in msg and 
            isinstance(msg["role"], str) and 
            isinstance(msg["content"], str) and 
            msg["role"].strip() and 
            msg["content"].strip()):
            cleaned_history.append(msg)
    
    chat_history = cleaned_history
        
    # BULLETPROOF: Validate context - NEVER replace valid context
    if context is None:
        context = ""
    elif not isinstance(context, str):
        context = str(context) if context else ""
    # DO NOT check if context is empty - preserve all valid string contexts
    
    # BULLETPROOF: Validate knowledge - NEVER replace valid knowledge
    if knowledge is None:
        knowledge = ""
    elif not isinstance(knowledge, str):
        knowledge = str(knowledge) if knowledge else ""
    # DO NOT check if knowledge is empty - preserve all valid string knowledge
    
    # BULLETPROOF: Validate database_overview
    if database_overview is None:
        database_overview = []
    elif not isinstance(database_overview, list):
        database_overview = []
        
    return user_input, context, knowledge, database_overview, chat_history


def _format_chat_history(chat_history):
    """
    Converts chat history to LangChain message format.
    
    Args:
        chat_history: List of chat messages with role and content
        
    Returns:
        list: Formatted chat history for LangChain
    """
    
    # BULLETPROOF: Validate input
    if not isinstance(chat_history, list):
        return []
    
    if not chat_history:
        return []
    
    formatted_history = []
    for i, msg in enumerate(chat_history):
        try:
            # BULLETPROOF: Validate message structure
            if not isinstance(msg, dict):
                continue
                
            if "role" not in msg or "content" not in msg:
                continue
            
            role = msg["role"]
            content = msg["content"]
            
            # BULLETPROOF: Validate role and content
            if not isinstance(role, str) or not isinstance(content, str):
                continue
            
            if not role.strip() or not content.strip():
                continue
            
            # BULLETPROOF: Create appropriate message type
            if role.strip().lower() == "user":
                formatted_history.append(HumanMessage(content=content.strip()))
            elif role.strip().lower() == "assistant":
                formatted_history.append(AIMessage(content=content.strip()))
            else:
                continue
                
        except Exception as e:
            continue
    
    return formatted_history


def message_bot(user_input, context, knowledge, document_id, database_overview, chat_history):
    """
    Processes a user message and returns a response from the chatbot.
    
    Args:
        user_input: The user's question or message
        context: Relevant document context from vector search
        knowledge: General knowledge context
        document_id: ID of the document being referenced
        chat_history: Previous conversation history
        
    Returns:
        str: The chatbot's response
    """
    try:
        # BULLETPROOF: Validate all inputs
        user_input, context, knowledge, database_overview, chat_history = _validate_inputs(
            user_input, context, knowledge, database_overview, chat_history
        )
        
        # BULLETPROOF: Validate document_id
        if document_id is None:
            document_id = ""
        elif not isinstance(document_id, str):
            document_id = str(document_id)
        
        # BULLETPROOF: Format chat history with validation
        try:
            formatted_history = _format_chat_history(chat_history)
        except Exception as e:
            formatted_history = []
        
        # BULLETPROOF: Create LLM and chain with error handling
        try:
            llm = _get_openai_client()
            prompt_template = _create_context_aware_prompt(bool(chat_history))
            chain = prompt_template | llm
        except Exception as e:
            return "Entschuldigung, es ist ein Fehler beim Erstellen des AI-Modells aufgetreten."
        
        # BULLETPROOF: Prepare chain parameters with validation
        chain_params = {
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "document_id": document_id,
            "database_overview": database_overview,
            "chat_history": formatted_history,
        }
        
        # BULLETPROOF: Validate all chain parameters
        for key, value in chain_params.items():
            if key == "chat_history" or key == "database_overview":
                if not isinstance(value, list):
                    chain_params[key] = []
            else:
                if not isinstance(value, str):
                    chain_params[key] = str(value) if value is not None else ""
        
        # STRUKTURIERTE AUSGABE - Nur die wichtigsten Informationen
        print("\n" + "="*80)
        print("CONTEXT:")
        print("-" * 40)
        print(context)  # Vollständiger Context ohne Kürzung
        
        print("\n" + "-" * 40)
        print("SYSTEM PROMPT:")
        print("-" * 40)
        # Erstelle den finalen System Prompt
        has_history = bool(chat_history)
        base_prompt = "Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität."
        
        if has_history:
            history_section = """

🔄 CHAT HISTORY BEACHTUNG - WICHTIG:
Du siehst eine Chat History mit vorherigen Nachrichten. BERÜCKSICHTIGE diese aktiv:
- Beziehe dich auf vorherige Fragen und Antworten
- Nutze den Kontext aus früheren Nachrichten
- Wenn der Nutzer "dazu", "darüber", "das" oder ähnliche Bezugswörter verwendet, beziehe dich auf vorherige Themen
- Beantworte Rückfragen oder Nachfragen basierend auf dem bisherigen Gesprächsverlauf
- Vermeide Wiederholungen bereits gegebener Antworten, es sei denn, es wird explizit verlangt
- Erkenne den Kontext der aktuellen Frage im Zusammenhang mit der Chat History"""
        else:
            history_section = """

📝 NEUE UNTERHALTUNG:
Dies ist der Beginn einer neuen Unterhaltung ohne vorherige Chat History."""

        sources_section = f"""

DEINE ANTWORTQUELLEN:
Diese Informationen stehen dir zur Verfügung:

HOCHSCHULSPEZIFISCHE INFORMATIONEN:
{context}

ZUSÄTZLICHES WISSEN:
{knowledge}



DOKUMENTEN-ID: {document_id}

VERHALTEN:
- Stütze deine Antworten auf die bereitgestellten Quellen
- Antworte natürlich und direkt, als würdest du mit Studierenden sprechen
- Bei Widersprüchen: Bevorzuge hochschulspezifische Informationen
- Bei fehlenden Informationen: Sage es klar und biete Hilfe an
- Gib ausführliche, aber präzise Antworten
- Achtung: Das was im Context steht ist nicht zwingend richtig und relevant, achte darauf dass der context zu der Frage passt.
- Verwende innerhalb der "answer" kein "" sondern nur ''


WICHTIGE REGEL - NIEMALS ÜBER KONTEXT SPRECHEN:
- Sprich NIEMALS über "verfügbare Dokumente", "Kontext" oder "Informationen in den Dokumenten"
- Sage NIEMALS "Basierend auf den verfügbaren Informationen..." oder ähnliches
- Wenn du nicht genug Informationen hast, stelle stattdessen PRÄZISE RÜCKFRAGEN
- Beispiel: Statt "Ich habe keine Informationen dazu" → "Auf welchen Studiengang bezieht sich deine Frage?"
- Beispiel: Statt "In den Dokumenten steht..." → Antworte direkt mit der Information

ANTWORTFORMAT:
{{
  "answer": "Deine ausführliche Antwort hier",
  "document_id": "{document_id}",
  "source": "Originaltext aus dem Kontext, der die Antwort stützt, korrekt formatiert."
}}"""

        full_system_prompt = base_prompt + history_section + sources_section
        print(full_system_prompt[:800] + "..." if len(full_system_prompt) > 800 else full_system_prompt)
        print("="*80 + "\n")
        
        # BULLETPROOF: Invoke chain with comprehensive error handling
        try:
            response = chain.invoke(chain_params)
        except Exception as e:
            return "Entschuldigung, es ist ein Fehler bei der AI-Verarbeitung aufgetreten."
        
        # BULLETPROOF: Validate response - always return something
        if not response or not hasattr(response, 'content') or not response.content or not isinstance(response.content, str):
            return "Entschuldigung, ich konnte keine gültige Antwort generieren."
            
        return response.content
        
    except Exception as e:
        return "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


def message_bot_stream(user_input, context, knowledge, document_id, database_overview, chat_history):
    """
    Streaming version of message_bot that yields real-time response chunks.
    
    Args:
        user_input: The user's question or message
        context: Relevant document context from vector search  
        knowledge: General knowledge context
        document_id: ID of the document being referenced
        database_overview: Overview of available documents
        chat_history: Previous conversation history
        
    Yields:
        str: Response chunks from the language model
    """
    try:
        user_input, context, knowledge, database_overview, chat_history = _validate_inputs(
            user_input, context, knowledge, database_overview, chat_history
        )
        
        formatted_history = _format_chat_history(chat_history)
        
        # Create streaming LLM and chain with context-aware prompt
        llm = _get_openai_client(streaming=True)
        prompt_template = _create_context_aware_prompt(bool(chat_history))
        
        chain = prompt_template | llm
        
        # Debug: Zeige welche Parameter tatsächlich an den Chain weitergegeben werden
        chain_params = {
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "document_id": document_id,
            "database_overview": database_overview,
            "chat_history": formatted_history,
        }
        
        # Stream the response
        for chunk in chain.stream(chain_params):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
        
    except ValueError:
        yield "Entschuldigung, ich konnte Ihre Nachricht nicht verstehen."
    except Exception as e:
        yield "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


