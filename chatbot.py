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
        model="gpt-4o-mini",
        api_key=api_key,
        streaming=streaming
    )


def _log_prompt_parameters(user_input, context, knowledge, database_overview, document_id, chat_history):
    """
    Logs all parameters that are being passed to the prompt for debugging.
    """
    print("\n" + "="*80)
    print("DEBUG: Parameter für den Systemprompt:")
    print("="*80)
    print(f"USER INPUT: {user_input}")
    print(f"DOCUMENT ID: {document_id}")
    print(f"CONTEXT LENGTH: {len(context) if context else 0} Zeichen")
    print(f"CONTEXT PREVIEW: {context[:200] if context else 'Kein Context'}...")
    print(f"KNOWLEDGE LENGTH: {len(knowledge) if knowledge else 0} Zeichen")
    print(f"KNOWLEDGE PREVIEW: {knowledge[:200] if knowledge else 'Kein Knowledge'}...")
    print(f"DATABASE OVERVIEW LENGTH: {len(database_overview) if database_overview else 0} Zeichen")
    print(f"DATABASE OVERVIEW PREVIEW: {database_overview[:200] if database_overview else 'Kein Database Overview'}...")
    print(f"CHAT HISTORY LÄNGE: {len(chat_history) if chat_history else 0} Nachrichten")
    print("="*80)
    print("\n")


def get_bot():
    """
    Creates and configures the chatbot chain with prompt template.
    
    Returns:
        Chain: Configured LangChain pipeline for the university chatbot
    """
    llm = _get_openai_client()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität. Deine Antworten basieren ausschließlich auf folgenden Quellen:  

HOCHSCHULSPEZIFISCHE INFORMATIONEN:
Modulhandbücher, Studien- und Prüfungsordnungen, Ablaufpläne oder interne Regelungen der Universität:
{context}

ZUSÄTZLICHES WISSEN:
{knowledge}

VERFÜGBARE DOKUMENTE:
Übersicht der verfügbaren Dokumente in der Datenbank:
{database_overview}

DOKUMENTEN-ID:  
Die Dokumenten-ID ({document_id}) übernimmst du exakt 1:1 in das Feld `document_id` deiner Antwort.  

Wichtige Regeln für dein Verhalten:  
- Antwortgrundlage: Stütze deine Antworten ausschließlich auf die bereitgestellten Quellen. Erfinde keine Inhalte und spekuliere nicht. Nutze alle verfügbaren Informationen aus Context, Knowledge und der Dokumentübersicht.
- Natürlicher Ton: Antworte so natürlich und flüssig wie möglich, als würdest du direkt mit einem Studierenden oder Mitarbeitenden der Universität sprechen. Erwähne keine internen Prozesse wie hochgeladene Dokumente, Datenbanken oder Quellenverarbeitung, um die Kommunikation nutzerfreundlich zu halten.  
- Widersprüche: Wenn sich allgemeines und spezifisches Wissen widersprechen, weise höflich darauf hin und beziehe dich auf die hochschulspezifischen Informationen, ohne Annahmen zu treffen.  
- Fehlende Informationen: Wenn die Quellen keine ausreichenden Informationen enthalten, teile dies freundlich und klar mit und biete an, bei einer genaueren Anfrage weiterzuhelfen.  
- Antwortstil: Antworte klar, professionell und verständlich. Gib ausführliche, aber präzise Antworten, die die Frage umfassend beantworten, ohne überflüssige Details. Vermeide zu kurze Antworten (z. B. nur vier Wörter auf ausführliche Fragen).  
- Rückfragen: Stelle Rückfragen, wenn die Nutzeranfrage unklar oder unvollständig ist, um die Anfrage präzise zu beantworten.  
- Chat History: Beachte die Chat History. Beantworte offene Fragen aus früheren Nachrichten, wenn sie noch nicht beantwortet wurden. Vermeide es, bereits beantwortete Fragen erneut zu beantworten, es sei denn, es wird explizit verlangt.  
- Struktur der Antwort: Gib die Antwort im folgenden JSON-Format zurück:  
  
  {{
    "answer": "Hier steht die ausführliche, aber präzise Antwort auf die Frage.",
    "document_id": "{document_id}",
    "source": "Hier steht der Originaltext oder Satz aus dem Kontext, der die Antwort stützt (1:1 übernommen)."
  }}
                    """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "{input}",
            ),
        ]
    )

    return prompt_template | llm


def _validate_inputs(user_input, context, knowledge, chat_history):
    """
    Validates and sanitizes input parameters for message processing.
    
    Args:
        user_input: User's message
        context: Document context
        knowledge: General knowledge context
        chat_history: Previous conversation history
        
    Returns:
        tuple: Validated and sanitized inputs
    """
    if not user_input or not isinstance(user_input, str):
        raise ValueError("User input must be a non-empty string")
    
    if not isinstance(chat_history, list):
        chat_history = []
        
    if not context or not isinstance(context, str):
        context = ""

    if not knowledge or not isinstance(knowledge, str):
        knowledge = ""
        
    return user_input.strip(), context, knowledge, chat_history


def _format_chat_history(chat_history):
    """
    Converts chat history to LangChain message format.
    
    Args:
        chat_history: List of chat messages with role and content
        
    Returns:
        list: Formatted chat history for LangChain
    """
    formatted_history = []
    for msg in chat_history:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            continue
            
        if msg["role"] == "user":
            formatted_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_history.append(AIMessage(content=msg["content"]))
    
    return formatted_history


def message_bot(user_input, context, knowledge, database_overview, document_id, chat_history):
    """
    Processes a user message and returns a response from the chatbot.
    
    Args:
        user_input: The user's question or message
        context: Relevant document context from vector search
        knowledge: General knowledge context
        database_overview: Overview of available documents in the namespace
        document_id: ID of the document being referenced
        chat_history: Previous conversation history
        
    Returns:
        str: The chatbot's response
    """
    try:
        user_input, context, knowledge, chat_history = _validate_inputs(
            user_input, context, knowledge, chat_history
        )
        
        # Debug logging hinzugefügt
        _log_prompt_parameters(user_input, context, knowledge, database_overview, document_id, chat_history)
        
        formatted_history = _format_chat_history(chat_history)
        chain = get_bot()
        
        # Debug: Zeige welche Parameter tatsächlich an den Chain weitergegeben werden
        chain_params = {
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "database_overview": database_overview,
            "document_id": document_id,
            "chat_history": formatted_history,
        }
        
        print("CHAIN PARAMETER:")
        print(f"Input: {chain_params['input']}")
        print(f"Context (erste 300 Zeichen): {chain_params['context'][:300] if chain_params['context'] else 'LEER'}...")
        print(f"Knowledge (erste 200 Zeichen): {chain_params['knowledge'][:200] if chain_params['knowledge'] else 'LEER'}...")
        print(f"Database Overview (erste 200 Zeichen): {str(chain_params['database_overview'])[:200] if chain_params['database_overview'] else 'LEER'}...")
        print(f"Document ID: {chain_params['document_id']}")
        print(f"Chat History Länge: {len(chain_params['chat_history'])}")
        print("\n")
        
        response = chain.invoke(chain_params)
        
        if not response or not hasattr(response, 'content'):
            return "Entschuldigung, ich konnte keine Antwort generieren."
            
        print("AI RESPONSE:", response.content)
        return response.content
        
    except ValueError as ve:
        print(f"Validation error in message_bot: {str(ve)}")
        return "Entschuldigung, ich konnte Ihre Nachricht nicht verstehen."
    except Exception as e:
        print(f"Error in message_bot: {str(e)}")
        return "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


def message_bot_stream(user_input, context, knowledge, database_overview, document_id, chat_history):
    """
    Streaming version of message_bot that yields real-time response chunks.
    
    Args:
        user_input: The user's question or message
        context: Relevant document context from vector search  
        knowledge: General knowledge context
        database_overview: Overview of available documents in the namespace
        document_id: ID of the document being referenced
        chat_history: Previous conversation history
        
    Yields:
        str: Response chunks from the language model
    """
    try:
        user_input, context, knowledge, chat_history = _validate_inputs(
            user_input, context, knowledge, chat_history
        )
        
        # Debug logging hinzugefügt
        _log_prompt_parameters(user_input, context, knowledge, database_overview, document_id, chat_history)
        
        formatted_history = _format_chat_history(chat_history)
        
        # Create streaming LLM and chain
        llm = _get_openai_client(streaming=True)
        chain = get_bot()
        
        # Debug: Zeige welche Parameter tatsächlich an den Chain weitergegeben werden
        chain_params = {
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "database_overview": database_overview,
            "document_id": document_id,
            "chat_history": formatted_history,
        }
        
        print("STREAM CHAIN PARAMETER:")
        print(f"Input: {chain_params['input']}")
        print(f"Context (erste 300 Zeichen): {chain_params['context'][:300] if chain_params['context'] else 'LEER'}...")
        print(f"Knowledge (erste 200 Zeichen): {chain_params['knowledge'][:200] if chain_params['knowledge'] else 'LEER'}...")
        print(f"Database Overview (erste 200 Zeichen): {str(chain_params['database_overview'])[:200] if chain_params['database_overview'] else 'LEER'}...")
        print(f"Document ID: {chain_params['document_id']}")
        print(f"Chat History Länge: {len(chain_params['chat_history'])}")
        print("\n")
        
        # Stream the response
        for chunk in chain.stream(chain_params):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
        
    except ValueError:
        yield "Entschuldigung, ich konnte Ihre Nachricht nicht verstehen."
    except Exception as e:
        print(f"Error in message_bot_stream: {str(e)}")
        yield "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


