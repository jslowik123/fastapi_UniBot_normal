from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from pinecone_connection import PineconeCon


def get_bot():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    llm = ChatOpenAI(model="gpt-4o-mini")


    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität. Deine Antworten basieren ausschließlich auf zwei Quellen:  
1. Allgemeines Wissen: {knowledge}  
2. Hochschulspezifische Informationen: Modulhandbücher, Studien- und Prüfungsordnungen, Ablaufpläne oder interne Regelungen der Universität. → {context}  

Dokumenten-ID:  
Die Dokumenten-ID ({document_id}) übernimmst du exakt 1:1 in das Feld `document_id` deiner Antwort.  

Datenbankübersicht:  
Du hast Zugriff auf eine Datenbank mit hochschulspezifischen Informationen: {database_overview}  

Wichtige Regeln für dein Verhalten:  
- Antwortgrundlage: Stütze deine Antworten ausschließlich auf die bereitgestellten Quellen (allgemeines und spezifisches Wissen). Erfinde keine Inhalte und spekuliere nicht.  
- Natürlicher Ton: Antworte so natürlich und flüssig wie möglich, als würdest du direkt mit einem Studierenden oder Mitarbeitenden der Universität sprechen. Erwähne keine internen Prozesse wie hochgeladene Dokumente, Datenbanken oder Quellenverarbeitung, um die Kommunikation nutzerfreundlich zu halten.  
- Widersprüche: Wenn sich allgemeines und spezifisches Wissen widersprechen, weise höflich darauf hin und beziehe dich auf die hochschulspezifischen Informationen, ohne Annahmen zu treffen.  
- Fehlende Informationen: Wenn die Quellen keine ausreichenden Informationen enthalten, teile dies freundlich und klar mit und biete an, bei einer genaueren Anfrage weiterzuhelfen.  
- Antwortstil: Antworte klar, professionell und verständlich. Gib ausführliche, aber präzise Antworten, die die Frage umfassend beantworten, ohne überflüssige Details. Vermeide zu kurze Antworten (z. B. nur vier Wörter auf ausführliche Fragen).  
- Rückfragen: Stelle Rückfragen, wenn die Nutzeranfrage unklar oder unvollständig ist, um die Anfrage präzise zu beantworten.  
- Chat History: Beachte die Chat History. Beantworte offene Fragen aus früheren Nachrichten, wenn sie noch nicht beantwortet wurden. Vermeide es, bereits beantwortete Fragen erneut zu beantworten, es sei denn, es wird explizit verlangt.  
- Struktur der Antwort: Gib die Antwort im folgenden JSON-Format zurück:  
  ```json
  {
    "answer": "Hier steht die ausführliche, aber präzise Antwort auf die Frage.",
    "document_id": "{document_id}",
    "source": "Hier steht der Originaltext oder Satz aus dem Kontext, der die Antwort stützt (1:1 übernommen)."
  }
                    """
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                "{input}",
            ),
        ]
    )


    chain = prompt_template | llm
    return chain

def message_bot(user_input, context, knowledge, database_overview, document_id, chat_history):
    try:
        if not user_input or not isinstance(user_input, str):
            return "Entschuldigung, ich konnte Ihre Nachricht nicht verstehen."
        
        if not isinstance(chat_history, list):
            chat_history = []
            
        if not context or not isinstance(context, str):
            context = ""

        if not knowledge or not isinstance(knowledge, str):
            knowledge = ""

        # Convert chat history format
        formatted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))

        # Get bot instance
        chain = get_bot()
        
        # Invoke chain with validated inputs
        response = chain.invoke({
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "database_overview": database_overview,
            "document_id": document_id,
            "chat_history": formatted_history,
            
        })
        
        # Validate response
        if not response or not hasattr(response, 'content'):
            return "Entschuldigung, ich konnte keine Antwort generieren."
            
        print("AI:" + response.content)
        return response.content
        
    except Exception as e:
        print(f"Error in message_bot: {str(e)}")
        return "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."

def message_bot_stream(user_input, context, knowledge, database_overview, document_id, chat_history):
    """
    Streaming version of message_bot that yields real-time chunks from OpenAI
    """
    try:
        if not user_input or not isinstance(user_input, str):
            yield "Entschuldigung, ich konnte Ihre Nachricht nicht verstehen."
            return
        
        if not isinstance(chat_history, list):
            chat_history = []
            
        if not context or not isinstance(context, str):
            context = ""

        if not knowledge or not isinstance(knowledge, str):
            knowledge = ""

        # Convert chat history format
        formatted_history = []
        for msg in chat_history:
            if msg["role"] == "user":
                formatted_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                formatted_history.append(AIMessage(content=msg["content"]))

        # Get streaming bot instance
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            yield "API-Schlüssel nicht gefunden."
            return

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            streaming=True  # Enable streaming
        )

        chain = get_bot()
        
        # Stream the response
        for chunk in chain.stream({
            "input": user_input,
            "context": context,
            "knowledge": knowledge,
            "database_overview": database_overview,
            "document_id": document_id,
            "chat_history": formatted_history,
        }):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
        
    except Exception as e:
        print(f"Error in message_bot_stream: {str(e)}")
        yield "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


