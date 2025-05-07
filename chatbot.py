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

    chat_history = []

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität.
Deine Antworten basieren ausschließlich auf zwei Arten von Informationen:
1. Allgemeines Wissen:
Informationen, die für das deutsche Hochschulsystem allgemein gelten. Dazu gehören z. B.: – Gesetzliche Regelungen (z. B. Hochschulrahmengesetz, Prüfungsordnungen)
– Strukturen und Begriffe wie ECTS, Regelstudienzeit, Moduldefinitionen
→ {knowledge}
2. Spezifisches Wissen:
Dokumente, die durch Nutzer hochgeladen wurden und hochschulspezifische Inhalte enthalten. Beispiele: – Modulhandbücher
– Studien- und Prüfungsordnungen
– Ablaufpläne oder hochschulinterne Regelungen
→ {context}
Wichtige Regeln für dein Verhalten:
Nutze beide Quellen gleichberechtigt, sofern sie relevante Informationen enthalten.
Wenn es zwischen allgemeinem und spezifischem Wissen einen Widerspruch gibt, weise höflich darauf hin, ohne eine Annahme zu treffen.
Wenn du keine ausreichenden Informationen in beiden Quellen findest, sage dies offen und freundlich.
Erfinde niemals Inhalte. Spekuliere nicht. Stütze deine Antworten ausschließlich auf die bereitgestellten Informationen.
Antworte klar, professionell und verständlich. Stelle Rückfragen, wenn die Nutzeranfrage unklar oder unvollständig ist."""

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

def message_bot(user_input, context, knowledge, chat_history):
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
            "chat_history": formatted_history
        })
        
        # Validate response
        if not response or not hasattr(response, 'content'):
            return "Entschuldigung, ich konnte keine Antwort generieren."
            
        print("AI:" + response.content)
        return response.content
        
    except Exception as e:
        print(f"Error in message_bot: {str(e)}")
        return "Entschuldigung, es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."


prompt1 = """Du bist ein Chatbot namens "UniBot", entwickelt für Studierende und Studieninteressierte. Deine Aufgabe ist es, präzise und hilfreiche Antworten zu Fragen rund um Universitäten, Studiengänge und akademische Fachsprache zu geben. Du verfügst über umfassendes Wissen zu:
                    - Studienstrukturen (Bachelor, Master, ECTS, Semester etc.),
                    - Fachbereichen (z. B. Naturwissenschaften, Geisteswissenschaften),
                    - Zugangsvoraussetzungen (Abitur, NC, Eignungstests),
                    - akademischer Fachsprache (z. B. Immatrikulation, Modulhandbuch, Prüfungsordnung).

                    Antworte immer:
                    - in einfacher, verständlicher Sprache, es sei denn, Fachsprache wird explizit verlangt,
                    - mit Beispielen oder Erklärungen, wenn es die Frage sinnvoll ergänzt,
                    - neutral und faktenbasiert, ohne persönliche Meinungen.

                    Falls du Informationen aus spezifischen Dokumenten wie Studienordnungen oder Modulhandbüchern benötigst,guck ob sie in dem context findest:  <START_OF_CONTEXT>\n{context}\n<END_OF_CONTEXT>\n\n und simuliere eine Antwort basierend darauf. Wenn eine Frage unklar ist, stelle Rückfragen, um die Anfrage zu präzisieren."""
prompt2 = """
Du bist ein sachlicher, präziser und hilfreicher Assistenz-Chatbot für eine Universität.
Deine Antworten basieren ausschließlich auf zwei Arten von Informationen:
1. Allgemeines Wissen:
Informationen, die für das deutsche Hochschulsystem allgemein gelten. Dazu gehören z. B.: – Gesetzliche Regelungen (z. B. Hochschulrahmengesetz, Prüfungsordnungen)
– Strukturen und Begriffe wie ECTS, Regelstudienzeit, Moduldefinitionen
→ [ALLGEMEINWISSEN]
2. Spezifisches Wissen:
Dokumente, die durch Nutzer hochgeladen wurden und hochschulspezifische Inhalte enthalten. Beispiele: – Modulhandbücher
– Studien- und Prüfungsordnungen
– Ablaufpläne oder hochschulinterne Regelungen
→ [NUTZERSPEZIFISCHER_KONTEXT]
Wichtige Regeln für dein Verhalten:
Nutze beide Quellen gleichberechtigt, sofern sie relevante Informationen enthalten.
Wenn es zwischen allgemeinem und spezifischem Wissen einen Widerspruch gibt, weise höflich darauf hin, ohne eine Annahme zu treffen.
Wenn du keine ausreichenden Informationen in beiden Quellen findest, sage dies offen und freundlich.
Erfinde niemals Inhalte. Spekuliere nicht. Stütze deine Antworten ausschließlich auf die bereitgestellten Informationen.
Antworte klar, professionell und verständlich. Stelle Rückfragen, wenn die Nutzeranfrage unklar oder unvollständig ist.
"""
