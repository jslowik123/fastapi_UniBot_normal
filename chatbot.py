from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from pinecon_con import PineconeCon


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
                """Du bist ein Chatbot namens "UniBot", entwickelt für Studierende und Studieninteressierte. Deine Aufgabe ist es, präzise und hilfreiche Antworten zu Fragen rund um Universitäten, Studiengänge und akademische Fachsprache zu geben. Du verfügst über umfassendes Wissen zu:
                    - Studienstrukturen (Bachelor, Master, ECTS, Semester etc.),
                    - Fachbereichen (z. B. Naturwissenschaften, Geisteswissenschaften),
                    - Zugangsvoraussetzungen (Abitur, NC, Eignungstests),
                    - akademischer Fachsprache (z. B. Immatrikulation, Modulhandbuch, Prüfungsordnung).

                    Antworte immer:
                    - in einfacher, verständlicher Sprache, es sei denn, Fachsprache wird explizit verlangt,
                    - mit Beispielen oder Erklärungen, wenn es die Frage sinnvoll ergänzt,
                    - neutral und faktenbasiert, ohne persönliche Meinungen.

                    Falls du Informationen aus spezifischen Dokumenten wie Studienordnungen oder Modulhandbüchern benötigst,guck ob sie in dem context findest:  <START_OF_CONTEXT>\n{context}\n<END_OF_CONTEXT>\n\n und simuliere eine Antwort basierend darauf. Wenn eine Frage unklar ist, stelle Rückfragen, um die Anfrage zu präzisieren."""

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

def message_bot(user_input, context, chat_history):
    chain = get_bot()
    response = chain.invoke({"input": user_input, "context": context, "chat_history": chat_history})
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
    print("AI:" + response.content)
    return response.content

