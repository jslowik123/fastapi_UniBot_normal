from main import _get_relevant_context

if __name__ == "__main__":
    # Beispiel-Input
    user_input = "Welche Module werden beim PO-Wechsel angerechnet?"
    namespace = "Wirtschaftsinformatik Pruefungsordnung Wechsel"
    history = []

    print("Testaufruf von _get_relevant_context...")
    context, database_overview, document_id, error = _get_relevant_context(user_input, namespace, history)

    print("--- Ergebnis ---")
    print("Context:")
    print(context)
    print("Database Overview:")
    print(database_overview)
    print("Document ID:")
    print(document_id)
    print("Error:")
    print(error) 