from firebase_connection import FirebaseConnection
from dotenv import load_dotenv
import os
import uuid

# Umgebungsvariablen laden
load_dotenv()

def test_append_metadata():
    # Firebase-Verbindung herstellen
    firebase = FirebaseConnection()
    
    # Test-Parameter
    namespace = "ww"
    fileID=  "-OPKhQmbfZczoukQINy6"
    chunk_count = 5
    keywords = ["Test", "Firebase", "Metadata"]
    summary = "Dies ist ein Test-Dokument zum Testen der append_metadata Methode."
    
    # Metadaten anhängen
    result = firebase.append_metadata(
        namespace=namespace,
        fileID=fileID,
        chunk_count=chunk_count,
        keywords=keywords,
        summary=summary
    )
    
    # Ergebnis ausgeben
    print("Ergebnis:")
    print(f"Status: {result['status']}")
    print(f"Nachricht: {result['message']}")
    print(f"Pfad: {result.get('path', 'Kein Pfad angegeben')}")
    
    # Prüfen, ob die Daten abrufbar sind
    fetch_result = firebase.get_document_metadata(
        namespace=namespace,
        fileID=fileID
    )
    
    print("\nAbgerufene Daten:")
    if fetch_result['status'] == 'success':
        print(f"Chunk Count: {fetch_result['data'].get('chunk_count')}")
        print(f"Keywords: {fetch_result['data'].get('keywords')}")
        print(f"Summary: {fetch_result['data'].get('summary')}")
    else:
        print(f"Fehler: {fetch_result['message']}")

if __name__ == "__main__":
    test_append_metadata() 



 