import requests
import sys

def test_start_bot():
    # URL des Servers
    url = "http://127.0.0.1:8000/start_bot"
    
    print("\n" + "="*50)
    print("Bot Start Test")
    print("="*50)
    
    try:
        # POST-Request senden
        print("\nSende Anfrage an den Server...")
        response = requests.post(url)
        
        # Antwort ausgeben
        print("\n" + "-"*50)
        print("ANTWORT VOM SERVER:")
        print("-"*50)
        print(f"Status: {'ERFOLGREICH' if response.status_code == 200 else 'FEHLER'}")
        print(f"Nachricht: {response.json()['message']}")
        print("-"*50 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\nFEHLER: Server nicht erreichbar!")
        print("Bitte stelle sicher, dass der Server läuft.")
    except Exception as e:
        print("\nFEHLER aufgetreten:", str(e))

if __name__ == "__main__":
    print("Test-Skript wird gestartet...")
    test_start_bot()
    input("\nDrücke Enter um das Programm zu beenden...") 