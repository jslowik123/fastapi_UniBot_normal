### Server für den normalen Chat-Modus

Der Server läuft auf Port `9000`.

```bash
# Virtuelle Umgebung erstellen
python3 -m venv venv

# Virtuelle Umgebung aktivieren
source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Server starten
./run_local.sh
