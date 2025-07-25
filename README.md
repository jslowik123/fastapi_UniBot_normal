### Server für den normalen ChatModus

Der Server läuft auf Port `9000`.
Siehe `.env.example` für benötigte Umgebungsvariablen.

```bash
# Virtuelle Umgebung aktivieren
source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Server starten
./run_local.sh
