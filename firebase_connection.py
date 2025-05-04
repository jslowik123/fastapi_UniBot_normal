import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from typing import Dict, Any, List
import json
import os
import re

class FirebaseConnection:
    def __init__(self):
        """
        Initialisiert die Firebase-Verbindung über Umgebungsvariablen:
        - FIREBASE_DATABASE_URL: URL der Firebase Realtime Database
        - FIREBASE_CREDENTIALS_PATH: Optionaler Pfad zu einer Credentials-Datei
        - FIREBASE_CREDENTIALS_JSON: Optionaler JSON-String mit Credentials
        """
        database_url = os.getenv('FIREBASE_DATABASE_URL')
        if not database_url:
            raise ValueError("FIREBASE_DATABASE_URL Umgebungsvariable muss gesetzt sein")
            
        if not firebase_admin._apps:
            # Prüfen, ob Credentials als JSON-String oder als Dateipfad vorhanden sind
            credentials_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
            credentials_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
            
            if credentials_json:
                # Aus JSON-String initialisieren
                cred_dict = json.loads(credentials_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
            elif credentials_path and os.path.exists(credentials_path):
                # Aus Datei initialisieren
                cred = credentials.Certificate(credentials_path)
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
            else:
                # Fallback ohne Credentials (nur für öffentliche DB ohne Authentifizierung)
                firebase_admin.initialize_app(options={
                    'databaseURL': database_url
                })
        
        self._db = db

    def append_metadata(self, namespace: str, fileID: str, chunk_count: int, keywords: List[str], summary: str) -> Dict[str, Any]:
        """
        Speichert Metadaten zu einem Dokument in Firebase.
        
        Args:
            namespace: Der Namespace, in dem das Dokument gespeichert ist
            fileID: Die ID des Dokuments
            chunk_count: Die Anzahl der Chunks
            keywords: Liste der Schlüsselwörter
            summary: Zusammenfassung des Dokuments
            
        Returns:
            Dict mit Statusinformationen
        """
        try:
            # Pfad zum Dokument in der Datenbank
            ref = self._db.reference(f'files/{namespace}/{fileID}')
            
            # Bestehende Daten abrufen
            existing_data = ref.get() or {}
            
            # Neue Daten mit bestehenden Daten zusammenführen
            existing_keywords = existing_data.get('keywords', [])
            combined_keywords = list(set(existing_keywords + keywords))
            
            updated_data = {
                'chunk_count': chunk_count,
                'keywords': combined_keywords,
                'summary': summary,
            }
            
            # Nur aktualisieren, wenn ein Feld nicht bereits existiert oder einen neuen Wert hat
            for key, value in updated_data.items():
                if key not in existing_data or existing_data[key] != value:
                    ref.child(key).set(value)
            
            return {
                'status': 'success',
                'message': f'Metadaten für {fileID} erfolgreich aktualisiert',
                'path': f'files/{namespace}/{fileID}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Fehler beim Firebase-Upload: {str(e)}'
            }
    
    def get_document_metadata(self, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Ruft die Metadaten eines Dokuments ab.
        
        Args:
            namespace: Der Namespace, in dem das Dokument gespeichert ist
            fileID: Die ID des Dokuments
            
        Returns:
            Dict mit den Metadaten oder Fehlermeldung
        """
        try:
            ref = self._db.reference(f'files/{namespace}/{fileID}')
            data = ref.get()
            
            if data:
                return {
                    'status': 'success',
                    'data': data
                }
            else:
                return {
                    'status': 'error',
                    'message': f'Keine Metadaten für {fileID} gefunden'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def list_documents(self, namespace: str = None) -> Dict[str, Any]:
        """
        Listet alle Dokumente oder Dokumente in einem bestimmten Namespace auf.
        
        Args:
            namespace: Optionaler Namespace zum Filtern
            
        Returns:
            Dict mit der Liste der Dokumente
        """
        try:
            if namespace:
                ref = self._db.reference(f'files/{namespace}')
            else:
                ref = self._db.reference('files')
                
            data = ref.get()
            
            return {
                'status': 'success',
                'data': data
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
            
    def delete_document_metadata(self, namespace: str, fileID: str) -> Dict[str, Any]:
        """
        Löscht die Metadaten eines Dokuments aus Firebase.
        
        Args:
            namespace: Der Namespace, in dem das Dokument gespeichert ist
            fileID: Die ID des Dokuments
            
        Returns:
            Dict mit Statusinformationen
        """
        try:
            ref = self._db.reference(f'files/{namespace}/{fileID}')
            
            # Prüfen, ob das Dokument existiert
            existing_data = ref.get()
            if not existing_data:
                return {
                    'status': 'error',
                    'message': f'Keine Metadaten für {fileID} gefunden'
                }
                
            # Dokument löschen
            ref.delete()
            
            return {
                'status': 'success',
                'message': f'Metadaten für {fileID} erfolgreich gelöscht',
                'path': f'files/{namespace}/{fileID}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Fehler beim Löschen der Metadaten: {str(e)}'
            }
            
    def delete_namespace_metadata(self, namespace: str) -> Dict[str, Any]:
        """
        Löscht alle Metadaten in einem Namespace aus Firebase.
        
        Args:
            namespace: Der zu löschende Namespace
            
        Returns:
            Dict mit Statusinformationen
        """
        try:
            ref = self._db.reference(f'files/{namespace}')
            
            # Prüfen, ob der Namespace existiert
            existing_data = ref.get()
            if not existing_data:
                return {
                    'status': 'error',
                    'message': f'Namespace {namespace} nicht gefunden'
                }
                
            # Namespace löschen
            ref.delete()
            
            return {
                'status': 'success',
                'message': f'Namespace {namespace} erfolgreich gelöscht',
                'path': f'files/{namespace}'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Fehler beim Löschen des Namespaces: {str(e)}'
            }



