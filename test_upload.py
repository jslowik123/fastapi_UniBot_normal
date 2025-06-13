#!/usr/bin/env python3
"""
Test script to verify PDF upload functionality.
"""

import os
from dotenv import load_dotenv
from doc_processor import DocProcessor

# Load environment variables
load_dotenv()

def test_upload():
    """Test the upload functionality with the PDF in the directory."""
    
    # Check if API keys are available
    pinecone_key = os.getenv("PINECONE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not pinecone_key or not openai_key:
        print("❌ API Keys fehlen!")
        print(f"PINECONE_API_KEY: {'✅' if pinecone_key else '❌'}")
        print(f"OPENAI_API_KEY: {'✅' if openai_key else '❌'}")
        return
    
    print("✅ API Keys gefunden")
    
    # Initialize processor
    try:
        processor = DocProcessor(pinecone_key, openai_key)
        print("✅ DocProcessor initialisiert")
    except Exception as e:
        print(f"❌ Fehler beim Initialisieren: {e}")
        return
    
    # Test with the PDF in the directory
    pdf_path = "PO_Master_WiWi.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF nicht gefunden: {pdf_path}")
        return
    
    print(f"✅ PDF gefunden: {pdf_path}")
    
    # Process the PDF
    namespace = "test_namespace"
    file_id = "test_upload_id"
    
    print("🚀 Starte Upload...")
    try:
        result = processor.process_pdf(pdf_path, namespace, file_id)
        
        print("\n" + "="*50)
        print("UPLOAD ERGEBNIS:")
        print("="*50)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Message: {result.get('message', 'no message')}")
        print(f"Chunks: {result.get('chunks', 0)}")
        
        if result.get('pinecone_result'):
            print(f"Pinecone Status: {result['pinecone_result'].get('status', 'unknown')}")
        
        if result.get('firebase_result'):
            print(f"Firebase Status: {result['firebase_result'].get('status', 'unknown')}")
            
        print("="*50)
        
        if result.get('status') == 'success':
            print("✅ Upload erfolgreich!")
        else:
            print("❌ Upload fehlgeschlagen!")
            
    except Exception as e:
        print(f"❌ Upload Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upload()
