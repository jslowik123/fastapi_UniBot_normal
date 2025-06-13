#!/usr/bin/env python3
"""
Test script für den OpenAI Assistant Agent.

Dieses Script demonstriert die Verwendung des neuen OpenAI Assistant
mit Tools und Function Calling für Universitätsdokument-Verarbeitung.
"""

import asyncio
import json
from openai_agent import UniversityAgent

async def test_agent():
    """Test the OpenAI Assistant functionality."""
    
    print("🤖 Initialisiere OpenAI Assistant...")
    
    try:
        # Initialize agent
        agent = UniversityAgent()
        print(f"✅ Assistant erstellt: {agent.assistant.id}")
        
        # Get agent info
        agent_info = agent.get_agent_info()
        print(f"📋 Agent Info: {json.dumps(agent_info, indent=2)}")
        
        # Test namespace (you'll need to replace this with an actual namespace)
        test_namespace = "test"  # Replace with your actual namespace
        test_query = "Was sind die Zugangsvoraussetzungen?"
        
        print(f"\n💬 Teste Nachricht: '{test_query}'")
        print(f"📁 Namespace: '{test_namespace}'")
        
        # Test regular message processing
        print("\n🔄 Verarbeite Nachricht (synchron)...")
        result = agent.process_message(test_query, test_namespace)
        
        if result.get("status") == "success":
            print("✅ Antwort erhalten:")
            print(f"📝 Response: {result['response']}")
            print(f"🆔 Thread ID: {result.get('thread_id', 'N/A')}")
        else:
            print(f"❌ Fehler: {result.get('message', 'Unknown error')}")
        
        print("\n" + "="*60)
        
        # Test streaming
        print("\n🌊 Teste Streaming-Antwort...")
        agent.reset_conversation()  # Reset for clean test
        
        async for chunk in agent.process_message_stream(test_query, test_namespace):
            chunk_type = chunk.get("type", "unknown")
            
            if chunk_type == "error":
                print(f"❌ Stream Error: {chunk.get('message')}")
                break
            elif chunk_type == "status":
                print(f"🔄 Status: {chunk.get('message')}")
            elif chunk_type == "tool_call":
                print(f"🔧 Tool Call: {chunk.get('function')} - {chunk.get('arguments')}")
            elif chunk_type == "complete":
                print(f"✅ Stream Complete!")
                print(f"📝 Final Response: {chunk.get('content')}")
                break
        
        print("\n" + "="*60)
        
        # Get conversation history
        print("\n📚 Konversationshistorie:")
        history = agent.get_conversation_history()
        for i, msg in enumerate(history):
            role = msg['role']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"{i+1}. [{role.upper()}]: {content}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_tools_description():
    """Show available tools and their descriptions."""
    print("\n🛠️ Verfügbare Tools:")
    print("1. get_namespace_overview - Ruft alle verfügbaren Dokumente ab")
    print("2. find_relevant_document - Findet das passende Dokument für eine Anfrage")
    print("3. search_document_content - Durchsucht Dokumentinhalte")
    print("\n🔄 Workflow:")
    print("Benutzer → get_namespace_overview → find_relevant_document → search_document_content → Antwort")

async def test_individual_functions():
    """Test individual tool functions."""
    print("\n🧪 Teste einzelne Tool-Funktionen...")
    
    agent = UniversityAgent()
    
    # Test get_namespace_overview
    print("\n1. Teste get_namespace_overview...")
    result = agent._execute_function("get_namespace_overview", {"namespace": "test_uni"})
    print(f"Result: {result[:200]}...")
    
    # Test with invalid namespace
    print("\n2. Teste mit ungültigem Namespace...")
    result = agent._execute_function("get_namespace_overview", {"namespace": "invalid_namespace"})
    print(f"Result: {result}")

if __name__ == "__main__":
    print("🚀 OpenAI Assistant Test")
    print("=" * 50)
    
    test_tools_description()
    
    print("\n⚠️  WICHTIG: Stelle sicher, dass du einen gültigen Namespace hast!")
    print("⚠️  Ersetze 'test_uni' mit deinem echten Namespace in test_agent.py")
    print("\n" + "="*50)
    
    # Run async tests
    asyncio.run(test_agent())
    
    print("\n" + "="*50)
    asyncio.run(test_individual_functions())
    
    print("\n🎉 Test abgeschlossen!") 