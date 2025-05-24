from doc_processor import DocProcessor
import os
if __name__ == "__main__":
    proc = DocProcessor( os.getenv("PINECONE_API_KEY"), os.getenv("OPENAI_API_KEY"),)
    proc.generate_global_summary("was laeuft")
