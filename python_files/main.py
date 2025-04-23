import uvicorn
from fastapi import FastAPI
app = FastAPI()
from python_files.pinecon_con import PineconeCon
import PyPDF2


if __name__ == "__main__":
    pc = PineconeCon(index_name="quickstart")
    pc.delete_embeddings("test.pdf")
    
        

