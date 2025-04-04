from pinecon_con import PineconeCon
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from chatbot import get_bot
import uvicorn
from fastapi import FastAPI
app = FastAPI()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    
    
        

