import os
import sys
import shutil
import warnings
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("langchain").setLevel(logging.ERROR)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(os.path.dirname(current_dir), "chroma_db")

class WordWrapCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.word_buffer = ""
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns - 2

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

def init_chatbot():
    print("ChatBot Loading...", end="", flush=True)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No API_KEY found!")
        raise ValueError("GEMINI_API_KEY not found!")
    
    try:
        genai.configure(api_key=api_key)
        model_found = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Model ID: {m.name}")
                model_found = True
        if not model_found:
            print("No Model Found!")
    except Exception as diag_e:
        print(f"\n Model error: {diag_e}")

    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.2)
    
        system_prompt = (
            "Bạn là một chuyên gia AI trả lời câu hỏi cực kỳ chi tiết, rõ ràng và logic. "
            "CHỈ sử dụng các đoạn trích xuất (ngữ cảnh) dưới đây để trả lời câu hỏi. "
            "Nếu câu trả lời không có trong ngữ cảnh, hãy nói 'Tôi không biết'. "
            "Luôn luôn trả lời bằng TIẾNG VIỆT tự nhiên."
            "\n\n--- Ngữ cảnh ---\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
  
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        print(" Done!")
        return rag_chain

    except Exception as e:
        print(f"\n Error: {e}")
        raise e

def chat_loop():
    if not os.path.exists(persist_directory):
         print(f"ERROR: {persist_directory}!")
         return
  
    try:
        rag_chain = init_chatbot()
    except Exception as e:
        print(f"Error: {e}")
        return
        
    print("\n Chatbot Ready!")
    while True:
        user_input = input("Ask: ")
        if user_input.lower() in ['exit', 'quit']: break
        if not user_input.strip(): continue
        try:
            print("ChatBot:\n", end="", flush=True)
            result = rag_chain.invoke({"input": user_input})
            print(result.get("answer", ""))
            print("\n")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()
