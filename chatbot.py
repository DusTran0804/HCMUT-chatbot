import os
import sys
import shutil
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

import os
os.environ["HF_HUB_OFFLINE"] = "1"
import huggingface_hub.utils as hf_utils
hf_utils.logging.set_verbosity_error()

from langchain_community.chat_models import ChatOllama
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import transformers
transformers.logging.set_verbosity_error()

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(current_dir, "chroma_db")

class WordWrapCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.word_buffer = ""
        self.current_line_len = 9  
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns - 2

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        for char in token:
            if char.isspace():
                word_len = len(self.word_buffer)
                if word_len > 0:
                    if self.current_line_len + word_len >= self.terminal_width:
                        sys.stdout.write("\n")
                        self.current_line_len = 0
                    
                    sys.stdout.write(self.word_buffer)
                    self.current_line_len += word_len
                    self.word_buffer = ""
                sys.stdout.write(char)
                if char == "\n":
                    self.current_line_len = 0
                else:
                    self.current_line_len += 1
            else:
                self.word_buffer += char
        sys.stdout.flush()

    def on_llm_end(self, *args, **kwargs) -> None:
        if self.word_buffer:
            if self.current_line_len + len(self.word_buffer) >= self.terminal_width:
                sys.stdout.write("\n")
            sys.stdout.write(self.word_buffer)
            self.word_buffer = ""
        sys.stdout.flush()

def init_chatbot():
    print("ChatBot Loading...", end="", flush=True)
    import io
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30
            }
        )

        llm = ChatOllama(model="llama3", temperature=0.2)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    print(" Done!")
    
    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved context to "
        "answer the question. If you don't know the answer, say that you don't know.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def chat_loop():
    if not os.path.exists(persist_directory):
         print("ERROR: Vector database not found. Please run `python ingest.py` first!")
         return
        
    try:
        rag_chain = init_chatbot()
    except Exception as e:
        print(f"Error initializing chatbot. Make sure Ollama is running! Details: {e}")
        return
        
    print("\n" + "="*50)
    print("Chatbot is already!")
    print("Please ask. Type 'exit' or 'quit' to out.")
    print("="*50 + "\n")
    
    # Enter the loop
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
            
        try:
            print("Thinking...")
            print("ChatBot:\n", end="", flush=True)
            rag_chain.invoke(
                {"input": user_input},
                config={"callbacks": [WordWrapCallbackHandler()]}
            )
            print("\n")
            
        except Exception as e:
            print(f"Error {e}")
            print("(`ollama pull llama3`)?")

if __name__ == "__main__":
    chat_loop()
    
