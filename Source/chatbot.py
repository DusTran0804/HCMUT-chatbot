import os
import sys
import shutil
import warnings
from dotenv import load_dotenv

# Load các biến môi trường từ file .env nếu có (cho chạy local)
load_dotenv()

# Tắt các cảnh báo DeprecationWarning và UserWarning
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
logging.getLogger("langchain").setLevel(logging.ERROR)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Setup paths - Lấy thư mục cha của Source/ làm gốc cho chroma_db
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(os.path.dirname(current_dir), "chroma_db")

class WordWrapCallbackHandler(BaseCallbackHandler):
    """Callback tùy chỉnh gom các chữ cái thành từ để không bị cắt đôi khi chat terminal."""
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
    
    # Hide all output from model loading
    import io
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        # Load API Key (lấy từ env var trên Render hoặc file .env local)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY not found in environment!")

        # 1. Load Vector DB dùng Google Embeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        
        # 2. Retriever (MMR strategy)
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 30}
        )
        
        # 3. Sử dụng Gemini Pro (Ổn định nhất)
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    print(" Done!")
    
    # 3. Prompt Template
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
    
    # 4. Chain it all together
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def chat_loop():
    if not os.path.exists(persist_directory):
         print(f"ERROR: Database vector không tìm thấy tại {persist_directory}!")
         return
  
    try:
        rag_chain = init_chatbot()
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return
        
    print("\n🤖 Chatbot đã sẵn sàng (Gemini Cloud mode)!")
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ['exit', 'quit']: break
        if not user_input.strip(): continue
        try:
            print("ChatBot:\n", end="", flush=True)
            rag_chain.invoke(
                {"input": user_input},
                config={"callbacks": [WordWrapCallbackHandler()]}
            )
            print("\n")
        except Exception as e:
            print(f"Lỗi: {e}")

if __name__ == "__main__":
    chat_loop()
