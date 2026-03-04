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
    """Callback tùy chỉnh gom các chữ cái thành từ (word) để không bị cắt làm đôi khi chạm mép màn hình terminal."""
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
        # 1. Load the database we created in `ingest.py`
        embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_model)
        
        # Create a retriever
        # Using simple similarity to ensure contiguous chunks like lists are kept together
        # Sử dụng MMR: Rút 30 mảnh từ database, sau đó lọc ra 8 mảnh đa dạng nhất đưa cho AI
        # Giữ tốc độ nhanh mà vẫn bao quát nhiều thông tin
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 30
            }
        )
        
        # 2. Initialize the Local LLM via Ollama
        llm = ChatOllama(model="llama3", temperature=0.2)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
    print(" Done!")
    
    # 3. Create the Prompt Template
    system_prompt = (
        "Bạn là một chuyên gia AI trả lời câu hỏi cực kỳ chi tiết, rõ ràng và logic. "
        "CHỈ sử dụng các đoạn trích xuất (ngữ cảnh) dưới đây để trả lời câu hỏi. "
        "Đọc ngữ cảnh RẤT cẩn thận. Bạn phải đặc biệt chú ý đến các con số một cách thật chính xác. "
        "Nếu người dùng hỏi 'bao nhiêu' và có một danh sách danh mục thay vì tổng số lượng, hãy tự đếm các mục trong danh sách để đưa ra câu trả lời cuối cùng. "
        "Nếu tính chất câu hỏi liệt kê nhiều loại, hãy trình bày rõ ràng bằng các gạch đầu dòng. "
        "Nếu câu trả lời không có trong ngữ cảnh, hãy nói 'Tôi không biết'. Tuyệt đối không đoán hoặc tự bịa ra thông tin. "
        "Luôn luôn trả lời bằng TIẾNG VIỆT một cách tự nhiên. Độ dài câu trả lời phụ thuộc vào độ phức tạp của câu hỏi, hãy phân tích tất cả các góc cạnh có trong văn bản."
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
    # Make sure the DB exists
    if not os.path.exists(persist_directory):
         print("ERROR: Vector database not found. Please run `python ingest.py` first!")
         return
         
    # Init everything
    try:
        rag_chain = init_chatbot()
    except Exception as e:
        print(f"Error initializing chatbot. Make sure Ollama is running! Details: {e}")
        return
        
    print("\n" + "="*50)
    print("🤖 Chatbot đã sẵn sàng!")
    print("Hãy đặt câu hỏi. Gõ 'exit' hoặc 'quit' để thoát.")
    print("="*50 + "\n")
    
    # Enter the loop
    while True:
        user_input = input("Bạn: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
            
        try:
            print("Đang suy nghĩ...")
            print("ChatBot:\n", end="", flush=True)
            rag_chain.invoke(
                {"input": user_input},
                config={"callbacks": [WordWrapCallbackHandler()]}
            )
            print("\n")
            
        except Exception as e:
            print(f"Lỗi khi trả lời câu hỏi: {e}")
            print("Ứng dụng Ollama đã chạy chưa và bạn đã tải mô hình llama3 chưa (`ollama pull llama3`)?")

if __name__ == "__main__":
    chat_loop()
    
