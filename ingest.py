import os
import sys
import warnings

# Tắt các cảnh báo DeprecationWarning và UserWarning
warnings.filterwarnings("ignore")
# Tắt log của transformers
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

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(current_dir, "chroma_db")

import shutil

def ingest_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    # Clear out the old database so the bot only remembers the new file
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    print("Chatbot Loading...", end="", flush=True)
    
    # 1. Load the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    
    # 3. Create Vector Store (Database)
    
    # We use a free, local, open-source embedding model from HuggingFace
    import io
    import sys
    
    # Hide all output from model loading
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        
        # Save the chunks to a local Chroma database
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_directory
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    print(" Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_your_text_file>")
        print("Example: python ingest.py my_own_dataset.txt")
    else:
        file_path = sys.argv[1]
        ingest_data(file_path)
