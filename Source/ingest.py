import os
import sys
from dotenv import load_dotenv

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LangchainDocument

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import ImageDocument

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(os.path.dirname(current_dir), "chroma_db")

import shutil

def ingest_data(file_path):
    if not os.path.exists(file_path):
        return
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    try:
        llama_docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
    except Exception as e:
        return
        
    documents = []
    has_image = any(
        isinstance(d, ImageDocument) or d.metadata.get("file_name", "").lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        for d in llama_docs
    )
    
    gemini_mm = None
    if has_image:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY not found!")
        gemini_mm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)

    for doc in llama_docs:
        if isinstance(doc, ImageDocument) or getattr(doc, "image_path", None) is not None or doc.metadata.get("file_name", "").lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            prompt = (
                "Bạn là một chuyên gia phân tích dữ liệu và sơ đồ. "
                "Hãy phân tích và mô tả cực kỳ chi tiết hình ảnh, sơ đồ hoặc bảng biểu này. "
                "Đừng bỏ sót bất kỳ văn bản, số liệu, cột, dòng hay mối quan hệ nào trong hình. "
                "Cung cấp bản mô tả rõ ràng để một hệ thống tìm kiếm (RAG) có thể hiểu hoàn toàn nội dung của hình ảnh này."
            )
            try:
                response = gemini_mm.complete(prompt=prompt, image_documents=[doc])
                content = f"Hình ảnh/Sơ đồ ({doc.metadata.get('file_name', '')}):\nMô tả chi tiết:\n{response.text}"
                documents.append(LangchainDocument(page_content=content, metadata={"source": file_path}))
            except Exception as e:
                pass
        else:
            documents.append(LangchainDocument(page_content=doc.text, metadata={"source": file_path}))

    if not documents:
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY not found!")
        embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=persist_directory
        )
    except Exception as e:
        raise e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_your_text_file>")
        print("Example: python ingest.py my_own_dataset.txt")
    else:
        file_path = sys.argv[1]
        ingest_data(file_path)
