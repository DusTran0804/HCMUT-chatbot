import os
import sys
import shutil
import gc
from dotenv import load_dotenv

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LangchainDocument

from llama_index.core import SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.schema import ImageDocument
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.base.llms.types import ImageBlock, TextBlock

current_dir = os.path.dirname(os.path.abspath(__file__))
persist_directory = os.path.join(os.path.dirname(current_dir), "chroma_db")

def ingest_data(file_path):
    if not os.path.exists(file_path):
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment!")

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    try:
        if os.path.isdir(file_path):
            llama_docs = SimpleDirectoryReader(input_dir=file_path, recursive=True, exclude_hidden=False).load_data()
        else:
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
        gemini_mm = GoogleGenAI(
            model="gemini-2.0-flash", 
            api_key=api_key,
            temperature=0.0, 
            max_tokens=8192  
        )

    for doc in llama_docs:
        # Check if the document is an image
        if isinstance(doc, ImageDocument) or getattr(doc, "image_path", None) is not None or doc.metadata.get("file_name", "").lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            prompt = (
                "Bạn là một chuyên gia phân tích tài liệu. Hãy trả lời câu hỏi dựa trên các đoạn văn bản được cung cấp." 
                "Bạn là một cỗ máy trích xuất dữ liệu quang học (OCR). " 
                "NHIỆM VỤ CỦA BẠN: Trích xuất chính xác 100% mọi văn bản và số liệu trong hình ảnh. " 
                "NẾU HÌNH ẢNH LÀ DẠNG BẢNG: Bạn phải kiên nhẫn đọc TỪNG DÒNG MỘT từ trên xuống dưới và liệt kê TẤT CẢ các dòng.\n" 
                "Với mỗi dòng trong bảng, hãy ghép với tiêu đề cột và tiêu đề bảng thành một câu văn hoàn chỉnh, KHÔNG dùng bảng Markdown.\n" 
                "Hãy làm tương tự cho TOÀN BỘ các dòng trong bảng. Đừng bỏ sót một dòng nào!"
            )
            try:
                img_path = getattr(doc, "image_path", None)
                if not img_path:
                    img_path = doc.metadata.get("file_path", None)
                
                if img_path and os.path.exists(img_path):
                    msg = ChatMessage(
                        role=MessageRole.USER,
                        blocks=[
                            TextBlock(text=prompt),
                            ImageBlock(path=img_path)
                        ]
                    )
                    response = gemini_mm.chat([msg])
                    content = f"Hình ảnh/Sơ đồ ({doc.metadata.get('file_name', '')}):\nMô tả chi tiết:\n{response.message.content}"
                    documents.append(LangchainDocument(page_content=content, metadata={"source": doc.metadata.get("file_name", file_path)}))
                else:
                    documents.append(LangchainDocument(page_content=doc.text, metadata={"source": doc.metadata.get("file_name", file_path)}))
            except Exception as e:
                print(f"Error processing image {doc.metadata.get('file_name', '')}: {e}")
        else:
            documents.append(LangchainDocument(page_content=doc.text, metadata={"source": doc.metadata.get("file_name", file_path)}))

    if not documents:
        return
 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    try:
        embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
  
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings_model
        )

        del llama_docs
        del documents
        gc.collect()
        
        batch_size = 100 
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            db.add_documents(batch)
            
    except Exception as e:
        raise e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python ingest.py <đường_dẫn_file_hoặc_thư_mục>")
    else:
        file_path = sys.argv[1]
        ingest_data(file_path)
