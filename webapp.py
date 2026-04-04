import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from Source.chatbot import init_chatbot
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Global RAG chain - Khởi tạo giá trị None ban đầu
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handler quản lý vòng đời của ứng dụng.
    Khởi tạo chatbot AI khi ứng dụng bắt đầu và dọn dẹp khi kết thúc.
    """
    global rag_chain
    print("🚀 Đang khởi khởi tạo Hệ thống AI Chatbot (Gemini Cloud mode)... Vui lòng chờ...")
    try:
        # Thực hiện việc nạp RAG chain (Khá nặng, mất khoảng 30-60s trên Render Free)
        rag_chain = init_chatbot()
        print("✅ Khởi tạo Chatbot thành công! Hệ thống đã sẵn sàng phục vụ.")
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi khởi tạo Chatbot: {e}")
    
    yield
    # Dọn dẹp tài nguyên nếu cần khi server tắt
    print("Shutting down...")

# Khởi tạo FastAPI với lifespan handler mới
app = FastAPI(lifespan=lifespan)

# Cấu hình CORS để cho phép GitHub Pages truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint xử lý hội thoại RAG.
    """
    if not rag_chain:
        # Nếu bộ não AI chưa nạp được, trả về lỗi 503 (Service Unavailable)
        raise HTTPException(
            status_code=503, 
            detail="Hệ thống AI đang khởi động hoặc chưa sẵn sàng. Vui lòng thử lại sau 30 giây."
        )
    
    user_input = request.message
    if not user_input.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")
        
    try:
        # Gọi Gemini AI kèm theo ngữ cảnh từ Vector DB
        result = rag_chain.invoke({"input": user_input})
        answer = result.get("answer", "Xin lỗi, tôi không tìm được câu trả lời phù hợp trong dữ liệu huấn luyện.")
        return {"answer": answer}
    except Exception as e:
        print(f"Lỗi khi xử lý hội thoại: {e}")
        # Trả về lỗi 500 kèm mô tả ngắn gọn
        raise HTTPException(status_code=500, detail="Lỗi trong quá trình AI xử lý câu trả lời.")

# Phục vụ file tĩnh (CSS/JS) nếu chạy trên localhost
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    # Render cung cấp cổng qua biến môi trường PORT
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Started on port {port}")
    uvicorn.run("webapp:app", host="0.0.0.0", port=port, reload=False)
