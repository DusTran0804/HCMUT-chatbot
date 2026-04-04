import os
import sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from Source.chatbot import init_chatbot
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Đang khởi tạo Hệ thống AI Chatbot (Gemini Cloud)...")
rag_chain = None
try:
 
    rag_chain = init_chatbot()
    print("\nKhởi tạo Chatbot thành công!")
except Exception as e:
    print(f"\nLỗi khi khởi tạo Chatbot: {e}")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="Mô hình AI chưa được khởi tạo thành công.")
    
    user_input = request.message
    if not user_input.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập câu hỏi.")
        
    try:
        result = rag_chain.invoke({"input": user_input})
        answer = result.get("answer", "")
        return {"answer": answer}
    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        raise HTTPException(status_code=500, detail=str(e))

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\nKhởi động Server thành công trên cổng {port}!")
    uvicorn.run("webapp:app", host="0.0.0.0", port=port, reload=False)
