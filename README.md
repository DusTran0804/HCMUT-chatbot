# RAG Chatbot với Ollama (Llama 3) & Langchain

Đây là một dự án Chatbot hỏi đáp dựa trên tài liệu (RAG - Retrieval-Augmented Generation) sử dụng **Ollama (mô hình Llama 3)**, **Langchain**, và **ChromaDB**. 

Chatbot có khả năng đọc dữ liệu từ một tệp văn bản (`.txt`), lưu trữ vào cơ sở dữ liệu vector từ và trả lời các câu hỏi của bạn dựa trên nội dung đó.

## Yêu cầu hệ thống (Prerequisites)

Trước khi chạy chương trình, bạn cần đảm bảo máy tính đã cài đặt:
1. **Python 3.8+**
2. **Ollama**: Tải và cài đặt từ [ollama.com](https://ollama.com/)
3. **Mô hình Llama 3 trên Ollama**:
   Mở terminal / command prompt và chạy lệnh sau để tải mô hình:
   ```bash
   ollama run llama3
   ```
   *(Hãy kiểm tra để chắc chắn Ollama đang chạy ngầm trên máy của bạn).*

## Môi trường cài đặt

Dự án này sử dụng môi trường ảo (`venv`) để quản lý các thư viện.

---

## Hướng dẫn cài đặt và thiết lập (Setup)

### Bước 1: Tải mã nguồn từ GitHub (Clone the repo)

Mở Terminal và chạy lệnh sau để tải toàn bộ mã nguồn về máy:

```bash
git clone <ĐƯỜNG_DẪN_GITHUB_CỦA_BẠN>
cd chatbot
```
*(Lưu ý: Thay `<ĐƯỜNG_DẪN_GITHUB_CỦA_BẠN>` bằng URL repo GitHub của dự án).*

### Bước 2: Tạo môi trường ảo (Virtual Environment)
Khuyến nghị tạo một môi trường ảo có tên là `venv`:
```bash
python3 -m venv venv
```

### Bước 3: Kích hoạt môi trường ảo
- **Trên macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```
- **Trên Windows**:
  ```bash
  venv\Scripts\activate
  ```

### Bước 4: Cài đặt các thư viện cần thiết
Hiện tại mã nguồn sử dụng các thư viện như `langchain`, `langchain-community`, `chromadb`, và `sentence-transformers`. Hãy chạy lệnh sau (hoặc lệnh tương đương dựa trên tệp `requirements.txt` nếu có):
```bash
pip install langchain langchain-community langchain-core chromadb sentence-transformers huggingface-hub
```

---

## Cách chạy chương trình

Dự án này hoạt động theo 2 bước:
1. Đọc tệp dữ liệu văn bản và nạp vào cơ sở dữ liệu (Ingest).
2. Chạy bot để bắt đầu chat (Chat).

Chúng ta có 2 cách để chạy chương trình: Chạy thủ công hoặc sử dụng Bash script có sẵn (`run_bot.sh`).

### Cách 1: Chạy tự động thông qua Script (Dành cho macOS/Linux)

Chỉ cần chạy file script `run_bot.sh`. File này sẽ tự động kích hoạt môi trường ảo (venv), nạp dữ liệu từ `sample_data.txt` vào cơ sở dữ liệu và gọi bot lên.

```bash
chmod +x run_bot.sh
./run_bot.sh
```

### Cách 2: Chạy thủ công (Dành cho mọi hệ điều hành)

Nếu bạn không thể chạy bash script hoặc muốn sử dụng một tệp dữ liệu khác, bạn có thể chạy theo các lệnh sau (Đảm bảo đã kích hoạt thư mục venv):

**1. Nạp dữ liệu vào Chatbot:**
Chạy file `ingest.py` cùng với đường dẫn đến file văn bản chứa nội dung bạn muốn chatbot ghi nhớ:
```bash
python ingest.py sample_data.txt
```
*(Chương trình sẽ hiển thị "Chatbot Loading... Done!" khi dữ liệu đã được nạp thành công vào thư mục `chroma_db`).*

**2. Khởi động Chatbot:**
Sau khi đã nạp dữ liệu, chạy lệnh sau để trò chuyện với bot:
```bash
python chatbot.py
```
*(Chờ đến khi hiện dòng chữ `Chatbot is already! Please ask. Type 'exit' or 'quit' to out.`)*

Gõ câu hỏi của bạn và nhấn Enter. Để thoát ứng dụng, hãy gõ `exit` hoặc `quit`.

---

## Lưu ý

- **Dữ liệu Vector**: Lịch sử dữ liệu nạp vào được lưu ở thư mục `chroma_db`. Mỗi lần bạn chạy file `ingest.py`, file cũ sẽ bị xóa cài đè để bot chỉ tập trung vào dữ liệu mới nhất.
- **Tốc độ phản hồi**: Thời gian chờ câu trả lời phụ thuộc vào cấu hình máy tính (Ollama/Llama 3 được xử lý cục bộ trên thiết bị của bạn).
