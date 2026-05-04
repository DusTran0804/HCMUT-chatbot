# Multimodal RAG Chatbot với Google Gemini & LlamaIndex

Đây là một dự án Chatbot hỏi đáp dựa trên tài liệu (RAG - Retrieval-Augmented Generation) tiên tiến, có khả năng xử lý **đa phương thức (Multimodal)** bao gồm văn bản và hình ảnh/sơ đồ/bảng biểu. Dự án đã được chuyển đổi từ mô hình cục bộ sang sử dụng sức mạnh của **Google Gemini API**, kết hợp với các framework **Langchain**, **LlamaIndex** và cơ sở dữ liệu vector **ChromaDB**.

Dự án cung cấp cả giao diện dòng lệnh (CLI) và một API web server sử dụng **FastAPI**, sẵn sàng để triển khai trên các nền tảng đám mây như Render.

## Tính năng nổi bật
- **Multimodal RAG (Xử lý hình ảnh & tài liệu)**: Khả năng đọc và trích xuất thông tin cực kỳ chính xác từ hình ảnh, sơ đồ và bảng biểu bằng model `gemini-2.5-flash` thông qua LlamaIndex (`ChatMessage` và `ImageBlock`).
- **Hỏi đáp thông minh**: Sử dụng model `gemini-3.1-flash-lite-preview` mạnh mẽ để tổng hợp ngữ cảnh và trả lời các câu hỏi phức tạp dựa trên dữ liệu đã nạp.
- **Vector Database**: Sử dụng `ChromaDB` cục bộ và `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`) để tối ưu hóa việc phân mảnh (chunking) và tìm kiếm ngữ cảnh.
- **Web API & Web App**: Tích hợp FastAPI để chạy dưới dạng API Server, phục vụ ứng dụng web tĩnh và cung cấp API RESTful cho frontend.

---

## Yêu cầu hệ thống (Prerequisites)

Trước khi chạy chương trình, bạn cần đảm bảo máy tính đã cài đặt:
1. **Python 3.8+**
2. **Gemini API Key**: Đăng ký và lấy API Key miễn phí tại [Google AI Studio](https://aistudio.google.com/).

*(Lưu ý: Bạn không cần cài đặt Ollama hay Llama 3 cục bộ nữa, mọi xử lý mô hình ngôn ngữ đều chạy qua Google Cloud).*

---

## Hướng dẫn cài đặt và thiết lập (Setup)

### Bước 1: Tải mã nguồn
Mở Terminal và chạy lệnh sau để tải toàn bộ mã nguồn về máy:

```bash
git clone <ĐƯỜNG_DẪN_GITHUB_CỦA_BẠN>
cd ChatBotHCMUT
```

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
Cài đặt tất cả các dependencies từ file `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 5: Cấu hình API Key
Tạo một file `.env` ở thư mục gốc của dự án (cùng cấp với `README.md`) và thêm Gemini API Key của bạn vào:
```env
GEMINI_API_KEY=your_api_key_here
```

---

## Cách chạy chương trình

Dự án này hoạt động theo 2 bước chính:
1. **Nạp dữ liệu (Ingest)**: Đọc thư mục/tệp (văn bản, hình ảnh, tài liệu) và nạp vào cơ sở dữ liệu vector.
2. **Khởi động Bot (Chat/Web App)**: Chạy ứng dụng để bắt đầu trò chuyện dựa trên kho tri thức.

### 1. Nạp dữ liệu vào Chatbot (Ingest)

Sử dụng script `Source/ingest.py` để phân tích và xử lý tài liệu. LlamaIndex sẽ tự động nhận diện và OCR hình ảnh nếu có. Bạn có thể truyền vào đường dẫn đến một file cụ thể hoặc một thư mục.

```bash
python Source/ingest.py <đường_dẫn_file_hoặc_thư_mục>
```
*Ví dụ:* `python Source/ingest.py Input_data/`
*(Lưu ý: Quá trình này sẽ gửi request đến Gemini để trích xuất văn bản/bảng biểu từ hình ảnh. Sau khi hoàn tất, CSDL vector sẽ được lưu tại thư mục `chroma_db`).*

### 2. Sử dụng Chatbot trên Terminal (CLI)

Sau khi đã nạp dữ liệu thành công, chạy lệnh sau để trò chuyện trực tiếp với bot trên cửa sổ Terminal:

```bash
python Source/chatbot.py
```
*(Gõ câu hỏi của bạn và nhấn Enter. Để thoát ứng dụng, hãy gõ `exit` hoặc `quit`).*

### 3. Khởi động Web API Server (FastAPI)

Nếu bạn muốn chạy hệ thống dưới dạng web server để tương tác qua giao diện web, sử dụng lệnh sau:

```bash
python webapp.py
```
Server sẽ chạy trên `http://0.0.0.0:8000` (hoặc cổng được cấu hình).
- Web App sẽ tự động nạp RAG Chain vào bộ nhớ khi khởi động.
- Trang web tĩnh (nếu có trong mục `/static`) sẽ được phục vụ tại endpoint gốc `/`.
- API Endpoint để gọi chat là `POST /api/chat` với dạng JSON: `{ "message": "Câu hỏi của bạn" }`.

---

## Triển khai (Deployment)

Dự án đã được cấu hình sẵn file `render.yaml` và `webapp.py` để hỗ trợ triển khai liền mạch lên nền tảng đám mây [Render](https://render.com/).
1. Commit và đẩy toàn bộ mã nguồn lên kho lưu trữ GitHub của bạn.
2. Tại bảng điều khiển Render, tạo mới dựa trên Blueprint (chọn kết nối repo này). Render sẽ tự động đọc `render.yaml` để thiết lập Web Service.
3. **Quan trọng**: Đảm bảo bạn đã thêm biến môi trường `GEMINI_API_KEY` vào mục Environment Variables trên Render Dashboard.

---

