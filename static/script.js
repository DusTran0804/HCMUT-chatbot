document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const sendBtn = document.getElementById('send-btn');

    // Cấu hình API URL: Nếu chạy trên GitHub Pages, hãy trỏ về URL của Render.com
    // Bạn có thể thay đổi URL này sau khi deploy lên Render xong (nếu tên web service khác)
    const BASE_API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? '' 
        : 'https://hcmut-chatbot.onrender.com';

    // Cửa sổ Markdown renderer settings
    marked.setOptions({
        breaks: true,
        gfm: true
    });

    const scrollToBottom = () => {
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const addMessage = (content, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = sender === 'bot' ? marked.parse(content) : content;
        msgDiv.appendChild(contentDiv);
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    const showLoading = () => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message bot-message`;
        msgDiv.id = 'loading-message';
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <div class="typing-indicator">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
        `;
        msgDiv.appendChild(contentDiv);
        chatBox.appendChild(msgDiv);
        scrollToBottom();
    };

    const removeLoading = () => {
        const loadingMsg = document.getElementById('loading-message');
        if (loadingMsg) loadingMsg.remove();
    };

    const streamBotText = (fullText) => {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message bot-message`;
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        msgDiv.appendChild(contentDiv);
        chatBox.appendChild(msgDiv);

        return new Promise((resolve) => {
            let i = 0;
            const typingSpeed = 10;
            function typeWriter() {
                if (i <= fullText.length) {
                    contentDiv.innerHTML = marked.parse(fullText.substring(0, i));
                    i++;
                    scrollToBottom();
                    setTimeout(typeWriter, typingSpeed);
                } else {
                    resolve();
                }
            }
            typeWriter();
        });
    };

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;
        
        userInput.value = '';
        userInput.disabled = true;
        sendBtn.disabled = true;
        
        addMessage(text, 'user');
        showLoading();
        
        try {
            // BASE_API_URL sẽ tự động chọn localhost hoặc Render depend on the domain
            const response = await fetch(`${BASE_API_URL}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            
            removeLoading();
            if (response.ok) {
                const data = await response.json();
                await streamBotText(data.answer);
            } else {
                addMessage('Lỗi xử lý từ hệ thống.', 'bot');
            }
        } catch (error) {
            removeLoading();
            addMessage('Lỗi kết nối server.', 'bot');
        } finally {
            userInput.disabled = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    userInput.focus();
});
