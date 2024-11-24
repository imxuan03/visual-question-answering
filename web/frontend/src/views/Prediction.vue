<template>
    <div class="chat-container">
        <!-- Hiển thị tin nhắn -->
        <div class="messages">
            <div v-for="(message, index) in chatMessages" :key="index" :class="['message', message.sender]">
                <span v-if="message.text">{{ message.text }}</span>
                <img v-if="message.image" :src="message.image" alt="Image" class="message-image" />
            </div>
        </div>

        <!-- Ô nhập và nút gửi -->
        <div class="input-box">
            <div class="preview" v-if="imagePreview">
                <img :src="imagePreview" alt="Ảnh đã chọn" class="preview-image" />
                <button class="remove-image-btn" @click="removeSelectedImage">✖</button>
            </div>
            <textarea v-model="userMessage" placeholder="Nhập câu hỏi..." @keyup.enter="sendMessage"
                class="input-textarea"></textarea>
            <input type="file" id="fileInput" accept="image/*" @change="onImageSelected" />
            <label for="fileInput" class="file-label">Thêm ảnh</label>
            <button @click="sendMessage">Gửi</button>
        </div>
    </div>
</template>

<script>
import axios from "axios"; // Import thư viện axios để gửi request
import PredictService from "../services/predict.service";

export default {
    name: "ChatBox",
    data() {
        return {
            userMessage: "", // Tin nhắn người dùng nhập
            chatMessages: [
                { text: "Hello! How can I help you?", sender: "bot" }
            ], // Lịch sử tin nhắn
            selectedImage: null, // File ảnh người dùng chọn
            imagePreview: null, // URL hiển thị ảnh tạm thời
        };
    },
    methods: {
        async sendMessage() {
            if (this.userMessage.trim() === "" && !this.selectedImage) {
                return; // Không có text hoặc ảnh => không gửi
            }

            // Thêm text vào danh sách tin nhắn
            if (this.userMessage.trim() !== "") {
                this.chatMessages.push({ text: this.userMessage, sender: "user" });
            }

            // Thêm ảnh vào danh sách tin nhắn
            if (this.selectedImage) {
                this.chatMessages.push({ image: this.imagePreview, sender: "user" });
            }

            const formData = new FormData();

            // Thêm text và ảnh vào formData để gửi lên backend
            if (this.userMessage.trim() !== "") {
                formData.append("question", this.userMessage);
            }
            if (this.selectedImage) {
                formData.append("image", this.selectedImage);
            }
            console.log(this.userMessage)
            console.log(this.selectedImage)
            console.log(formData)
            // Reset input text và ảnh sau khi gửi
            this.userMessage = "";
            this.selectedImage = null;
            this.imagePreview = null;

            try {
                
                const response = await PredictService.uploadImage(formData);
                console.log("API Response:", response);

                // Thêm câu trả lời của bot vào danh sách
                if (response.data && response.data.answer) {
                    this.chatMessages.push({ text: response.data.answer, sender: "bot" });
                } else {
                    this.chatMessages.push({ text: "Sorry, no answer was received!", sender: "bot" });
                }
            } catch (error) {
                console.error("Lỗi khi gửi yêu cầu:", error);
                this.chatMessages.push({ text: "Sorry, an error occurred!", sender: "bot" });
            }
        },
        onImageSelected(event) {
            // Khi chọn ảnh, tạo URL tạm thời để hiển thị
            this.selectedImage = event.target.files[0];
            this.imagePreview = URL.createObjectURL(this.selectedImage);
        },
        removeSelectedImage() {
            // Xóa ảnh đã chọn
            this.selectedImage = null;
            this.imagePreview = null;
            const fileInput = document.getElementById("fileInput");
            if (fileInput) {
                fileInput.value = null; // Reset giá trị của input file
            }
        },
    },
};
</script>


<style scoped>
/* Tổng thể khung chat */
.chat-container {
    width: 100%;
    max-width: 500px;
    margin: 20px auto;
    border-radius: 10px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    height: 600px;
    background-color: #f9fafb;
    border: 1px solid rgb(190, 184, 184);
}


/* Khu vực hiển thị tin nhắn */
.messages {
    flex: 1;
    padding: 15px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
    /* Khoảng cách giữa các tin nhắn */
}

/* Tin nhắn của bot (trái) */
.message.bot {
    align-self: flex-start;
    background-color: #3a3a55;
    /* Nền xám đậm */
    color: #fff;
    /* Chữ trắng */
    padding: 10px 15px;
    border-radius: 15px 15px 15px 0;
    /* Bo góc cho tin nhắn bot */
    max-width: 70%;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
}

/* Tin nhắn của người dùng (phải) */
.message.user {
    align-self: flex-end;
    background-color: #d4d4d4;
    /* Nền cam */
    color: black;
    /* Chữ trắng */
    padding: 10px 15px;
    border-radius: 15px 15px 0 15px;
    /* Bo góc cho tin nhắn người dùng */
    max-width: 70%;
    font-size: 14px;
    line-height: 1.5;
    word-wrap: break-word;
}

/* Tin nhắn nhiều dòng sẽ tự động xuống hàng */
.message {
    display: inline-block;
    word-break: break-word;
}

/* Hình ảnh trong tin nhắn */
.message-image {
    max-width: 100%;
    border-radius: 10px;
    margin-top: 5px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

/* Khu vực nhập */
.input-box {
    display: flex;
    align-items: center;
    padding: 10px;
    border-top: 1px solid #ddd;
    background-color: #f4f7f6;
    gap: 10px;
    /* Khoảng cách giữa các thành phần */
}

/* Ô nhập tin nhắn */
input[type="text"] {
    flex: 1;
    padding: 12px 15px;
    border: 1px solid #ddd;
    border-radius: 20px;
    font-size: 14px;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
}

input[type="text"]:focus {
    outline: none;
    border-color: #4caf50;
    box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
}

/* Ô nhập tin nhắn (textarea) */
.input-textarea {
    flex: 1;
    padding: 12px 15px;
    border: 1px solid #ddd;
    border-radius: 20px;
    font-size: 14px;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
    resize: none;
    /* Tắt nút kéo giãn */
    overflow: hidden;
    /* Ẩn thanh cuộn ngang */
    line-height: 1.5;
    min-height: 40px;
    /* Chiều cao tối thiểu */
    max-height: 100px;
    /* Chiều cao tối đa */
    background-color: #fff;
    /* Nền trắng */
}

.input-textarea:focus {
    outline: none;
    border-color: #4caf50;
    box-shadow: 0 0 5px rgba(76, 175, 80, 0.5);
}

/* Nút thêm ảnh */
input[type="file"] {
    display: none;
}

.file-label {
    padding: 10px 15px;
    font-size: 14px;
    font-weight: bold;
    color: #fff;
    background-color: #2196f3;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.file-label:hover {
    background-color: #1976d2;
}

/* Nút gửi */
button {
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
    color: #fff;
    background-color: #4caf50;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

button:hover {
    background-color: #45a049;
}

button:active {
    transform: scale(0.98);
}

/* Hiển thị ảnh đã chọn */
.preview {
    display: flex;
    align-items: center;
    position: relative;
    gap: 5px;
}

.preview-image {
    max-width: 80px;
    max-height: auto;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.remove-image-btn {
    background: #ff6b6b;
    color: #fff;
    font-size: 14px;
    border: none;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    position: absolute;
    top: 0;
    right: 0;
    transform: translate(50%, -50%);
}

.remove-image-btn:hover {
    background: #ff3b3b;
}
</style>