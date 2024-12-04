import torch
from transformers import ViTImageProcessor, BertTokenizer, BertModel, ViTModel
from PIL import Image
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Đường dẫn tới file CSV
csv_path = 'E:/Niên Luận/VQA/NienLuan/train_vqa_dataset.csv'  # Thay bằng đường dẫn file của bạn
data = pd.read_csv(csv_path)

# Tính toán output_dim dựa trên số câu trả lời duy nhất
unique_answers = data['answer'].unique()
output_dim = len(unique_answers)  # Số lượng lớp

# Log để kiểm tra
print(f"Number of unique answers (output_dim): {output_dim}")

# Label Encoder để ánh xạ câu trả lời
label_encoder = LabelEncoder()
label_encoder.fit(data['answer'])  # Mã hóa các câu trả lời

# Load các mô hình
feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model_vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

# Định nghĩa mô hình VQA
class VQAModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, combined_features):
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Khởi tạo model và load weights
input_dim = 768 + 768  # 768 cho ViT và 768 cho BERT
hidden_dim = 512
model_path = 'E:/Niên Luận/VQA/NienLuan/web/backend/backend/models/vqa_model_final12.pth'  # Thay bằng đường dẫn model đã lưu
model = VQAModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Định nghĩa API view
class VqaAPI(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        try:
            # Nhận ảnh và câu hỏi từ yêu cầu
            image_file = request.FILES['image']
            question_text = request.data['question']

            # Load ảnh và trích xuất đặc trưng bằng ViT
            image = Image.open(image_file).convert('RGB')
            image_inputs = feature_extractor(images=image, return_tensors="pt")
            image_features = model_vit(**image_inputs).last_hidden_state[:, 0, :]  # First token's features

            # Tokenize câu hỏi và trích xuất đặc trưng bằng BERT
            question_inputs = tokenizer(question_text, return_tensors="pt", truncation=True, padding=True)
            question_features = model_bert(**question_inputs).last_hidden_state[:, 0, :]  # First token's features

            # Kết hợp đặc trưng ảnh và câu hỏi
            combined_features = torch.cat((image_features, question_features), dim=1).squeeze()

            # Dự đoán câu trả lời
            output = model(combined_features.unsqueeze(0))  # Add batch dimension
            _, predicted = torch.max(output, 1)
            predicted_index = predicted.item()

            # Decode câu trả lời
            answer = label_encoder.inverse_transform([predicted_index])[0]

            return JsonResponse({"question": question_text, "answer": answer}, status=status.HTTP_200_OK)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
