import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, BertTokenizer
from PIL import Image
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

# Load các model đã được huấn luyện trước
class VQAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, combined_features):
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model weights
model_path = "E:/Niên Luận/VQA/NienLuan/web/backend/backend/models/vqa_model.pth"  # Đường dẫn tới model đã được lưu
model_vqa = VQAModel(input_dim=1536, hidden_dim=512, output_dim=24)  # Điều chỉnh output_dim phù hợp với số lớp của bạn
model_vqa.load_state_dict(torch.load(model_path, map_location=device))
model_vqa.to(device)
model_vqa.eval()

# Load tokenizer và feature extractor
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Chuyển đổi hình ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ImageDetectAPI(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        try:
            # Nhận dữ liệu từ request
            image_file = request.FILES.get("image")
            question_text = request.data.get("question")

            if not image_file or not question_text:
                return Response({"error": "Cần gửi đầy đủ ảnh và câu hỏi."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Xử lý ảnh
            image = Image.open(image_file).convert("RGB")
            image_inputs = feature_extractor(images=image, return_tensors="pt").to(device)

            # Xử lý câu hỏi
            question_inputs = tokenizer(
                question_text,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(device)

            # Trích xuất đặc trưng
            with torch.no_grad():
                image_features = model_vqa.fc1(model_vqa.relu(image_inputs["pixel_values"]))
                question_features = model_vqa.fc1(model_vqa.relu(question_inputs["input_ids"]))

                combined_features = torch.cat((image_features, question_features), dim=1)

            # Dự đoán kết quả
            output = model_vqa(combined_features)
            predicted_class = torch.argmax(output, dim=1).item()

            return Response({"answer": f"Class {predicted_class}"}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
