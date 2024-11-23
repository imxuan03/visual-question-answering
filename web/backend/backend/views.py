import requests
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Img_predictions
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from django.conf import settings

model_path = "E:/Niên Luận/VQA/NienLuan/web/backend/backend/models/leaf-model.h5"

model = load_model(model_path)

categories = ['sake', 'bang', 'dua-gang', 'oi', 'mit']

class ImageDetectAPI(APIView):

    def post(self, request, *args, **kwargs):
        # Lấy ảnh từ request
        image_input = request.FILES.get('image_input')

        if not image_input:
            return Response({"error": "No image provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = self.prepare_image(image_input)
            # Dự đoán loại bệnh từ ảnh
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction, axis=1)
            label = categories[predicted_class[0]]
            return Response({'output': label})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def prepare_image(self, image_file):
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (64, 64))
        image = np.array(image, dtype="float") / 255.0
        image = np.expand_dims(image, axis=0)
        return image

