from rest_framework import serializers
from .models import Img_predictions


class ImgPredictionsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Img_predictions
        fields = '__all__'