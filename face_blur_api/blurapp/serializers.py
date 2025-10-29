
from rest_framework import serializers
from .models import ImageProcess
class ImageProcessSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageProcess
        fields = '__all__'
