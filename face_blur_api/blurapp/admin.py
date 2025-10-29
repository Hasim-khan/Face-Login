
from django.contrib import admin
from .models import ImageProcess
@admin.register(ImageProcess)
class ImageProcessAdmin(admin.ModelAdmin):
    list_display = ('id','num_faces','is_blurry','blur_level','created_at')
