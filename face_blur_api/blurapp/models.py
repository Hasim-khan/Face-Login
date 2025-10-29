
from django.db import models
class ImageProcess(models.Model):
    original = models.ImageField(upload_to='uploads/')
    processed = models.ImageField(upload_to='processed/', null=True, blank=True)
    num_faces = models.IntegerField(default=0)
    blur_level = models.FloatField(default=0.0)
    is_blurry = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"ImageProcess({self.id})"
