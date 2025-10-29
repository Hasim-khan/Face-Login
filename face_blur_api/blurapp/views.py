from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from django.shortcuts import get_object_or_404, render
from .models import ImageProcess
from .serializers import ImageProcessSerializer
from .utils.face_detection import detect_faces_from_array
from .utils.blur_analysis import analyze_blur_from_array
from .utils.deblur import deblur_image  # Make sure this improves clarity
import os, cv2
# from .utils.deblur import remove_blur

class ImageUploadAPIView(APIView):
    def post(self, request):
        image = request.FILES.get('image')
        if not image:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        obj = ImageProcess.objects.create(original=image)
        return Response({'id': obj.id, 'message': 'Image uploaded successfully.'}, status=status.HTTP_201_CREATED)

class AnalyzeImageAPIView(APIView):
    def post(self, request):
        image_id = request.data.get('id')
        if not image_id:
            return Response({'error': 'Provide id in body'}, status=400)

        obj = get_object_or_404(ImageProcess, id=image_id)
        image_path = obj.original.path
        img = cv2.imread(image_path)

        # Step 1: Detect faces
        faces = detect_faces_from_array(img)

        # Step 2: Analyze before
        is_blur, variance_before = analyze_blur_from_array(img)

        # Step 3: Deblur if needed
        out_img = deblur_image(img)
        new_is_blur, variance_after = analyze_blur_from_array(out_img)

        # Save processed
        processed_fname = os.path.basename(image_path)
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        full_out = os.path.join(processed_dir, processed_fname)
        cv2.imwrite(full_out, out_img)
        obj.processed.name = f'processed/{processed_fname}'

        obj.num_faces = len(faces)
        obj.blur_level = variance_after
        obj.is_blurry = new_is_blur
        obj.save()

        serializer = ImageProcessSerializer(obj)
        resp = {
            "faces_detected": len(faces),
            "was_blurry": is_blur,
            "blur_level_before": variance_before,
            "blur_level_after": variance_after,
            "data": serializer.data,
        }
        return Response(resp, status=200)


class ReportAPIView(APIView):
    def get(self, request, pk):
        obj = get_object_or_404(ImageProcess, pk=pk)
        serializer = ImageProcessSerializer(obj)
        return Response(serializer.data)


def index(request):
    return render(request, 'index.html')
