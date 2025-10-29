
# Face Blur Detection & Correction API (Django DRF)

## Description
Django REST Framework project that accepts images, detects faces (OpenCV), checks blur using Laplacian variance,
applies a simple unsharp mask to "deblur" when blur is detected, and returns JSON metadata with processed image URL.

## Quickstart (local)
1. Create and activate a Python virtualenv (Python =  3.10.2 + recommended and django = 4.0.4)
 ```powershell / and can use bash
    python -m venv venv
    venv\Script\activate
   ```
2. Install dependencies:
   ```powershell / and can use bash
   cd .\face_blur_api\
   pip install -r requirements.txt
   ```
3. Run migrations:
   ```powershell / and can use bash
   python manage.py makemigrations
   python manage.py migrate
   ```
4. Create superuser (optional, for admin):
   ```powershell / and can use bash
   python manage.py createsuperuser
   ```
5. Run server:
   ```powershell / and can use bash
   python manage.py runserver
   ```
6. Visit http://127.0.0.1:8000/ to open the sample UI and test images.

## Endpoints
- POST `/api/upload/` (multipart/form-data) -> returns { id }
- POST `/api/analyze/` (json body: { "id": <id> }) -> returns analysis + processed path
- GET `/api/report/<id>/` -> get stored result object

## Notes
- Uses OpenCV Haar cascades for face detection. This is free and fast.
- Blur detection uses the Laplacian variance method (threshold adjustable in blur_analysis.py).
- For stronger deblurring, replace `unsharp_mask` logic with an open-source model like DeblurGANv2 or Real-ESRGAN.
