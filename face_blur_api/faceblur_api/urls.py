
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('api/', include('blurapp.urls')),
    path('', include('blurapp.front_urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
