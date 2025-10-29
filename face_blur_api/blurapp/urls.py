
from django.urls import path
from .views import ImageUploadAPIView, AnalyzeImageAPIView, ReportAPIView
from . import views as v
from django.urls import path

urlpatterns = [
    path('upload/', ImageUploadAPIView.as_view(), name='upload'),
    path('analyze/', AnalyzeImageAPIView.as_view(), name='analyze'),
    path('report/<int:pk>/', ReportAPIView.as_view(), name='report'),
]

# front urls included from project to render sample UI
front_urls = [
    path('', v.index, name='index'),
]
