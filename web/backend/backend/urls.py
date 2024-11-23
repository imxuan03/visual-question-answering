from django.conf.urls.static import static
from django.conf import settings
from django.contrib import admin
from django.urls import path
from .views import ImageDetectAPI

urlpatterns = [
    path('api/detect/', ImageDetectAPI.as_view(), name='image-detect-api'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)