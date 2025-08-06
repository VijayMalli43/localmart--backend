# LocalMartAI_Project/LocalMartAI/urls.py

from django.contrib import admin
from django.urls import path, include
from django.conf import settings # Needed for MEDIA_URL
from django.conf.urls.static import static # Needed for serving media files in development

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')), # Includes all API endpoints from your 'core' app
]

# Serve media files only during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)