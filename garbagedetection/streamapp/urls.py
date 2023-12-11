from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('detect', views.get_camera, name='detect'),
    path('detections', views.detections, name='detections') 
] 

# /stream
# /stream/detect
# /stream/detection