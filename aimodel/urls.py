from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("detect_fasterrcnn", views.detect_fasterrcnn, name="detect_fasterrcnn"),
    path("detect_yolo", views.detect_yolo, name="detect_yolo"),
    path("hold_on", views.hold_on, name="hold_on"),
    path("test", views.test, name="test"),
    path("detect_yolo_lower_fps", views.detect_yolo_lower_fps, name="detect_yolo_lower_fps"),
    path("detect_fasterrcnn_lower_fps", views.detect_fasterrcnn_lower_fps, name="detect_fasterrcnn_lower_fps"),
    path("detect_image", views.detect_image, name="detect_image"),
    path("detect_ssd", views.detect_ssd, name="detect_ssd")
]