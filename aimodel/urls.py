from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("detect_fasterrcnn", views.detect_fasterrcnn, name="detect_fasterrcnn"),
    path("detect_yolo", views.detect_yolo, name="detect_yolo"),
    path("hold_on", views.hold_on, name="hold_on")
]