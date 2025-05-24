from django.db import models
from django.core.files.storage import FileSystemStorage


fs = FileSystemStorage(location="/DB_files")

class Video(models.Model):
    video = models.FileField(upload_to="DB_files")
    num_in_crowd = models.IntegerField(default=0)


class Processed_video(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    processed_video = models.FileField(upload_to="DB_files")
    list_nums_persons = models.JSONField()
