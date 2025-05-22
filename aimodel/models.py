from django.db import models


class Video(models.Model):
    video = models.FileField()
    num_in_crowd = models.IntegerField(default=0)


class Screenshots(models.Model):
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    screenshot = models.FileField()
    num_in_screenshot = models.IntegerField(default=0)
