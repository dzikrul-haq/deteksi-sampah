from uuid import uuid4
from django.db import models

# Create your models here.


class TimeStampedModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid4, editable=False)
    date_created = models.DateTimeField(auto_now_add=True)
    date_updated = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Detection(TimeStampedModel):
    image = models.ImageField(upload_to="output/")

    class Meta:
        verbose_name = "Detection"
        verbose_name_plural = "Detections"
