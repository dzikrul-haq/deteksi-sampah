from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators import gzip
from .utils.video_camera import VideoCamera
from .utils.yolo_detection import ModelYOLO
from .models import Detection



@gzip.gzip_page
def index(request):
    return render(request, "streamapp/pages/camera.html")


def detections(request):
    detections = Detection.objects.all()
    context = {"data": detections}
    return render(request, "streamapp/pages/detections.html", context)


def get_camera(response):
    cam = VideoCamera(3)
    try:
        return StreamingHttpResponse(
            gen(cam), content_type="multipart/x-mixed-replace;boundary=frame"
        )
    except FileNotFoundError:
        cam.__del__()
        pass


def gen(camera):
    try:
        model = ModelYOLO()

        while True:
            frame = camera.get_frame()
            det = model.detect(original_image=frame)

            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + det + b"\r\n\r\n"
            )

    except TypeError:
        camera.__del__()
        pass
