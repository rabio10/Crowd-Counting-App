import io
from django.shortcuts import render
from django.http import HttpResponse, FileResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import *
from wsgiref.util import FileWrapper
import json
from django.core.files import File
import ffmpeg
import pickle
import skimage


import cv2
import torchvision
import torch
import os
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
import time
from ultralytics import YOLO

video_num_frames = -1

FPS_COEFF = 15

def index(request):
    return HttpResponse("Hello, world. You're at the ai model index.")

@csrf_exempt
@require_http_methods(["POST"])
def detect_fasterrcnn(request):
    print(request.FILES)
    print(request.FILES)
    if request.method == 'POST' and request.FILES.get('file'):
        # save file in db 
        instance = Video(video=request.FILES['file'])
        instance.save()
        print("file saved in db")

        output_path, counts_over_frames = process_video(request)
        
        print("returning processed video")
         # save to db
        file_obj = File(open(output_path, 'rb'), name="out_yolo.mp4")
        processed_instance = Processed_video(video=instance ,processed_video=file_obj, list_nums_persons=json.dumps(counts_over_frames))
        processed_instance.save()
        # Return processed video
        file = FileWrapper(open(output_path, 'rb'))
        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'
        response['X-list-counts'] = json.dumps(counts_over_frames)
        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        # Add CORS headers
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST'
        response['Access-Control-Allow-Headers'] = '*'
    
        return response
    elif not request.FILES.get('file'):
        print(JsonResponse({'error': 'something wrong with the body of the request'}, status=400))
        return JsonResponse({'error': 'something wrong with the body of the request'}, status=400)
    return JsonResponse({'error': 'either the request is not POST or body doesnt have the correct key for accessing the video file'}, status=400)

def process_video(request):
    """
    return output_path
    """
    # Set device
    device = 'cpu'
    
    # Load model (you'll need to replace this with your actual model loading code)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.eval()

    model.to(device)
    model.eval()
    print(f"model loaded : device=",device)
    
    # Save uploaded file temporarily
    file = request.FILES['file']
    temp_path = "aimodel/temp_files/temp.mp4"
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)
    
    print("starting to process video")
    # Process video
    cap = cv2.VideoCapture(temp_path)
    output_path = 'aimodel/temp_files/out_fasterrcnn.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                            (int(cap.get(3)), int(cap.get(4))))
    global video_num_frames
    video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_num_frames)
    
    
    frames_counter = 0
    time1 = time.time()
    counts_over_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        print(f"passing frame [{frames_counter}] to model")    
        # Preprocess frame and run inference
        inputs = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)[0]
        
        # Draw boxes for each detection above threshold
        boxes_count = 0
        for box in outputs['boxes'].detach().cpu().numpy():
            boxes_count += 1
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"number person : {boxes_count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
        out.write(frame)
        counts_over_frames.append(boxes_count)
        #print("frame writen")
        frames_counter +=1
    time2 = time.time()
    print(f"video processed in {time2-time1}") # 4-7 sec for each frame approx.
    
    cap.release()
    out.release()
    
    print("cleaning up temp files")
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path, counts_over_frames

@csrf_exempt
@require_http_methods(["POST"])
def detect_yolo(request):
    print(request.FILES.get('file'))
    print(request.FILES['file'])
    if request.method == 'POST' and request.FILES.get('file'):
        # save file in db 
        instance = Video(video=request.FILES['file'])
        instance.save()
        print("file saved in db")

        output_path, counts_over_frames = process_video_yolo(request)
        
        file = FileWrapper(open(output_path, 'rb'))
        print("returning processed video")
        # save to db
        file_obj = File(open(output_path, 'rb'), name="out_yolo.mp4")
        processed_instance = Processed_video(video=instance ,processed_video=file_obj, list_nums_persons=json.dumps(counts_over_frames))
        processed_instance.save()
        # Return processed video
        
        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'
        response['X-list-counts'] = json.dumps(counts_over_frames)
        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        # Add CORS headers
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST'
        response['Access-Control-Allow-Headers'] = '*'
    
        return response
    elif not request.FILES.get('file'):
        print(JsonResponse({'error': 'something wrong with the body of the request'}, status=400))
        return JsonResponse({'error': 'something wrong with the body of the request'}, status=400)
    return JsonResponse({'error': 'either the request is not POST or body doesnt have the correct key for accessing the video file'}, status=400)


def process_video_yolo(request):
    """
    return output_path
    """
    # Set device
    device = 'cpu'
    
    # Load model 
    model = YOLO("best_yolo.pt")
    model.eval()

    model.to(device)
    print(f"model loaded : device=",device)
    
    # Save uploaded file temporarily
    file = request.FILES['file']
    temp_path = "aimodel/temp_files/temp.mp4"
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

    print("starting to process video")
    # Process video
    cap = cv2.VideoCapture(temp_path)
    output_path = 'aimodel/temp_files/out_yolo.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                            (int(cap.get(3)), int(cap.get(4))))
    global video_num_frames
    video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_num_frames)
    
    
    frames_counter = 0
    time1 = time.time()
    counts_over_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        print(f"passing frame [{frames_counter}] to model")    
        # Preprocess frame and run inference
        
        #frame = torchvision.transforms.functional.resize(frame, [1056, 1920])
        inputs = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
        inputs = torchvision.transforms.functional.resize(inputs, [1056, 1920])
        with torch.no_grad():
            outputs = model(inputs)[0]
        #print(type(outputs))
        #print(list(outputs.keys()))
        # Draw boxes for each detection above threshold
        boxes_count = 0
        
        for box in outputs.boxes.xyxy.cpu().numpy():
            boxes_count += 1
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"number person : {boxes_count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
        out.write(frame)
        counts_over_frames.append(boxes_count)
        #print("frame writen")
        frames_counter +=1
    time2 = time.time()
    print(f"video processed in {time2-time1}") # 4-7 sec for each frame approx.
    
    cap.release()
    out.release()
    
    print("cleaning up temp files")
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path, counts_over_frames
    
def hold_on(request):
    hello = request
    # do nothing about it
    print("i am holding on ...", flush=True)
    return JsonResponse({'msg' : "i am holding on...", 'num_frames': video_num_frames})

@csrf_exempt
@require_http_methods(["POST"])
def test(request):
    file = request.FILES['file']
    temp_path = "aimodel/temp_files/temp.mp4"
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)
    file = FileWrapper(open(temp_path, 'rb'))
    
    response = HttpResponse(file, content_type='video/mp4')
    response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'

    # Add CORS headers
    response['Access-Control-Allow-Origin'] = '*'
    response['Access-Control-Allow-Methods'] = 'POST'
    response['Access-Control-Allow-Headers'] = '*'

    return response

@csrf_exempt
@require_http_methods(["POST"])
def detect_yolo_lower_fps(request):
    print(request.FILES.get('file'))
    print(request.FILES['file'])
    if request.method == 'POST' and request.FILES.get('file'):
        # save file in db 
        instance = Video(video=request.FILES['file'])
        instance.save()
        print("file saved in db")

        output_path, counts_over_frames = process_video_yolo_lower_fps(request)
        
        file = FileWrapper(open(output_path, 'rb'))
        print("returning processed video")
        # save to db
        file_obj = File(open(output_path, 'rb'), name="out_yolo.mp4")
        processed_instance = Processed_video(video=instance ,processed_video=file_obj, list_nums_persons=json.dumps(counts_over_frames))
        processed_instance.save()
        # Return processed video
        
        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'
        response['X-list-counts'] = json.dumps(counts_over_frames)
        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        # Add CORS headers
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST'
        response['Access-Control-Allow-Headers'] = '*'
    
        return response
    elif not request.FILES.get('file'):
        print(JsonResponse({'error': 'something wrong with the body of the request'}, status=400))
        return JsonResponse({'error': 'something wrong with the body of the request'}, status=400)
    return JsonResponse({'error': 'either the request is not POST or body doesnt have the correct key for accessing the video file'}, status=400)

def process_video_yolo_lower_fps(request):
    """
    return output_path
    """
    # Set device
    device = 'cpu'
    
    # Load model 
    model = YOLO("best_yolo.pt")
    model.eval()

    model.to(device)
    #model.eval()
    print(f"model loaded : device=",device)
    
    # Save uploaded file temporarily
    file = request.FILES['file']
    temp_path = "aimodel/temp_files/temp.mp4"
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)
    
    print("starting to process video")
    # Process video
    cap = cv2.VideoCapture(temp_path)
    output_path = 'aimodel/temp_files/out_yolo.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                            (int(cap.get(3)), int(cap.get(4))))
    global video_num_frames
    video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_num_frames)
    
    
    frames_counter = 0
    time1 = time.time()
    counts_over_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        # ATTENTION : not all frames
        if frames_counter % FPS_COEFF == 0:
            print(f"passing frame [{frames_counter}] to model")    
            # Preprocess frame and run inference
            
            #frame = torchvision.transforms.functional.resize(frame, [1056, 1920])
            inputs = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
            inputs = torchvision.transforms.functional.resize(inputs, [1056, 1920])
            with torch.no_grad():
                outputs = model(inputs)[0]
            #print(type(outputs))
            #print(list(outputs.keys()))
            # Draw boxes for each detection above threshold
            boxes_count = 0
            
            for box in outputs.boxes.xyxy.cpu().numpy():
                boxes_count += 1
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(frame, f"number person : {boxes_count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
            for i in range(FPS_COEFF):
                out.write(frame)
            counts_over_frames.append(boxes_count)
            #print("frame writen")
        frames_counter +=1
    time2 = time.time()
    print(f"video processed in {time2-time1}") # 4-7 sec for each frame approx.
    
    cap.release()
    out.release()
    
    print("cleaning up temp files")
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path, counts_over_frames

@csrf_exempt
@require_http_methods(["POST"])
def detect_fasterrcnn_lower_fps(request):
    print(request.FILES)
    print(request.FILES)
    if request.method == 'POST' and request.FILES.get('file'):
        # save file in db 
        instance = Video(video=request.FILES['file'])
        instance.save()
        print("file saved in db")

        output_path, counts_over_frames = process_video_lower_fps(request)
        
        print("returning processed video")
         # save to db
        file_obj = File(open(output_path, 'rb'), name="out_yolo.mp4")
        processed_instance = Processed_video(video=instance ,processed_video=file_obj, list_nums_persons=json.dumps(counts_over_frames))
        processed_instance.save()
        # Return processed video
        file = FileWrapper(open(output_path, 'rb'))
        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'
        response['X-list-counts'] = json.dumps(counts_over_frames)
        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        # Add CORS headers
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST'
        response['Access-Control-Allow-Headers'] = '*'
    
        return response
    elif not request.FILES.get('file'):
        print(JsonResponse({'error': 'something wrong with the body of the request'}, status=400))
        return JsonResponse({'error': 'something wrong with the body of the request'}, status=400)
    return JsonResponse({'error': 'either the request is not POST or body doesnt have the correct key for accessing the video file'}, status=400)

def process_video_lower_fps(request):
    """
    return output_path
    """
    # Set device
    device = 'cpu'
    
    # Load model (you'll need to replace this with your actual model loading code)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.eval()

    model.to(device)
    model.eval()
    print(f"model loaded : device=",device)
    
    # Save uploaded file temporarily
    file = request.FILES['file']
    temp_path = "aimodel/temp_files/temp.mp4"
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)
    
    print("starting to process video")
    # Process video
    cap = cv2.VideoCapture(temp_path)
    output_path = 'aimodel/temp_files/out_fasterrcnn.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                            (int(cap.get(3)), int(cap.get(4))))
    global video_num_frames
    video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_num_frames)
    
    
    frames_counter = 0
    time1 = time.time()
    counts_over_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        # ATTENTION : not all frames
        if frames_counter % FPS_COEFF == 0:
            print(f"passing frame [{frames_counter}] to model")    
            # Preprocess frame and run inference
            inputs = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(inputs)[0]
            
            # Draw boxes for each detection above threshold
            boxes_count = 0
            for box in outputs['boxes'].detach().cpu().numpy():
                boxes_count += 1
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cv2.putText(frame, f"number person : {boxes_count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
            for i in range(FPS_COEFF):
                out.write(frame)
            counts_over_frames.append(boxes_count)
            #print("frame writen")
        frames_counter +=1
    time2 = time.time()
    print(f"video processed in {time2-time1}") # 4-7 sec for each frame approx.
    
    cap.release()
    out.release()
    
    print("cleaning up temp files")
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path, counts_over_frames

@csrf_exempt
@require_http_methods(["POST"])
def detect_image(request):
    print(request.FILES)
    print(request.FILES)
    if request.method == 'POST' and request.FILES.get('file'):

        output_path = process_img(request)
        
        print("returning processed video")
        
        file_obj = File(open(output_path, 'rb'), name="out_img")
        
        # Return processed video
        file = FileWrapper(open(output_path, 'rb'))
        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='image/jpeg')
        response['Content-Disposition'] = 'attachment; filename=out_img.jpeg'
        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        # Add CORS headers
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST'
        response['Access-Control-Allow-Headers'] = '*'
    
        return response
    elif not request.FILES.get('file'):
        print(JsonResponse({'error': 'something wrong with the body of the request'}, status=400))
        return JsonResponse({'error': 'something wrong with the body of the request'}, status=400)
    return JsonResponse({'error': 'either the request is not POST or body doesnt have the correct key for accessing the video file'}, status=400)

def process_img(request):
    """
    return output_path
    """
    # Set device
    device = 'cpu'
    
    # Load model (you'll need to replace this with your actual model loading code)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = 2  # 1 class (person) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.eval()

    model.to(device)
    model.eval()
    print(f"model loaded : device=",device)
    
    # Save uploaded file temporarily
    file = request.FILES['file']
    temp_path = "aimodel/temp_files/temp_img.jpeg"
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)
    # convert to jpeg
    
    
    
    
    print("starting to process img")
    # Process video
    cap = cv2.imread(temp_path, cv2.IMREAD_COLOR)
    output_path = 'aimodel/temp_files/out_img.jpeg'

    print(f"passing image to model")    
    # Preprocess frame and run inference
    inputs = torchvision.transforms.functional.to_tensor(cap).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)[0]
    
    # Draw boxes for each detection above threshold
    boxes_count = 0
    for box in outputs['boxes'].detach().cpu().numpy():
        boxes_count += 1
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(cap, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.putText(cap, f"number person : {boxes_count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
    cv2.imwrite(output_path, cap)

    
    print("cleaning up temp files")
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path

@csrf_exempt
@require_http_methods(["POST"])
def detect_ssd(request):
    print(request.FILES)
    print(request.FILES)
    if request.method == 'POST' and request.FILES.get('file'):
        # save file in db 
        instance = Video(video=request.FILES['file'])
        instance.save()
        print("file saved in db")

        output_path, counts_over_frames = process_video_ssd(request)
        
        print("returning processed video")
         # save to db
        file_obj = File(open(output_path, 'rb'), name="out_yolo.mp4")
        processed_instance = Processed_video(video=instance ,processed_video=file_obj, list_nums_persons=json.dumps(counts_over_frames))
        processed_instance.save()
        # Return processed video
        file = FileWrapper(open(output_path, 'rb'))
        with open(output_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'
        response['X-list-counts'] = json.dumps(counts_over_frames)
        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        # Add CORS headers
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST'
        response['Access-Control-Allow-Headers'] = '*'
    
        return response
    elif not request.FILES.get('file'):
        print(JsonResponse({'error': 'something wrong with the body of the request'}, status=400))
        return JsonResponse({'error': 'something wrong with the body of the request'}, status=400)
    return JsonResponse({'error': 'either the request is not POST or body doesnt have the correct key for accessing the video file'}, status=400)

def process_video_ssd(request):
    """
    return output_path
    """
    # Set device
    device = 'cpu'

    
    # Load model (you'll need to replace this with your actual model loading code)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', map_location='cpu')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    #num_classes = 2  # 1 class (person) + background
    #in_features = model.roi_heads.box_predictor.cls_score.in_features
    #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #model.load_state_dict(torch.load("model_weights.pth", map_location=device, weights_only=True))
    model.eval()
    classes_to_labels = utils.get_coco_object_dictionary()
    model.to(device)
    #model.eval()
    print(f"model loaded : device=",device)
    
    # Save uploaded file temporarily
    file = request.FILES['file']
    temp_path = "aimodel/temp_files/temp.mp4"
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)
    
    print("starting to process video")
    # Process video
    cap = cv2.VideoCapture(temp_path)
    output_path = 'aimodel/temp_files/out_fasterrcnn.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                            (int(cap.get(3)), int(cap.get(4))))
    global video_num_frames
    video_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_num_frames)
    
    
    frames_counter = 0
    time1 = time.time()
    counts_over_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        print(f"passing frame [{frames_counter}] to model")    
        # Preprocess frame and run inference
        #inputs = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0).to(device)
        inputs = utils.prepare_input(frame)
        tensor = utils.prepare_tensor(inputs)
        with torch.no_grad():
            outputs = model(tensor)
        
        results_per_input = utils.decode_results(outputs)
        best_results_per_input = utils.pick_best(results_per_input, 0.40)

        bboxes, classes, confidences = best_results_per_input
        
        # Draw boxes for each detection above threshold
        boxes_count = 0
        for box in bboxes.detach().cpu().numpy():
            if 1 in classes:
                boxes_count += 1
                left, bot, right, top = box
                x1, y1, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
                x1, y1, x2, y2 = box.astype(int)
                x2 = x1 + w
                y2 = y1 + h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, f"number person : {boxes_count}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),3)
        out.write(frame)
        counts_over_frames.append(boxes_count)
        #print("frame writen")
        frames_counter +=1
    time2 = time.time()
    print(f"video processed in {time2-time1}") # 4-7 sec for each frame approx.
    
    cap.release()
    out.release()
    
    print("cleaning up temp files")
    # Clean up temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    return output_path, counts_over_frames


"""
A class for storing and manipulating inference results.

    This class provides comprehensive functionality for handling inference results from various
    Ultralytics models, including detection, segmentation, classification, and pose estimation.
    It supports visualization, data export, and various coordinate transformations.

    Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        speed (dict): Dictionary containing inference speed information.
        names (dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Update the Results object with new detection data.
        cpu: Return a copy of the Results object with all tensors moved to CPU memory.
        numpy: Convert all tensors in the Results object to numpy arrays.
        cuda: Move all tensors in the Results object to GPU memory.
        to: Move all tensors to the specified device and dtype.
        new: Create a new Results object with the same image, path, names, and speed attributes.
        plot: Plot detection results on an input RGB image.
        show: Display the image with annotated inference results.
        save: Save annotated inference results image to file.
        verbose: Return a log string for each task in the results.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        summary: Convert inference results to a summarized dictionary.
        to_df: Convert detection results to a Pandas Dataframe.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.
        to_xml: Convert detection results to XML format.
        to_html: Convert detection results to HTML format.
        to_sql: Convert detection results to an SQL-compatible format.

    Examples:
        >>> results = model("path/to/image.jpg")
        >>> result = results[0]  # Get the first result
        >>> boxes = result.boxes  # Get the boxes for the first result
        >>> masks = result.masks  # Get the masks for the first result
        >>> for result in results:
        >>>     result.plot()  # Plot detection results
"""


"""
A class for managing and manipulating detection boxes.

    This class provides comprehensive functionality for handling detection boxes, including their coordinates,
    confidence scores, class labels, and optional tracking IDs. It supports various box formats and offers
    methods for easy manipulation and conversion between different coordinate systems.

    Attributes:
        data (torch.Tensor | numpy.ndarray): The raw tensor containing detection boxes and associated data.
        orig_shape (Tuple[int, int]): The original image dimensions (height, width).
        is_track (bool): Indicates whether tracking IDs are included in the box data.
        xyxy (torch.Tensor | numpy.ndarray): Boxes in [x1, y1, x2, y2] format.
        conf (torch.Tensor | numpy.ndarray): Confidence scores for each box.
        cls (torch.Tensor | numpy.ndarray): Class labels for each box.
        id (torch.Tensor | None): Tracking IDs for each box (if available).
        xywh (torch.Tensor | numpy.ndarray): Boxes in [x, y, width, height] format.
        xyxyn (torch.Tensor | numpy.ndarray): Normalized [x1, y1, x2, y2] boxes relative to orig_shape.
        xywhn (torch.Tensor | numpy.ndarray): Normalized [x, y, width, height] boxes relative to orig_shape.

    Methods:
        cpu: Return a copy of the object with all tensors on CPU memory.
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        numpy: Return a copy of the object with all tensors as numpy arrays.
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        cuda: Return a copy of the object with all tensors on GPU memory.
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:
        to: Return a copy of the object with tensors on specified device and dtype.

    Examples:

    Examples:
    Examples:
        >>> import torch
        >>> import torch
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes_data = torch.tensor([[100, 50, 150, 100, 0.9, 0], [200, 150, 300, 250, 0.8, 1]])
        >>> orig_shape = (480, 640)  # height, width
        >>> orig_shape = (480, 640)  # height, width
        >>> boxes = Boxes(boxes_data, orig_shape)
        >>> print(boxes.xyxy)
        >>> print(boxes.conf)
        >>> print(boxes.cls)
        >>> print(boxes.xywhn)
"""