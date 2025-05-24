from django.shortcuts import render
from django.http import HttpResponse, FileResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import *
from wsgiref.util import FileWrapper
import json



import cv2
import torchvision
import torch
import os
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
import time

video_num_frames = -1

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
        # Return processed video
        file = FileWrapper(open(output_path, 'rb'))
        response = HttpResponse(file, content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'
        response['X-list-counts'] = json.dumps(counts_over_frames)
        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
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
    output_path = 'aimodel/temp_files/out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

def detect_yolo(request):
    print(request.FILES.get('vid'))
    print(request.FILES['vid'])
    if request.method == 'POST' and request.FILES.get('vid'):
        # save file in db 
        instance = Video(video=request.FILES['vid'])
        instance.save()
        print("file saved in db")

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
        file = request.FILES['vid']
        temp_path = "aimodel/temp_files/temp.mp4"
        with open(temp_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        
        print("starting to process video")
        # Process video
        cap = cv2.VideoCapture(temp_path)
        output_path = 'aimodel/temp_files/out.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, 
                             (int(cap.get(3)), int(cap.get(4))))
        
        frames_counter = 0
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
            for box in outputs['boxes'].detach().cpu().numpy():
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            out.write(frame)
            #print("frame writen")
            frames_counter +=1
            
        
        cap.release()
        out.release()
        
        print("cleaning up temp files")
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        print("returning processed video")
        # Return processed video
        file = FileWrapper(open(output_path, 'rb'))
        response = HttpResponse(file, content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename=processed_video.mp4'

        #response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        #response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        return response
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def hold_on(request):
    hello = request
    # do nothing about it
    print("i am holding on ...", flush=True)
    return JsonResponse({'msg' : "i am holding on...", 'num_frames': video_num_frames})