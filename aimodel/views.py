from django.shortcuts import render
from django.http import HttpResponse, FileResponse, JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import *


import cv2
import torchvision
import torch
import os
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor



def index(request):
    return HttpResponse("Hello, world. You're at the ai model index.")

@csrf_exempt
@require_http_methods(["POST"])
def detect(request):
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
        response = FileResponse(open(output_path, 'rb'), content_type='video/mp4')
        response['Content-Disposition'] = 'attachment; filename="processed_video.mp4"'
        return response
    
    return JsonResponse({'error': 'Invalid request'}, status=400)

def testo(request):
    return HttpResponse('hello testo')