# Crowd-Counting-App

To contribute : 
1. create a virtual environement ``python -m venv .venv``
2. use the venv ``.venv/scripts/activate``
3. install all required libraries from requirements.txt
4. run ``python manage.py runserver``

# Endpoints
set the url : localhost:8000/
## Faster R-CNN
#### POST : `url/aimodel/detect_fasterrcnn`
=> in the Body : form-data
| key    | value |
| -------- | ------- |
| 'file'  | 'path_to_video_file'|


=> response : in headers
| key    | value |
| -------- | ------- |
| 'Content-Type'  | 'video/mp4'|
| 'Content-Disposition'  | 'attachment; filename=processed_video.mp4'|
| 'X-list-counts'  | [5,4,6,7] |




## YOLO
#### POST : `url/aimodel/detect_yolo`
=> in the Body : form-data
| key    | value |
| -------- | ------- |
| 'file'  | 'path_to_video_file'|


=> response : in headers
| key    | value |
| -------- | ------- |
| 'Content-Type'  | 'video/mp4'|
| 'Content-Disposition'  | 'attachment; filename=processed_video.mp4'|
| 'X-list-counts'  | [5,4,6,7] |


yep done