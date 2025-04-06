import cv2
import supervision as sv
from ultralytics import YOLOv10
from supervision.utils.video import get_video_frames_generator

#load the YOLOv10 model
model=YOLOv10('models/b6.pt')

#initialize the ByteTrack tracker
traker=sv.ByteTrack()

#initialize the BoxAnnotator and LabelAnnotator and the smoother
color_annotator=sv.ColorAnnotator()
label_annotator=sv.LabelAnnotator()
smoother=sv.DetectionsSmoother()
mask_annotator=sv.MaskAnnotator()


unqiue_tracker_ids=set()
confidence_threshold=0.55

video_path='videos/v12.mp4'
genrator=get_video_frames_generator(video_path)
fly_count=0

def genrate_frames():
    def callback(image_slice) -> sv.Detections:
        result = model(image_slice)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence >= confidence_threshold]
        return detections

    slicer = sv.InferenceSlicer(callback=callback)
    
    for frame in genrator:
        # Perform slicing and detection
        detections = slicer(frame)        
        detections = traker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
       
        # Track unique flies
        for tracker_id in detections.tracker_id:
            if tracker_id not in unqiue_tracker_ids:
                unqiue_tracker_ids.add(tracker_id)
               
        # Create labels and annotate frame
        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = color_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        global  fly_count
        fly_count=len(unqiue_tracker_ids)
        fly_count_text= f"Number of flies: {len(unqiue_tracker_ids)}"
        cv2.putText(annotated_frame, fly_count_text, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), cv2.LINE_4)
        
        # Encode the frame
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        # Convert to bytes and yield for streaming
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

               
               
               
def genrate_frame():
    cap=cv2.VideoCapture(0)
    while True:
        ret,frame=cap.read()
        
        if not ret:
            break
       
        ret,buffer=cv2.imencode('.jpg',frame)
        if not ret:
            continue
       
        frame=buffer.tobytes()
        yield(b'--frame\r\n'
                         b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')       


def  flies_number():
    while True:
        
        yield fly_count
        
