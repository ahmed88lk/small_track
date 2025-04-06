import cv2
import supervision as sv
from ultralytics import YOLOv10
from supervision.utils.video import get_video_frames_generator
from supervision.annotators.core import BoxAnnotator, LabelAnnotator
import os

os.makedirs("models", exist_ok=True)

# Check if model exists, if not, attempt to download or notify user
model_path = os.path.join("models", "b6.pt")
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}. Please ensure the model file exists.")
    print("You can either:")
    print("1. Place your trained model file at this location")
    print("2. Modify the model_path variable to point to your model file")
    # Uncomment the line below if you want to download a default model
    # model = YOLOv10()  # This will download the default model
    exit(1)

model = YOLOv10(model_path)

# Initialize the ByteTrack tracker
tracker = sv.ByteTrack()

# Define the video path
video_path = 'videos/v5.mp4'

# Initialize the BoxAnnotator and LabelAnnotator
color_annotator = sv.ColorAnnotator()
label_annotator = LabelAnnotator()
smoother = sv.DetectionsSmoother()
mask_annotator = sv.MaskAnnotator()

# To track the number of unique flies
unique_tracker_ids = set()

# Set the confidence threshold
confidence_threshold = 0.55

# Get the video frames generator
generator = get_video_frames_generator(video_path)

# Define the callback function for slicing inference
def callback(image_slice) -> sv.Detections:
    result = model(image_slice)[0]
    detections = sv.Detections.from_ultralytics(result)
    # Filter detections based on confidence threshold
    detections = detections[detections.confidence >= confidence_threshold]
    return detections

# Initialize the slicer with the callback
slicer = sv.InferenceSlicer(callback=callback)

for frame in generator:
    # Perform slicing inference
    detections = slicer(frame)
    
    # Update tracker with detections
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)
    
    # Update unique tracker IDs
    for tracker_id in detections.tracker_id:
        if tracker_id not in unique_tracker_ids:
            unique_tracker_ids.add(tracker_id)
    
    # Prepare labels for tracked objects
    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    
    # Annotate the frame with bounding boxes and labels
    annotated_frame = color_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # Display the count of unique flies
    fly_count_text = f"Unique Flies: {len(unique_tracker_ids)}"
    cv2.putText(annotated_frame, fly_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame with annotations and fly count
    cv2.imshow('Annotated Video', annotated_frame)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources
cv2.destroyAllWindows()
