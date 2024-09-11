import cv2
import torch
from yolov5 import detect
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load the YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize DeepSORT for tracking
tracker = DeepSort(max_age=30, n_init=3, nn_budget=100)

# Class names for COCO dataset (Person class ID is 0)
person_class_id = 0

def process_video(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define video writer to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv5 inference on the frame
        results = model(frame)
        detections = results.xywh[0].numpy()  # x_center, y_center, width, height, confidence, class
        
        # Filter detections to only keep 'person' class
        person_detections = [det for det in detections if int(det[5]) == person_class_id]
        
        # Prepare detections for DeepSORT
        bbox_xywh = [[x[0], x[1], x[2], x[3]] for x in person_detections]
        confidences = [x[4] for x in person_detections]
        
        # Update tracker with current frame's detections
        tracks = tracker.update_tracks(bbox_xywh, confidences, frame=frame)
        
        # Loop through the tracked objects and draw bounding boxes with IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltwh()  # Convert to left-top-width-height
            x1, y1, w, h = [int(i) for i in bbox]
            
            # Draw the bounding box and the track ID
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Write the frame with detections to the output video
        out.write(frame)
        
        # Optionally, display the frame (comment out if not needed)
        cv2.imshow('Person Detection & Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test_video.mp4"  # Path to input test video
    output_path = "output_video.mp4"  # Path to save the output video
    
    process_video(video_path, output_path)
