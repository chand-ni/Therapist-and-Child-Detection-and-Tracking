Person Detection and Tracking System for ASD Therapies
Overview
This project implements a person detection and tracking system designed to identify and track children with Autism Spectrum Disorder (ASD) and therapists in video footage. Using YOLOv5 for person detection and DeepSORT for tracking, the system assigns unique IDs to individuals, handles occlusions and re-entries, and provides post-occlusion tracking.

The system processes long-duration videos and generates output with bounding boxes around detected individuals (children and adults), along with unique ID numbers that are tracked across the video.

Key Features:
Person Detection using YOLOv5 pre-trained on the COCO dataset.
Unique ID Assignment and Tracking using DeepSORT.
Post-occlusion Tracking ensures that IDs are retained even if the person is briefly occluded.
Handling Re-entries by assigning new IDs to persons entering for the first time and re-using existing IDs for persons re-entering the frame.
Getting Started
1. Clone the repository:
bash
Copy code
git clone https://github.com/your-repository.git
cd your-repository
2. Install Dependencies:
You need Python 3.x installed. Install the required libraries using pip:

bash
Copy code
pip install opencv-python torch torchvision
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
pip install deep_sort_realtime
3. Download Test Videos:
Download the test video from the provided Google Drive link here and place it in the project directory. Rename the file to test_video.mp4 or adjust the file path in the script accordingly.

4. Run the Inference Script:
To process the video and generate the output video with bounding boxes and unique IDs, run the following command:

bash
Copy code
python person_detection_tracking.py
5. View Output:
The output video with tracked persons will be saved as output_video.mp4 in the project directory.

Project Structure
bash
Copy code
.
├── yolov5/                     # YOLOv5 model directory (cloned from YOLOv5 repo)
├── person_detection_tracking.py # Main script for detection and tracking
├── README.md                    # Documentation file
├── requirements.txt             # Dependencies file (optional)
├── test_video.mp4               # Test video file (add manually)
├── output_video.mp4             # Output video with predictions (generated)
Main Components:
person_detection_tracking.py: This script performs the following:
Loads the YOLOv5 model to detect persons in the video frames.
Tracks persons using DeepSORT, which assigns unique IDs, handles occlusions, and tracks re-entries.
Writes the processed frames to a new output video file with overlaid bounding boxes and unique IDs.
Explanation of Key Steps
Person Detection:

The YOLOv5 model is used to detect individuals in each video frame. It outputs bounding boxes and class labels (we filter for the person class).
Tracking with DeepSORT:

DeepSORT takes the detected bounding boxes from YOLOv5 and tracks each individual across frames by assigning unique IDs.
It handles:
Occlusion: When individuals are partially or completely occluded, DeepSORT tries to continue tracking the individual and assigns the correct ID once they reappear.
Re-entry: If an individual exits the frame and then re-enters, DeepSORT either assigns a new ID or reassigns the original ID based on the appearance features.
Output Video:

The system overlays the detection results (bounding boxes and unique IDs) onto each frame and saves the video for analysis.
Customization
You can fine-tune the model and tracker settings by adjusting parameters in person_detection_tracking.py:

Tracker Parameters: The DeepSORT tracker can be adjusted for the tracking speed, occlusion handling, and re-entry handling by modifying max_age, n_init, and nn_budget parameters.

YOLOv5 Model: You can switch to a different YOLOv5 variant (such as yolov5m, yolov5l, or yolov5x) depending on your resource availability and accuracy requirements.

Results
The output video will display:

Bounding boxes around detected persons.
Unique IDs that are tracked across frames, ensuring the same individual is assigned the same ID throughout the video.
Future Improvements
Multi-class Detection: Extend the model to classify between children and adults using fine-tuning on a custom dataset.
Behavior and Emotion Tracking: Further developments can include emotion recognition and behavior tracking using additional models.
License
This project is licensed under the MIT License