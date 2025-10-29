import cv2
from ultralytics import YOLO
import os

class YOLOVideoInspector: 
    def __init__(self, model_path="yolov8n.pt",video_path=None):
        self.model = YOLO(model_path)
        self.video_path = video_path 
        self.cap = None 
        self.video = None
        
    def open_video(self):
        if self.video_path is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        # else if video_path is a directory, read all frames in the folder in sorted order
        elif os.path.isdir(self.video_path):
            image_files = sorted([os.path.join(self.video_path, f) for f in os.listdir(self.video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            # create a video of the images in memory using cv2.VideoCapture and save it to self.video
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            height, width = cv2.imread(image_files[0]).shape[:2]
            self.video = cv2.VideoWriter('temp_video.avi', fourcc, 50.0, (width, height))
            for img_file in image_files:
                frame = cv2.imread(img_file)
                self.video.write(frame)
            self.video.release()
            self.cap = cv2.VideoCapture('temp_video.avi')
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video source: {self.video_path if self.video_path else 'camera'}")
    

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        return annotated_frame
    
    def process_frame_raw(self, frame):
        results = self.model(frame)
        return results
    
    def run(self):
        self.open_video()
        while True: 
            ret, frame = self.cap.read()
            if not ret:
                break
            annotated_frame = self.process_frame(frame)
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()


        
if __name__ == "__main__":
    video_file = "data/sequence_000000/proc/flir/frame"
    inspector = YOLOVideoInspector(model_path="checkpoints/yolo11m_375_test.pt", video_path=video_file)
    inspector.run()