import cv2
from ultralytics import YOLO

class YOLOVideoInspector: 
    def __init__(self, model_path="yolov8n.pt",video_path=None):
        self.model = YOLO(model_path)
        self.video_path = video_path 
        self.cap = None 
        
    def open_video(self):
        if self.video_path is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video source: {self.video_path if self.video_path else 'camera'}")
        
    def process_frame(self, frame):
        results = self.model(frame)
        annotated_frame = results[0].plot()
        return annotated_frame
    
    def run(self):
        self.open_video()
        while True: 
            ret, frame = self.cap.read()
            if not ret:
                break
            annotated_frame = self.process_frame(frame)
            cv2.imshow("YOLOv8 Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    video_file = "/Users/itamrakar/Documents/Projects/fast_dataset/data/IMG_3550.MOV"
    inspector = YOLOVideoInspector(model_path="yolov8n.pt", video_path=video_file)
    inspector.run()
    