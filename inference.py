from ultralytics import YOLO
import cv2

img_folder_path = "/Users/itamrakar/Documents/Projects/fast_object_detection_dataset/data/A001_09221707_C006_frames"
model = YOLO("YOLOv8-Finetuned/run3/weights/best.pt")
results = model.predict(source= img_folder_path, imgsz=640, conf=0.25)

for r in results:
    im_array = r.plot()  
    cv2.imshow("Detections", im_array)
    # wait until q is pressed for next frame
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
