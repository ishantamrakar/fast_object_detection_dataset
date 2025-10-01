from ultralytics import YOLO
import cv2

img_folder_path = "/Users/itamrakar/Documents/Projects/fast_object_detection_dataset/data/A001_09221707_C006_frames"
model = YOLO("/Users/itamrakar/Documents/Projects/fast_object_detection_dataset/checkpoints/best_yolov11_300.pt")
# results = model.predict(source= img_folder_path, imgsz=640, conf=0.25)
results = model.track(img_folder_path, show=True) #tracker="bytetrack.yaml")  # with ByteTrack. there is also botsort.yaml

print("\n\n\n\n Live Detections: \n")

for r in results:
    # r consists of boxes and boxes consists of cls, conf, data (xywh)
    # print(r)
    im_array = r.plot()  
    cv2.imshow("Detections", im_array)
    # wait until q is pressed for next frame
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
