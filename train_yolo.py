import cv2
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2 
import os

def main(): 
    
    transform = A.Compose([
    A.RandomCrop(width=640, height=640),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.2),
    ToTensorV2()
    ])
    
    project_dir = "YOLOv11-Finetuned"
    yaml_path = "data/collated_dataset/data.yaml"
    os.environ["WANDB_DISABLED"] = "true"
    best_ckpt_path = f"{project_dir}/run1/weights/best.pt"
    if os.path.exists(best_ckpt_path):
        model = YOLO(best_ckpt_path)
        resume_flag = False   # start a NEW training from best.pt
        print("Continuing training from best checkpoint (new run).")
    else:
        model = YOLO('yolo11m.pt')
        resume_flag = False
        print("Starting training from scratch.")

    model.train(data=yaml_path, 
                epochs=300,  
                imgsz=640, 
                batch=16, 
                exist_ok=True, 
                resume=resume_flag, 
                project='YOLOv11-Finetuned',
                name='run1',
                augment=True,
                plots=True,
                save=True)
    model.val(data = yaml_path) 
    
    print("Training and validation complete.")
    
              

if __name__ == "__main__":
    main()