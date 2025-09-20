import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor


class SAM2VideoRunner:
    def __init__(self, frames_folder, model_id="facebook/sam2-hiera-large", device="cpu"):
        self.frames_folder = frames_folder
        
        # split folder name with "_" and take the first part as base name
        self.save_dir = os.path.basename(frames_folder).split("_f")[0] + "_masks"
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {device}")
        
        if self.device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
        self.predictor = SAM2VideoPredictor.from_pretrained(
            "facebook/sam2-hiera-large",
            device=self.device
        )
        self.state = None
        
        self.current_frame = None
        self.labels = {}
        self.points = None
        self.box = None 
        self.vis_frame_stride = 30
        self.annotated_frame = None 
        self.video_segments = {}
        
        self.frame_names = [
            p for p in os.listdir(frames_folder)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        
    def show_mask(self, mask, ax, obj_id=None, random_color=False):
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                cmap = plt.get_cmap("tab10")
                cmap_idx = 0 if obj_id is None else obj_id
                color = np.array([*cmap(cmap_idx)[:3], 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)
          
    def show_mask_opencv(self, mask, frame, obj_id=None, random_color=False, alpha=0.5):
        # Squeeze mask to 2D if needed
        if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
            mask = mask.squeeze(0).squeeze(0)
        elif mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        # Resize mask if dimensions don't match
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32), (frame.shape[1], frame.shape[0]))
        if random_color:
            color_rgb = np.random.randint(0, 256, size=3)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = cmap(cmap_idx)[:3]
            color_rgb = np.array([int(255 * c) for c in color])
        mask_bool = mask > 0.5
        overlay = frame.copy()
        # Apply overlay safely using boolean indexing
        for c in range(3):
            overlay[..., c][mask_bool] = (1 - alpha) * overlay[..., c][mask_bool] + alpha * color_rgb[c]
        overlay = overlay.astype(np.uint8)
        return overlay

    def init_video(self):
        """Initialize video state."""
        with torch.inference_mode():
            self.state = self.predictor.init_state(self.frames_folder)

    def add_point_prompt(self, points, labels, obj_id=1):
        with torch.inference_mode():
            frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state= self.state, frame_idx = 0, points = points, labels = labels, obj_id=obj_id
            )
        return frame_idx, object_ids, masks

    def add_box_prompt(self, box, obj_id=1):
        box_array = np.array([box])
        with torch.inference_mode():
            frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state = self.state, frame_idx = 0, obj_id = obj_id, box = box_array)
        return frame_idx, object_ids, masks

    def add_click_prompt(self):
        frame_files = sorted([f for f in os.listdir(self.frames_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not frame_files:
            print("No image files found in frames folder.")
            return

        first_frame_path = os.path.join(self.frames_folder, frame_files[0])
        original_frame = cv2.imread(first_frame_path)
        if original_frame is None:
            print(f"Failed to load image: {first_frame_path}")
            return

        obj_id = len(self.labels) + 1
        
        if self.annotated_frame is None: 
            self.annotated_frame = original_frame.copy()
            
        
        while True:
            points = []
            labels = []
            frame = self.annotated_frame.copy()
            fresh_frame = original_frame.copy()

            def click_point(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    print(f"Clicked at {x}, {y}")
                    points.append([float(x), float(y)])  # <-- float list
                    labels.append(1)
                    cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                    cv2.imshow("Frame", frame)

            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame", frame)
            cv2.setMouseCallback("Frame", click_point)
            

            print(f"Object {obj_id}: Please click points (left click). Press ENTER to confirm, ESC to finish all objects.")

            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == 13 or key == 10:  # ENTER key
                    if points:
                        # Convert points and labels to correct format before passing
                        coords = np.array(points, dtype=np.float32).reshape(-1, 2)
                        lbls = np.array(labels, dtype=np.int32).reshape(-1)
                        frame_idx, object_ids, masks = self.add_point_prompt(coords, lbls, obj_id=obj_id)
                        
                        # Visualize the mask on the original frame
                        self.annotated_frame = fresh_frame.copy()
                        for i, obj_id in enumerate(object_ids):
                            self.annotated_frame = self.show_mask_opencv(masks[i].cpu().numpy(), self.annotated_frame, obj_id=obj_id, random_color=False, alpha=0.5)
                        cv2.imshow("Frame", self.annotated_frame)
                        
                        
                        # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                        # ax.imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                        # for i, obj_id in enumerate(object_ids):
                        #     self.show_mask(masks[i].cpu().numpy(), ax, obj_id=obj_id, random_color=False)
                        # plt.axis('off')
                        # plt.show()
                        
                        object_name = f"obj_{obj_id}" # placeholder 
                        self.labels[object_name] = obj_id
                        print(f"Added object {obj_id} with {len(points)} points.")
                        obj_id = len(self.labels) + 1
                        
                    else:
                        print("No points clicked for this object.")
                    break
                elif key == 27:  # ESC key
                    cv2.destroyAllWindows()
                    cv2.waitKey(1) 
                    print("Finished adding objects.")
                    print(f"Labels: {self.labels}")
                    return
            
    def propagate(self, save = True, visualize=False):
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.state):
                self.video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                if save: 
                    if not os.path.exists(self.save_dir):
                        os.makedirs(self.save_dir)
                    for obj_id, mask in self.video_segments[out_frame_idx].items():
                        np.save(os.path.join(self.save_dir, f"mask_obj{obj_id}_{out_frame_idx:04d}.npy"), mask)

    # visualize the masks on frames at a given stride 
    def visualize_masks(self, stride=30):
        for out_frame_idx in range(0, len(self.frame_names), stride):
            frame_path = os.path.join(self.frames_folder, self.frame_names[out_frame_idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            if out_frame_idx in self.video_segments:
                for obj_id, mask in self.video_segments[out_frame_idx].items():
                    frame = self.show_mask_opencv(mask, frame, obj_id=obj_id, random_color=False, alpha=0.5)
            cv2.imshow("Mask Visualization", frame)
            cv2.waitKey(100)
        cv2.destroyAllWindows()
        
    # creates bounding boxes for each object in the mask 
    def create_bounding_boxes(self, masks):
        boxes = {}
        for obj_id, mask in masks.items():
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            boxes[obj_id] = (x_min, y_min, x_max, y_max)
        return boxes


if __name__ == "__main__":
    frames_path = "/Users/itamrakar/Documents/Projects/fast_object_detection_dataset/data/IMG_3550_frames"

    runner = SAM2VideoRunner(frames_path, device="cpu")
    runner.init_video()

    # Point prompt with multiple points and multiple objects
    runner.add_click_prompt()

    # Propagate
    runner.propagate(save=True, visualize=True)
    
    # Visualize masks on frames at a given stride
    runner.visualize_masks(stride=1)