import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor
import shutil
import random
import yaml

# Import the label selection GUI function
from run_gui import get_label


class SAM2VideoRunner:
    def __init__(self, frames_folder, model_id="facebook/sam2-hiera-large", device="cpu"):
        self.frames_folder = frames_folder
        
        self.save_dir = os.path.basename(frames_folder).split("_f")[0] + "_masks"
        self.dataset_dir = os.path.join(os.path.dirname(frames_folder), "yolov8_dataset")

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
                        for i, oid in enumerate(object_ids):
                            self.annotated_frame = self.show_mask_opencv(masks[i].cpu().numpy(), self.annotated_frame, obj_id=oid, random_color=False, alpha=0.5)
                        cv2.imshow("Frame", self.annotated_frame)

                        # --- LABEL SELECTION GUI ---
                        # Show the GUI to select a label for this object
                        print("Please select a label for the object (after closing this window).")
                        selected_label, current_mapping = get_label()
                        print("selected label: ", selected_label)
                        if selected_label is None:
                            print("No label selected. Skipping object.")
                            break
                        # Assign the selected label to this object id
                        self.labels = current_mapping
                        print(f"Added object {obj_id} with label '{selected_label}' and {len(points)} points.")
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
    def visualize_masks(self, stride=30, bbox=True):
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
    def create_bounding_boxes(self, visualize=False):
        boxes = {}
        
        for out_frame_idx in self.video_segments:
            masks = self.video_segments[out_frame_idx]
            frame_path = os.path.join(self.frames_folder, self.frame_names[out_frame_idx])
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            for obj_id, mask in masks.items():
                # Squeeze mask to 2D if needed
                if mask.ndim == 4 and mask.shape[0] == 1 and mask.shape[1] == 1:
                    mask_squeezed = mask.squeeze(0).squeeze(0)
                elif mask.ndim == 3 and mask.shape[0] == 1:
                    mask_squeezed = mask.squeeze(0)
                else:
                    mask_squeezed = mask
                # Resize mask if dimensions don't match
                if mask_squeezed.shape != frame.shape[:2]:
                    mask_resized = cv2.resize(mask_squeezed.astype(np.float32), (frame.shape[1], frame.shape[0]))
                else:
                    mask_resized = mask_squeezed
                ys, xs = np.where(mask_resized > 0.5)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                boxes[(obj_id, out_frame_idx)] = (x_min, y_min, x_max, y_max)
                
                if visualize:
                    # visualize the box on the frame with consistent color
                    cmap = plt.get_cmap("tab10")
                    cmap_idx = obj_id if obj_id is not None else 0
                    color_float = cmap(cmap_idx)[:3]
                    color = tuple(int(255 * c) for c in color_float)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, f"obj {obj_id}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if visualize:
                cv2.imshow("Bounding Box", frame)
                cv2.waitKey(100)
        cv2.destroyAllWindows()
        return boxes

    def export_yolov8_dataset(self, train_ratio=0.8, write_data_yaml=True):
        root_dir = self.dataset_dir
        images_train_dir = os.path.join(root_dir, "images", "train")
        images_val_dir = os.path.join(root_dir, "images", "val")
        labels_train_dir = os.path.join(root_dir, "labels", "train")
        labels_val_dir = os.path.join(root_dir, "labels", "val")

        os.makedirs(images_train_dir, exist_ok=True)
        os.makedirs(images_val_dir, exist_ok=True)
        os.makedirs(labels_train_dir, exist_ok=True)
        os.makedirs(labels_val_dir, exist_ok=True)

        # Map object names to class ids (0-based)
        sorted_labels = sorted(self.labels.items(), key=lambda x: x[1])
        class_name_to_id = {name: idx for idx, (name, _) in enumerate(sorted_labels)}

        # Get bounding boxes
        boxes = self.create_bounding_boxes(visualize=False)

        # Prepare list of frames with bounding boxes
        frames_with_boxes = set()
        for (obj_id, frame_idx) in boxes.keys():
            frames_with_boxes.add(frame_idx)

        frame_indices = sorted(list(frames_with_boxes))
        num_train = int(len(frame_indices) * train_ratio)
        train_indices = set(frame_indices[:num_train])
        val_indices = set(frame_indices[num_train:])

        for frame_idx in frame_indices:
            frame_name = self.frame_names[frame_idx]
            frame_path = os.path.join(self.frames_folder, frame_name)
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Warning: Failed to read image {frame_path}, skipping.")
                continue

            h, w = img.shape[:2]

            # Determine destination directories
            if frame_idx in train_indices:
                img_dst_dir = images_train_dir
                label_dst_dir = labels_train_dir
            else:
                img_dst_dir = images_val_dir
                label_dst_dir = labels_val_dir

            # Copy image
            img_dst_path = os.path.join(img_dst_dir, frame_name)
            shutil.copyfile(frame_path, img_dst_path)

            # Write label file
            label_file_name = os.path.splitext(frame_name)[0] + ".txt"
            label_file_path = os.path.join(label_dst_dir, label_file_name)

            # Collect all boxes for this frame
            lines = []
            for (obj_id, f_idx), (x_min, y_min, x_max, y_max) in boxes.items():
                if f_idx != frame_idx:
                    continue
                # Find class_id by matching obj_id to labels
                # self.labels maps object_name -> obj_id, so invert it
                # but object_name is like "obj_{obj_id}"
                # So we find the object_name with matching obj_id
                obj_name = None
                for name, id_ in self.labels.items():
                    if id_ == obj_id:
                        obj_name = name
                        break
                if obj_name is None:
                    print(f"Warning: obj_id {obj_id} not found in labels, skipping.")
                    continue
                class_id = class_name_to_id[obj_name]

                # Normalize coordinates to YOLO format: x_center, y_center, width, height
                x_center = ((x_min + x_max) / 2) / w
                y_center = ((y_min + y_max) / 2) / h
                box_width = (x_max - x_min) / w
                box_height = (y_max - y_min) / h

                # Clamp values to [0,1]
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                box_width = min(max(box_width, 0.0), 1.0)
                box_height = min(max(box_height, 0.0), 1.0)

                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

            with open(label_file_path, "w") as f:
                f.write("\n".join(lines))

        if write_data_yaml:
            data_yaml_path = os.path.join(root_dir, "data.yaml")
            data_yaml = {
                'path': root_dir,
                'train': 'images/train',
                'val': 'images/val',
                'names': [name for name, _ in sorted_labels]
            }
            with open(data_yaml_path, "w") as f:
                yaml.dump(data_yaml, f)

        print(f"YOLOv8 dataset exported to {root_dir} with train/val split {train_ratio}")


if __name__ == "__main__":
    frames_path = "/Users/itamrakar/Documents/Projects/fast_object_detection_dataset/data/IMG_3550_frames"

    runner = SAM2VideoRunner(frames_path, device="cpu")
    runner.init_video()

    # Point prompt with multiple points and multiple objects
    runner.add_click_prompt()

    # Propagate
    runner.propagate(save=True, visualize=True)
    
    # Visualize masks on frames at a given stride
    # runner.visualize_masks(stride=1)
    
    # show bounding boxes
    boxes = runner.create_bounding_boxes(visualize=True)
    
    # Export to YOLOv8 format
    runner.export_yolov8_dataset(train_ratio=0.8, write_data_yaml=True)