import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor
import shutil
import glob
import json
import random
import yaml
from tqdm import tqdm

# Import the label selection GUI function
from run_gui import get_label


class SAM2VideoRunner:
    def __init__(self, frames_folder, model_id="facebook/sam2-hiera-large", device="cpu"):
        self.frames_folder = frames_folder

        self.seed = None
        number = ''.join(filter(str.isdigit, os.path.basename(frames_folder)))

        
        self.save_dir = os.path.basename(frames_folder).split("_f")[0] + "_masks"
        self.dataset_dir = os.path.join(os.path.dirname(frames_folder), "yolov8_dataset_" + number)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA is available. Using GPU.")
        else:
            self.device = torch.device("cpu")
        print(f"using device: {self.device}")

        # self.device = torch.device("cpu")
        
        if self.device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
        self.predictor = SAM2VideoPredictor.from_pretrained(
            "facebook/sam2-hiera-small",
            device=self.device
        )
        self.state = None
        
        self.current_frame = None
        self.labels = {}
        self.object_id_to_label = {}  # Mapping from obj_id to selected label
        self.points = None
        self.box = None 
        self.vis_frame_stride = 5
        self.annotated_frame = None 
        self.video_segments = {}

        self.frame_names = [
            p for p in os.listdir(frames_folder)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))        
        
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
                    cv2.circle(frame, (x, y), 4, (0, 200, 255), -1)
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
                        self.object_id_to_label[obj_id] = selected_label
                        print(f"Added object {obj_id} with label '{selected_label}' and {len(points)} points.")
                        obj_id = len(self.labels) + 1
                    else:
                        print("No points clicked for this object.")
                    break
                elif key == 27:  # ESC key
                    # exit if no points were added for this object
                    if not self.labels:
                        print("No objects were selected. Exiting.")
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                        return 
                        
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    print("Finished adding objects.")
                    print(f"Labels: {self.labels}")
                    return
            
    def propagate(self, save = True, visualize=False):
        if self.labels is None or len(self.labels) == 0:
            print("No labels defined. Please add prompts first.")
            return
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.state):
                frame_path = os.path.join(self.frames_folder, self.frame_names[out_frame_idx])
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                self.video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                if save: 
                    pass
                    # if not os.path.exists(self.save_dir):
                    #     os.makedirs(self.save_dir)
                    # for obj_id, mask in self.video_segments[out_frame_idx].items():
                    #     np.save(os.path.join(self.save_dir, f"mask_obj{obj_id}_{out_frame_idx:04d}.npy"), mask)
                
                if visualize:
                    for obj_id, mask in self.video_segments[out_frame_idx].items():
                        frame = self.show_mask_opencv(mask, frame, obj_id=obj_id, random_color=False, alpha=0.5)
                    cv2.imshow("Propagation", frame)
                    cv2.waitKey(1)

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
            cv2.waitKey(200)
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
                # save the frame with boxes
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                cv2.imwrite(os.path.join(self.save_dir, f"bbox_{out_frame_idx:04d}.jpg"), frame)
                cv2.waitKey(100)
        cv2.destroyAllWindows()
        return boxes

    def export_yolov8_dataset(self, train_ratio=0.8, write_data_yaml=True):
        import shutil
        root_dir = self.dataset_dir
        images_train_dir = os.path.join(root_dir, "images", "train")
        images_val_dir = os.path.join(root_dir, "images", "val")
        labels_train_dir = os.path.join(root_dir, "labels", "train")
        labels_val_dir = os.path.join(root_dir, "labels", "val")

        os.makedirs(images_train_dir, exist_ok=True)
        os.makedirs(images_val_dir, exist_ok=True)
        os.makedirs(labels_train_dir, exist_ok=True)
        os.makedirs(labels_val_dir, exist_ok=True)

        # Use a temporary directory for label files
        labels_temp_dir = os.path.join(root_dir, "labels_temp")
        os.makedirs(labels_temp_dir, exist_ok=True)

        # Map object names to class ids (0-based) using sorted labels to preserve original IDs from GUI
        sorted_labels = sorted(self.labels.items(), key=lambda x: x[1])
        print("sorted labels: ", sorted_labels)
        class_name_to_id = {name: idx for name, idx in sorted_labels}
        print("class_name_to_id (from GUI mapping): ", class_name_to_id)

        # Get bounding boxes
        boxes = self.create_bounding_boxes(visualize=False)

        # Prepare list of frames with bounding boxes
        frames_with_boxes = set()
        for (obj_id, frame_idx) in boxes.keys():
            frames_with_boxes.add(frame_idx)

        frame_indices = sorted(list(frames_with_boxes))

        # Step 1: Write all label files and collect labeled frames
        labeled_frames = []
        for frame_idx in frame_indices:
            frame_name = self.frame_names[frame_idx]
            frame_path = os.path.join(self.frames_folder, frame_name)
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Warning: Failed to read image {frame_path}, skipping.")
                continue
            h, w = img.shape[:2]
            # Write label file in a temp location
            label_file_name = os.path.splitext(frame_name)[0] + ".txt"
            label_file_path = os.path.join(labels_temp_dir, label_file_name)
            os.makedirs(os.path.dirname(label_file_path), exist_ok=True)

            # Collect all boxes for this frame
            lines = []
            for (obj_id, f_idx), (x_min, y_min, x_max, y_max) in boxes.items():
                if f_idx != frame_idx:
                    continue
                # Use object_id_to_label to get the correct label for this obj_id
                if obj_id not in self.object_id_to_label:
                    print(f"Warning: obj_id {obj_id} not found in object_id_to_label mapping, skipping.")
                    continue
                label = self.object_id_to_label[obj_id]
                if label not in class_name_to_id:
                    print(f"Warning: label '{label}' for obj_id {obj_id} not found in class_name_to_id mapping, skipping.")
                    continue
                class_id = class_name_to_id[label]

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
            # Collect this as a labeled frame (image path, label path, frame name)
            labeled_frames.append({
                "frame_idx": frame_idx,
                "frame_name": frame_name,
                "image_path": frame_path,
                "label_path": label_file_path
            })

        # Step 2: Shuffle and split into train/val
        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(labeled_frames)
        num_train = int(len(labeled_frames) * train_ratio)
        train_frames = labeled_frames[:num_train]
        val_frames = labeled_frames[num_train:]

        # Step 3: Move/copy images and label files into train/val folders
        for split_frames, img_dst_dir, label_dst_dir in [
            (train_frames, images_train_dir, labels_train_dir),
            (val_frames, images_val_dir, labels_val_dir)]:
            for item in split_frames:
                # Copy image
                dst_img_path = os.path.join(img_dst_dir, item["frame_name"])
                shutil.copyfile(item["image_path"], dst_img_path)
                # Copy label (from temp directory)
                label_file_name = os.path.splitext(item["frame_name"])[0] + ".txt"
                dst_label_path = os.path.join(label_dst_dir, label_file_name)
                shutil.copyfile(item["label_path"], dst_label_path)

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

        # Clean up the temporary labels directory
        try:
            shutil.rmtree(labels_temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temp label directory {labels_temp_dir}: {e}")

        print(f"YOLOv8 dataset exported to {root_dir} with train/val split {train_ratio}")
        
    def collate_datasets(self, base_dir, target_dir, seed=42, keep_splits=False):
        image_exts = (".jpg", ".jpeg", ".png")

        # discover candidate dataset folders inside base_dir
        candidate_dirs = []
        if not os.path.isdir(base_dir):
            print(f"collate_datasets: base_dir does not exist: {base_dir}")
            return 0
        for entry in os.listdir(base_dir):
            p = os.path.join(base_dir, entry)
            if not os.path.isdir(p):
                continue
            # heuristics: contains images/ or images/train
            if os.path.isdir(os.path.join(p, "images")) or os.path.isdir(os.path.join(p, "images", "train")):
                candidate_dirs.append(p)
        if not candidate_dirs:
            print(f"collate_datasets: no dataset-like folders found in {base_dir}")
            return 0

        # create target directories
        if keep_splits:
            images_train_dir = os.path.join(target_dir, "images", "train")
            images_val_dir = os.path.join(target_dir, "images", "val")
            labels_train_dir = os.path.join(target_dir, "labels", "train")
            labels_val_dir = os.path.join(target_dir, "labels", "val")
            os.makedirs(images_train_dir, exist_ok=True)
            os.makedirs(images_val_dir, exist_ok=True)
            os.makedirs(labels_train_dir, exist_ok=True)
            os.makedirs(labels_val_dir, exist_ok=True)
        else:
            images_dir = os.path.join(target_dir, "images")
            labels_dir = os.path.join(target_dir, "labels")
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

        # gather image/label pairs
        entries = []
        seen = set()
        for d in candidate_dirs:
            # check several possible image root locations
            for images_root in [os.path.join(d, "images", "train"), os.path.join(d, "images", "val"), os.path.join(d, "images")]:
                if not os.path.isdir(images_root):
                    continue
                for fname in os.listdir(images_root):
                    if not any(fname.lower().endswith(ext) for ext in image_exts):
                        continue
                    img_path = os.path.join(images_root, fname)
                    if img_path in seen:
                        continue
                    seen.add(img_path)
                    base = os.path.splitext(fname)[0]
                    label_path = None
                    # search for corresponding label in standard places
                    for lbl_root in [os.path.join(d, "labels", "train"), os.path.join(d, "labels", "val"), os.path.join(d, "labels")]:
                        candidate_lbl = os.path.join(lbl_root, base + ".txt")
                        if os.path.exists(candidate_lbl):
                            label_path = candidate_lbl
                            break
                    entries.append({"image": img_path, "label": label_path, "source": d})

        # shuffle
        if seed is not None:
            random.seed(seed)
        random.shuffle(entries)

        # copy and rename
        manifest = []
        idx = 0
        for e in entries:
            idx += 1
            src_img = e["image"]
            src_lbl = e.get("label")
            ext = os.path.splitext(src_img)[1].lower()
            new_basename = f"{idx:06d}"
            new_img_name = new_basename + ext

            if keep_splits:
                # try infer original split from path
                low = src_img.lower()
                if os.sep + "train" + os.sep in low:
                    dst_img_dir = images_train_dir
                    dst_lbl_dir = labels_train_dir
                elif os.sep + "val" + os.sep in low:
                    dst_img_dir = images_val_dir
                    dst_lbl_dir = labels_val_dir
                else:
                    # default to train if unknown
                    print(f"Warning: cannot infer train/val split for {src_img}, defaulting to train for all datapoints")
                    dst_img_dir = images_train_dir
                    dst_lbl_dir = labels_train_dir
            else:
                dst_img_dir = images_dir
                dst_lbl_dir = labels_dir

            dst_img_path = os.path.join(dst_img_dir, new_img_name)
            dst_lbl_path = os.path.join(dst_lbl_dir, new_basename + ".txt")

            try:
                shutil.copyfile(src_img, dst_img_path)
            except Exception as exc:
                print(f"Failed to copy image {src_img} -> {dst_img_path}: {exc}")
                continue

            if src_lbl and os.path.exists(src_lbl):
                shutil.copyfile(src_lbl, dst_lbl_path)
            else:
                # create empty label file if none exists so YOLO dataset loaders don't error
                print(f"Warning: no label file for image {src_img}, creating empty label file")
                open(dst_lbl_path, "w").close()

            manifest.append({
                "source": e["source"],
                "original_image": src_img,
                "original_label": src_lbl,
                "new_image": dst_img_path,
                "new_label": dst_lbl_path,
            })

        # write manifest for debug / reproducibility
        os.makedirs(target_dir, exist_ok=True)
        manifest_path = os.path.join(target_dir, "collate_manifest.json")
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

        print(f"Collated {len(manifest)} images into {target_dir}")
        return len(manifest)

    @staticmethod
    def convert_npy_to_jpg(npy_folder, jpg_folder):
        if not os.path.exists(jpg_folder):
            os.makedirs(jpg_folder)
        
        npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))

        pbar = tqdm(total=len(npy_files), desc="Converting .npy to .jpg")

        for filename in os.listdir(npy_folder):
            if filename.lower().endswith(".npy"):
                npy_path = os.path.join(npy_folder, filename)
                # get only the number part of the filename
                frame_number = filename.split("_")[-1].split(".")[0]    
                jpg_filename = frame_number + ".jpg"
                
                jpg_path = os.path.join(jpg_folder, jpg_filename)

                img = np.load(npy_path)

                # crop 640x640 region from center if larger
                h, w = img.shape[:2]
                if h > 640 and w > 640:
                    start_y = (h - 640) // 2
                    start_x = (w - 640) // 2
                    img = img[start_y:start_y + 640, start_x:start_x + 640]

                if img.ndim != 3 or img.shape[2] !=3:
                    print(f"Skipping {npy_path}: not a 3-channel image.")
                    continue
                
                cv2.imwrite(jpg_path, img) 
                pbar.update(1)
        pbar.close()

        # # add progress bar
        # pbar = tqdm(total=len(npy_files), desc="Converting .npy to .jpg")
        # for npy_file in npy_files:
        #     mask = np.load(npy_file)
        #     # Normalize mask to 0-255
        #     mask_norm = (mask * 255).astype(np.uint8)
        #     # convert bgr to rgb
        #     mask_norm = cv2.cvtColor(mask_norm, cv2.COLOR_BGR2RGB)
        #     jpg_file = os.path.join(jpg_folder, os.path.basename(npy_file).replace(".npy", ".jpg"))
        #     cv2.imwrite(jpg_file, mask_norm)
        #     pbar.update(1)
        # pbar.close()
        print(f"Converted {len(npy_files)} .npy files to .jpg in {jpg_folder}")


if __name__ == "__main__":
    # covert to jpg
    npy_folder = "/home/matt/projects/NatureNeuromorphicComputingForRobots/inference/calibrate_data_2025_10_24_data/sequence_000003/flir_23604512/frame"
    jpg_folder = "/home/matt/projects/NatureNeuromorphicComputingForRobots/inference/calibrate_data_2025_10_24_data/sequence_000003/flir_23604512/frame_jpg"
    SAM2VideoRunner.convert_npy_to_jpg(npy_folder, jpg_folder)

    frames_path = jpg_folder
    runner = SAM2VideoRunner(frames_path)
    
    runner.init_video()

    # Point prompt with multiple points and multiple objects
    runner.add_click_prompt()

    # Propagate
    runner.propagate(save=True, visualize=True)
    
    # Visualize masks on frames at a given stride
    # runner.visualize_masks(stride=1)
    
    # show bounding boxes
    # boxes = runner.create_bounding_boxes(visualize=True)
    
    # Export to YOLOv8 format
    # runner.seed = 42  
    # runner.export_yolov8_dataset(train_ratio=0.8, write_data_yaml=True)
    
    # Collate multiple datasets
    # base_dir = "/Users/itamrakar/Documents/Projects/fast_object_detection_dataset/data"
    # target_dir = "/Users/itamrakar/Documents/Projects/fast_object_detection_dataset/collated_dataset"
    # runner.collate_datasets(base_dir, target_dir, seed=42, keep_splits=True)
    
    