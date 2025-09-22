import cv2
import os
import tqdm

def video_to_frames(video_path="./data/IMG_3550.MOV", output_folder=None, frame_sampling_rate=1, size=640):
    if output_folder is None:
        # just get the name of the video file without extension and create a folder with that name
        output_folder = os.path.join(os.path.dirname(video_path), os.path.splitext(os.path.basename(video_path))[0] + "_frames")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    frame_idx = 0
    saved_frame_idx = 0

    # Get total number of frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=total_frames, desc="Extracting frames")
    while True:
        pbar.update(1)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_sampling_rate == 0:
            filename = os.path.join(output_folder, f"{saved_frame_idx:06d}.jpg")
            # resize frame 
            # if frame is not square, resize and pad to make it square using opencv
            h, w, _ = frame.shape
            if h != w:
                # crop and pad to make it square and without black borders
                print(f"Frame {frame_idx} is not square: {w}x{h}, cropping and resizing. Objects may be cut off.")
                dim = min(h, w)
                top = (h - dim) // 2
                left = (w - dim) // 2
                frame = frame[top:top+dim, left:left+dim]
                frame = cv2.resize(frame, (size, size))
            else:
                frame = cv2.resize(frame, (size, size))
            cv2.imwrite(filename, frame)
            saved_frame_idx += 1

        frame_idx += 1
    pbar.close()

    cap.release()
    print(f"Extracted {saved_frame_idx} frames to {output_folder}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video_path", nargs='?', default="./data/IMG_3550.MOV", type=str, help="Path to the video file")
    parser.add_argument("output_folder", nargs='?', default=None, type=str, help="Folder to save extracted frames")
    parser.add_argument("--frame_sampling_rate", type=int, default=50, help="Save every Nth frame (default=1)")
    parser.add_argument("--size", type=int, default=640, help="Resize frames to this size (default=640)")

    args = parser.parse_args()

    video_to_frames(args.video_path, args.output_folder, args.frame_sampling_rate, args.size)