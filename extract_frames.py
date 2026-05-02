import cv2
import os

def extract_frames(video_dir, output_dir, step=10):
    os.makedirs(output_dir, exist_ok=True)
    for video in os.listdir(video_dir):
        if video.endswith(".mp4"):
            cap = cv2.VideoCapture(os.path.join(video_dir, video))
            frame_id = 0
            saved = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % step == 0:
                    out_path = os.path.join(output_dir, f"{video}_{saved}.jpg")
                    cv2.imwrite(out_path, frame)
                    saved += 1
                frame_id += 1
            cap.release()

# Paths (do not change unless your folders are different)
extract_frames(r"D:\FFPP\data\original_sequences\youtube\c23\videos", r"D:\FFPP\frames\real")
extract_frames(r"D:\FFPP\data\manipulated_sequences\Deepfakes\c23\videos", r"D:\FFPP\frames\fake")
