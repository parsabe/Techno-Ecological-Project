import os
import glob
import cv2
import torch

# Directory Paths
VIDEO_DIR = r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\Dehazed Videos"
ALL_FRAMES_DIR = r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\All Frames"

def verify_cuda():
    """Verify that CUDA GPU is strictly available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is strictly required for this project, but PyTorch could not detect a CUDA GPU!")
    print(f"[CUDA ENFORCED] Using GPU: {torch.cuda.get_device_name(0)}")

def extract_all_frames(video_dir=VIDEO_DIR, output_dir=ALL_FRAMES_DIR):
    """Extract all video frames recursively and save to All Frames directory."""
    verify_cuda()
    os.makedirs(output_dir, exist_ok=True)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    
    video_files = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))
                
    print(f"Found {len(video_files)} video(s) in '{video_dir}'.")
    
    total_frames_extracted = 0
    for video_path in video_files:
        rel_path = os.path.relpath(video_path, video_dir)
        folder_prefix = os.path.splitext(rel_path)[0].replace(os.sep, "_").replace(" ", "_")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file '{video_path}'")
            continue
            
        frame_idx = 0
        video_extracted = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_filename = f"{folder_prefix}_frame_{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, frame_filename)
            
            cv2.imwrite(out_path, frame)
            frame_idx += 1
            video_extracted += 1
            
        cap.release()
        print(f"Extracted {video_extracted} frames from '{os.path.basename(video_path)}'")
        total_frames_extracted += video_extracted
        
    print(f"Extraction complete! Total {total_frames_extracted} frames saved in '{output_dir}'.")

if __name__ == "__main__":
    extract_all_frames()
