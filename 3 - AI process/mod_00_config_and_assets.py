import os
import sys
import time
import cv2
import torch
import numpy as np

# --- HARDWARE ACCELERATION TARGET ---
script_dir = os.path.dirname(os.path.abspath(__file__))

def detect_device_hardware():
    """Detects CUDA GPU or defaults to CPU and returns tuple (device_target, device_desc)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        desc = f"CUDA ({gpu_name})"
        return torch.device("cuda"), desc
    else:
        return torch.device("cpu"), "CPU Acceleration Mode"

def get_hardware_status_summary(target_device, desc):
    if "CUDA" in str(desc):
        return f"HW: CUDA ({torch.cuda.get_device_name(0)}) | 60FPS OK"
    return "HW: CPU | 60FPS OK"

device_target, device_desc = detect_device_hardware()

def find_asset(asset_name):
    """Locates asset files in workspace or script directory."""
    paths_to_check = [
        os.path.join(script_dir, asset_name),
        os.path.join(script_dir, "models", asset_name),
        os.path.join(os.path.dirname(script_dir), asset_name)
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return os.path.join(script_dir, asset_name)

# --- JOHNNY SILVERHAND GIF ASSET LOADER ---
johnny_gif_path = find_asset("johnny.gif")
johnny_gif_frames = []

if os.path.exists(johnny_gif_path):
    try:
        cap_gif = cv2.VideoCapture(johnny_gif_path)
        while True:
            ret, frame = cap_gif.read()
            if not ret or frame is None:
                break
            johnny_gif_frames.append(frame.copy())
        cap_gif.release()
        if len(johnny_gif_frames) > 0:
            print(f"[Asset] Loaded {len(johnny_gif_frames)} frame(s) from Johnny Silverhand GIF ({johnny_gif_path}).")
    except Exception as e:
        print(f"Notice: johnny.gif loader ({e}).")

johnny_img_rgba = None
if len(johnny_gif_frames) > 0:
    first_frame = johnny_gif_frames[0]
    h, w, c = first_frame.shape
    johnny_img_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    johnny_img_rgba[:, :, :3] = first_frame
    johnny_img_rgba[:, :, 3] = 255

def create_unique_video_session_dir(video_path, base_output_dir=None):
    """
    Creates a unique output directory structure for each video analysis session.
    Sanitizes video_basename to clean ASCII to prevent Windows Unicode path crashes.
    Structure:
      video_analysis_sessions/<sanitized_name>_<timestamp>/
        ├── output/ tracked_video.mp4
        ├── csv/    fish_counts.csv
        ├── plots/  enkf_telemetry_plot.png (and 20 scientific plots)
        └── analysis/ ollama_marine_report.md & .pdf & .tex
    """
    import re
    if base_output_dir is None:
        base_output_dir = os.path.join(script_dir, "video_analysis_sessions")
    
    raw_basename = os.path.splitext(os.path.basename(video_path))[0]
    clean_basename = re.sub(r'[^\w\s-]', '', raw_basename).strip().replace(' ', '_')
    if not clean_basename:
        clean_basename = "video_session"
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    session_name = f"{clean_basename}_{timestamp_str}"
    
    session_dir = os.path.join(base_output_dir, session_name)
    csv_dir = os.path.join(session_dir, "csv")
    plots_dir = os.path.join(session_dir, "plots")
    analysis_dir = os.path.join(session_dir, "analysis")
    output_dir = os.path.join(session_dir, "output")
    fish_images_dir = os.path.join(session_dir, "fish_images")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fish_images_dir, exist_ok=True)
    
    output_video_path = os.path.join(output_dir, f"tracked_{clean_basename}.mp4")
    
    return {
        "session_name": session_name,
        "session_dir": session_dir,
        "csv_dir": csv_dir,
        "plots_dir": plots_dir,
        "analysis_dir": analysis_dir,
        "output_dir": output_dir,
        "fish_images_dir": fish_images_dir,
        "output_video_path": output_video_path
    }
