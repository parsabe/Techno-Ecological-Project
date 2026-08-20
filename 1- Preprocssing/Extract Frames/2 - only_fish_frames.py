import os
import glob
import shutil
import cv2
import torch
from ultralytics import YOLO

# Directory & Model Paths
ALL_FRAMES_DIR = r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\All Frames"
FISH_FRAMES_DIR = r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\Extracted fish objects in frames"
MODEL_PATH = r"C:\Users\parsa\Desktop\Code\0 - Preprocessing\0 - Source Codes\Extract Frames\yolov8n.pt"

def verify_cuda():
    """Verify that CUDA GPU is strictly available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is strictly required for this project, but PyTorch could not detect a CUDA GPU!")
    print(f"[CUDA ENFORCED] Using GPU: {torch.cuda.get_device_name(0)}")

def extract_only_fish_frames(all_frames_dir=ALL_FRAMES_DIR, output_dir=FISH_FRAMES_DIR, model_path=MODEL_PATH):
    """
    Detect fish objects in all frames extracted in 'All Frames' directory
    using YOLOv8 (CUDA accelerated) and copy matching frames to 
    'Extracted fish objects in frames' directory without removing originals.
    """
    verify_cuda()
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model file not found at: {model_path}")
        
    print(f"Loading YOLO model on CUDA GPU: '{model_path}'")
    model = YOLO(model_path)
    model.to('cuda')
    
    # Identify class IDs corresponding to 'fish' if available, otherwise check detected objects
    fish_class_ids = [cls_id for cls_id, name in model.names.items() if 'fish' in str(name).lower()]
    if not fish_class_ids:
        print(f"Note: 'fish' class name not explicitly present in model class mapping: {model.names}")
        print("Defaulting to evaluating all detected objects in model output.")
        target_ids = list(model.names.keys())
    else:
        print(f"Fish class ID(s) detected in YOLO model: {fish_class_ids}")
        target_ids = fish_class_ids

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [
        os.path.join(all_frames_dir, f) for f in os.listdir(all_frames_dir)
        if f.lower().endswith(image_extensions)
    ] if os.path.exists(all_frames_dir) else []
    
    print(f"Scanning {len(image_files)} image(s) from '{all_frames_dir}' for fish detection...")
    
    fish_count = 0
    batch_size = 32
    for i in range(0, len(image_files), batch_size):
        batch_paths = image_files[i:i + batch_size]
        # Run YOLO inference strictly on CUDA GPU
        results = model.predict(source=batch_paths, device='cuda', verbose=False)
        
        for img_path, result in zip(batch_paths, results):
            has_fish = False
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    if cls_id in target_ids:
                        has_fish = True
                        break
                        
            if has_fish:
                dest_path = os.path.join(output_dir, os.path.basename(img_path))
                shutil.copy2(img_path, dest_path)
                fish_count += 1
                
    print(f"Fish frame extraction complete! {fish_count} frame(s) with fish copied to '{output_dir}'.")

if __name__ == "__main__":
    extract_only_fish_frames()
