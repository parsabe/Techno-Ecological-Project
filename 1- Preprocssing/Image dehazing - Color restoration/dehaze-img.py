import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from pathlib import Path

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("[CRITICAL] CUDA is not available. Check your PyTorch installation.")

# Set device
device = torch.device("cuda:0")
print(f"[INFO] Using GPU device: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. CUDA-ACCELERATED MSRCR ENGINE
# ==========================================
@torch.no_grad() # Extremely important: Prevents memory leaks & boosts speed during inference
def apply_msrcr_cuda(img_np):
    # Convert numpy to GPU tensor [C, H, W], add 1.0 to avoid log(0)
    # Using float32 for maximum CUDA core efficiency
    img_tensor = torch.from_numpy(img_np).to(device, dtype=torch.float32) + 1.0
    img_tensor = img_tensor.permute(2, 0, 1) # OpenCV [H,W,C] -> PyTorch [C,H,W]
    
    sigmas = [15, 80, 250]
    retinex = torch.zeros_like(img_tensor, device=device)
    
    for s in sigmas:
        # Calculate kernel size (6 * sigma + 1, must be odd)
        k_size = int(6 * s + 1)
        k_size = k_size if k_size % 2 != 0 else k_size + 1
        
        # Use torchvision's highly optimized CUDA blur
        blur = TF.gaussian_blur(img_tensor, kernel_size=[k_size, k_size], sigma=[s, s])
        
        # Retinex formula (add small epsilon 1e-6 to prevent log(0) anomalies)
        retinex += torch.log10(img_tensor) - torch.log10(blur + 1e-6)
        
    retinex /= len(sigmas)
    
    # Color restoration on GPU
    sum_channels = torch.sum(img_tensor, dim=0, keepdim=True) + 1.0
    color_rest = torch.log10(125.0 * img_tensor) - torch.log10(sum_channels)
    
    msrcr = retinex * color_rest
    
    # Min-Max Normalization to [0, 255]
    min_val = msrcr.min()
    max_val = msrcr.max()
    msrcr = 255.0 * (msrcr - min_val) / (max_val - min_val + 1e-6) # 1e-6 prevents div by zero
    
    # Clamp safety, convert to uint8, pull back to CPU [H, W, C] for YOLO/OpenCV
    enhanced_np = msrcr.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    
    return enhanced_np

# ==========================================
# 2. PIPELINE CONFIGURATION
# ==========================================
# Use pathlib for smart, relative pathing based on where the script is run
BASE_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
INPUT_IMAGE_PATH = BASE_DIR / "main2.png"
OUTPUT_FOLDER = BASE_DIR / "image-dehazed"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Load YOLOv8 model directly to GPU
print("[INFO] Loading YOLOv8 model to CUDA...")
model = YOLO('yolov8n.pt')
model.to(device)

# ==========================================
# 3. SINGLE IMAGE INFERENCE
# ==========================================
if __name__ == "__main__":
    print(f"[INFO] Looking for target image at: {INPUT_IMAGE_PATH}")

    if not INPUT_IMAGE_PATH.exists():
        print(f"[ERROR] Image not found! Please ensure 'main2.png' is in {BASE_DIR}")
    else:
        img = cv2.imread(str(INPUT_IMAGE_PATH))
        
        if img is None:
            print("[ERROR] Failed to load image. File may be corrupted.")
        else:
            # Step A: GPU-Accelerated Enhancement
            print("[INFO] Applying MSRCR enhancement (CUDA)...")
            enhanced_img = apply_msrcr_cuda(img)

            # Step B: YOLO Inference
            print("[INFO] Running YOLOv8 detection (CUDA)...")
            # Call the model directly, passing device ensures it stays on GPU
            results = model(enhanced_img, device=device, conf=0.25, verbose=False)

            # Step C: Visualization & Save
            result_img = results[0].plot() 
            
            save_path = OUTPUT_FOLDER / "det_main2.png"
            cv2.imwrite(str(save_path), result_img)

            print(f"[SUCCESS] Processed and saved to: {save_path}")

    print("[INFO] Pipeline Complete.")