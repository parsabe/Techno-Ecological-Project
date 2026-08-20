import os
import glob
import cv2
import torch

# Directories to resize & Target Resolution
ALL_FRAMES_DIR = r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\All Frames"
FISH_FRAMES_DIR = r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\Extracted fish objects in frames"
TARGET_SIZE = (512, 512)

def verify_cuda():
    """Verify that CUDA GPU is strictly available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is strictly required for this project, but PyTorch could not detect a CUDA GPU!")
    print(f"[CUDA ENFORCED] Using GPU: {torch.cuda.get_device_name(0)}")

def resize_directory_images(directory, target_size=TARGET_SIZE):
    """Resize all images in the given directory to target_size (512x512) using CUDA GPU tensors and overwrite."""
    verify_cuda()
    if not os.path.exists(directory):
        print(f"Directory does not exist, skipping: '{directory}'")
        return

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print(f"No image files found to resize in '{directory}'.")
        return

    print(f"Resizing {len(image_files)} image(s) in '{directory}' to {target_size[0]}x{target_size[1]} using CUDA GPU...")

    device = torch.device('cuda')
    resized_count = 0
    
    for img_path in image_files:
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Convert BGR (OpenCV) to RGB PyTorch Tensor on CUDA GPU
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
            
            # CUDA accelerated bilinear interpolation resize
            resized_tensor = torch.nn.functional.interpolate(
                tensor, size=target_size, mode='bilinear', align_corners=False
            )
            
            # Convert back to uint8 BGR array and overwrite existing file
            resized_numpy = resized_tensor.squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
            resized_bgr = cv2.cvtColor(resized_numpy, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(img_path, resized_bgr)
            resized_count += 1
        except Exception as e:
            print(f"Error resizing image '{img_path}': {e}")

    print(f"Successfully resized and overwrote {resized_count} image(s) in '{directory}'.")

def resize_all():
    """Perform resize operation on both 'All Frames' and 'Extracted fish objects in frames' directories."""
    verify_cuda()
    print("--- Starting Image Resizing Stage (512x512, CUDA Accelerated) ---")
    resize_directory_images(ALL_FRAMES_DIR)
    resize_directory_images(FISH_FRAMES_DIR)
    print("--- Image Resizing Stage Completed ---")

if __name__ == "__main__":
    resize_all()
