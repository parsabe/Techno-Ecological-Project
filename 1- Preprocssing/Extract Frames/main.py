import os
import sys
import importlib
import torch

def verify_cuda():
    """Verify CUDA availability and print GPU device information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is strictly required for this project, but PyTorch could not detect a CUDA GPU!")
    print("=" * 70)
    print("  [CUDA ENFORCED] GPU Acceleration Active")
    print(f"  Device Name : {torch.cuda.get_device_name(0)}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    print("=" * 70)

def main():
    # Enforce CUDA first
    verify_cuda()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    print("\n[STAGE 1/3] Extracting all frames from video datasets...")
    step1 = importlib.import_module("1 - all_frames")
    step1.extract_all_frames()

    print("\n[STAGE 2/3] Detecting and extracting fish frames using YOLOv8 (CUDA)...")
    step2 = importlib.import_module("2 - only_fish_frames")
    step2.extract_only_fish_frames()

    print("\n[STAGE 3/3] Resizing images to 512x512 and overwriting in both directories (CUDA Tensors)...")
    step3 = importlib.import_module("3 - resize")
    step3.resize_all()

    print("\n" + "=" * 70)
    print("  SUCCESS: ALL VIDEO FRAME & FISH EXTRACTION STAGES COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
