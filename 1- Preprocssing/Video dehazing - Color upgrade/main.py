import os
import sys
import glob
import shutil
import argparse
import subprocess
import torch

# Default fallback dataset directory containing dehazed videos in subfolders
DEFAULT_DEHAZED_VIDEOS_DIR = r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\Dehazed Videos"

def verify_cuda():
    """Verify that CUDA GPU is strictly available."""
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] CUDA is strictly required for this project, but PyTorch could not detect a CUDA GPU!")
    print("=" * 75)
    print("  [CUDA ENFORCED] GPU Acceleration Active")
    print(f"  Device Name : {torch.cuda.get_device_name(0)}")
    print(f"  Device Count: {torch.cuda.device_count()}")
    print("=" * 75)

def process_video_pipeline(input_video_path, output_video_path, script_dir, scale="0.5", fp16=True, dehaze=True):
    """
    Executes MSRCR dehazing and model inference on a video file using RTX Tensor Cores (fp16) 
    and feature scale=0.5 to prevent cuDNN GPU memory allocation errors.
    Writes output to output_video_path upon success.
    """
    print(f"\n" + "-" * 75)
    print(f"[PIPELINE START] Input Video : '{input_video_path}'")
    print(f"                 Output Video: '{output_video_path}'")
    print("-" * 75)

    if not os.path.exists(input_video_path):
        print(f"[ERROR] Input video does not exist: '{input_video_path}'")
        return False

    out_dir = os.path.dirname(os.path.abspath(output_video_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    base, ext = os.path.splitext(output_video_path)
    temp_output = base + "_inference_temp.mp4"

    script_path = os.path.join(script_dir, "inference_video.py")
    if not os.path.exists(script_path):
        print(f"[ERROR] Inference script not found: '{script_path}'")
        return False

    dehaze_flag_str = " --dehaze" if dehaze else ""
    fp16_flag_str = " --fp16" if fp16 else ""
    print(f"[STAGE 1/1] Running Video Pipeline ('inference_video.py' --scale {scale}{fp16_flag_str}{dehaze_flag_str})...")

    cmd = [sys.executable, script_path, "--video", input_video_path, "--output", temp_output, "--scale", str(scale)]
    if fp16:
        cmd.append("--fp16")
    if dehaze:
        cmd.append("--dehaze")

    result = subprocess.run(cmd, cwd=script_dir)

    if result.returncode == 0 and os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
        if os.path.exists(output_video_path) and os.path.abspath(temp_output) != os.path.abspath(output_video_path):
            os.remove(output_video_path)
        shutil.move(temp_output, output_video_path)
        print(f"[SUCCESS] Successfully processed video. Saved to '{output_video_path}'.")
        return True
    else:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"[WARNING] Inference failed for '{os.path.basename(input_video_path)}'. Output not written.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Video Dehazing & Color Upgrade Inference Pipeline")
    parser.add_argument("input_pos", nargs="?", default=None, help="Path to input video file or directory")
    parser.add_argument("output_pos", nargs="?", default=None, help="Path to output video file or directory")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to input video file or directory")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output video file or directory")
    parser.add_argument("--scale", type=str, default="0.5", help="Scale factor for model inference (default: 0.5)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enforce FP16 precision (default: True)")
    parser.add_argument("--dehaze", action="store_true", default=True, help="Apply MSRCR dehazing pre-processing (default: True)")
    
    args = parser.parse_args()

    verify_cuda()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_path = args.input or args.input_pos
    if not input_path:
        try:
            input_path = input("Enter path of input video file or directory: ").strip()
        except EOFError:
            input_path = ""

    if input_path:
        input_path = input_path.strip("'\"")
        if not os.path.isabs(input_path):
            input_path = os.path.abspath(input_path)

    if not input_path or not os.path.exists(input_path):
        if not input_path and os.path.exists(DEFAULT_DEHAZED_VIDEOS_DIR):
            print(f"[INFO] Using default dataset directory: '{DEFAULT_DEHAZED_VIDEOS_DIR}'")
            input_path = DEFAULT_DEHAZED_VIDEOS_DIR
        else:
            print(f"[ERROR] Specified input path does not exist or is invalid: '{input_path}'")
            sys.exit(1)

    output_path = args.output or args.output_pos
    if not output_path:
        try:
            output_path = input("Enter path of output video file or directory (press Enter to overwrite in-place): ").strip()
        except EOFError:
            output_path = ""

    if output_path:
        output_path = output_path.strip("'\"")
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(output_path)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

    # Case 1: Single file processing
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(video_extensions):
            print(f"[ERROR] Specified input file is not a supported video file: '{input_path}'")
            sys.exit(1)

        target_output = output_path
        if not target_output:
            # Default to in-place overwrite if no output specified
            target_output = input_path
        elif os.path.isdir(target_output) or target_output.endswith(("\\", "/")):
            target_output = os.path.join(target_output, os.path.basename(input_path))
        elif not os.path.splitext(target_output)[1]:
            # Directory path without trailing slash
            target_output = os.path.join(target_output, os.path.basename(input_path))

        process_video_pipeline(input_path, target_output, script_dir, scale=args.scale, fp16=args.fp16, dehaze=args.dehaze)

    # Case 2: Directory processing
    elif os.path.isdir(input_path):
        video_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(video_extensions) and not file.endswith("_temp.mp4"):
                    video_files.append(os.path.join(root, file))

        video_files = sorted(list(set(video_files)))

        if not video_files:
            print(f"[INFO] No video files found in '{input_path}'.")
            return

        print(f"Found {len(video_files)} video file(s) in '{input_path}'.")

        for vid_path in video_files:
            rel_path = os.path.relpath(vid_path, input_path)
            if output_path:
                target_output = os.path.join(output_path, rel_path)
            else:
                target_output = vid_path  # In-place overwrite if no output specified

            process_video_pipeline(vid_path, target_output, script_dir, scale=args.scale, fp16=args.fp16, dehaze=args.dehaze)

    print("\n" + "=" * 75)
    print("  ALL STAGES COMPLETED: VIDEO INFERENCE FINISHED")
    print("=" * 75)

if __name__ == "__main__":
    main()
