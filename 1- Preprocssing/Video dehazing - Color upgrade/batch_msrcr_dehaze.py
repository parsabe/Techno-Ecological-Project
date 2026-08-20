import os
import sys
import cv2
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")

# --- Custom Video Format Error ---
class VideoFormatError(Exception):
    """Exception raised when a video format is unsupported, unreadable, or missing container metadata."""
    pass


# --- Device Configuration ---
if not torch.cuda.is_available():
    raise RuntimeError("[ERROR] CUDA is not available! CUDA is strictly required for performance and execution.")

device = torch.device("cuda")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print(f"[INFO] Using CUDA device: {torch.cuda.get_device_name(0)}")


# --- Separable Gaussian Blur on GPU ---
def gaussian_blur_gpu(tensor, sigma):
    """
    Computes 2D Gaussian blur using separable 1D convolutions on PyTorch GPU tensors.
    """
    target_sigma = 4.0
    h, w = tensor.shape[2], tensor.shape[3]

    if sigma > target_sigma:
        scale = sigma / target_sigma
        h_down = max(16, int(h / scale))
        w_down = max(16, int(w / scale))
        tensor_down = F.interpolate(tensor, size=(h_down, w_down), mode='bilinear', align_corners=False)
        actual_sigma = target_sigma
    else:
        tensor_down = tensor
        actual_sigma = sigma

    ksize = int(6 * actual_sigma) | 1
    x = torch.arange(ksize, dtype=torch.float32, device=tensor_down.device)
    mean = (ksize - 1) / 2.0
    variance = actual_sigma ** 2
    kernel_1d = torch.exp(-((x - mean) ** 2) / (2.0 * variance))
    kernel_1d = kernel_1d / torch.clamp(kernel_1d.sum(), min=1e-8)

    channels = tensor_down.shape[1]
    kernel_h = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
    kernel_w = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)

    pad = ksize // 2
    h_down_curr, w_down_curr = tensor_down.shape[2], tensor_down.shape[3]

    if pad >= h_down_curr or pad >= w_down_curr:
        pad = min(h_down_curr - 1, w_down_curr - 1, pad)
        if pad < 0:
            pad = 0
            ksize = 1
        else:
            ksize = 2 * pad + 1

        if ksize > 0:
            x = torch.arange(ksize, dtype=torch.float32, device=tensor_down.device)
            mean = (ksize - 1) / 2.0
            kernel_1d = torch.exp(-((x - mean) ** 2) / (2.0 * variance))
            kernel_1d = kernel_1d / torch.clamp(kernel_1d.sum(), min=1e-8)
            kernel_h = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
            kernel_w = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)
        else:
            return tensor_down

    padded = F.pad(tensor_down, (pad, pad, pad, pad), mode='replicate')
    blurred = F.conv2d(padded, kernel_h, groups=channels)
    blurred = F.conv2d(blurred, kernel_w, groups=channels)

    if sigma > target_sigma:
        blurred = F.interpolate(blurred, size=(h, w), mode='bilinear', align_corners=False)

    return blurred


# --- MSRCR Dehazing Implementation ---
def apply_msrcr_cuda(img, device_obj=device):
    """
    Applies Multi-Scale Retinex with Color Restoration (MSRCR) on CUDA GPU.
    """
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device_obj, dtype=torch.float32)
    img_tensor = img_tensor + 1.0

    sigmas = [15, 80, 250]
    retinex = torch.zeros_like(img_tensor, dtype=torch.float32)

    for s in sigmas:
        blur = gaussian_blur_gpu(img_tensor, s)
        log_img = torch.log10(torch.clamp(img_tensor, min=1e-5))
        log_blur = torch.log10(torch.clamp(blur, min=1e-5))
        retinex += log_img - log_blur

    retinex /= len(sigmas)

    channel_sum = torch.sum(img_tensor, dim=1, keepdim=True)
    color_rest = torch.log10(125.0 * torch.clamp(img_tensor, min=1e-5)) - torch.log10(torch.clamp(channel_sum + 1.0, min=1e-5))

    msrcr = retinex * color_rest

    min_val = msrcr.min()
    max_val = msrcr.max()
    if max_val > min_val:
        msrcr = (msrcr - min_val) / (max_val - min_val) * 255.0
    else:
        msrcr = torch.zeros_like(msrcr)

    msrcr = torch.clamp(msrcr, 0, 255)
    return msrcr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def apply_msrcr_cpu(img):
    """CPU fallback implementation for MSRCR dehazing."""
    img_float = np.float32(img) + 1.0
    sigmas = [15, 80, 250]
    retinex = np.zeros_like(img_float)
    for s in sigmas:
        blur = cv2.GaussianBlur(img_float, (0, 0), s)
        retinex += np.log10(np.clip(img_float, 1e-5, None)) - np.log10(np.clip(blur, 1e-5, None))

    retinex /= len(sigmas)
    channel_sum = np.sum(img_float, axis=2, keepdims=True)
    color_rest = np.log10(125.0 * np.clip(img_float, 1e-5, None)) - np.log10(np.clip(channel_sum + 1.0, 1e-5, None))

    msrcr = retinex * color_rest
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
    return msrcr.astype(np.uint8)


def apply_msrcr(img, device_obj=device):
    if device_obj.type != 'cuda':
        raise RuntimeError("[ERROR] CUDA device is required for MSRCR dehazing processing.")
    return apply_msrcr_cuda(img, device_obj)


# --- Dynamic Video Format Inspector & Validator ---
def validate_video(video_path):
    """
    Validates that a video file can be opened and decoded properly by OpenCV.
    Raises VideoFormatError if the format is invalid, corrupt, or raw stream without container.
    """
    if not os.path.exists(video_path):
        raise VideoFormatError(f"Video file does not exist: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    valid_exts = ['.mp4', '.avi', '.mov', '.mkv', '.h264', '.264']
    if ext not in valid_exts:
        raise VideoFormatError(
            f"Unsupported video extension '{ext}' for file '{video_path}'. "
            f"Please convert the video to standard H.264 MP4 before processing."
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoFormatError(
            f"Unable to open video container for '{os.path.basename(video_path)}'. "
            f"The video format/codec requires conversion (e.g. convert raw stream to standard H.264 MP4)."
        )

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    cap.release()

    if not ret or first_frame is None or frame_width <= 0 or frame_height <= 0:
        raise VideoFormatError(
            f"Video '{os.path.basename(video_path)}' has unreadable frames or zero resolution. "
            f"Format needs conversion (e.g., convert raw elementary stream to standard H.264 MP4)."
        )

    return {
        'width': frame_width,
        'height': frame_height,
        'fps': fps if fps > 0 else 30.0,
        'total_frames': tot_frames
    }


# --- Audio Preservation Utility ---
def transfer_audio(source_video, target_video):
    """Transfers audio track from source_video to target_video using ffmpeg."""
    temp_dir = "./temp_audio"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    temp_audio_file = os.path.join(temp_dir, "audio.mkv")
    extract_cmd = f'ffmpeg -y -i "{source_video}" -c:a copy -vn "{temp_audio_file}" -loglevel error'
    res = os.system(extract_cmd)

    if res != 0 or not os.path.exists(temp_audio_file) or os.path.getsize(temp_audio_file) == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    target_no_audio = os.path.splitext(target_video)[0] + "_noaudio.mp4"
    os.rename(target_video, target_no_audio)

    merge_cmd = f'ffmpeg -y -i "{target_no_audio}" -i "{temp_audio_file}" -c copy "{target_video}" -loglevel error'
    merge_res = os.system(merge_cmd)

    if merge_res == 0 and os.path.exists(target_video) and os.path.getsize(target_video) > 0:
        os.remove(target_no_audio)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    # Fallback to AAC transcoding if direct copy fails
    temp_audio_aac = os.path.join(temp_dir, "audio.m4a")
    os.system(f'ffmpeg -y -i "{source_video}" -c:a aac -b:a 160k -vn "{temp_audio_aac}" -loglevel error')
    os.system(f'ffmpeg -y -i "{target_no_audio}" -i "{temp_audio_aac}" -c copy "{target_video}" -loglevel error')

    if os.path.exists(target_video) and os.path.getsize(target_video) > 0:
        os.remove(target_no_audio)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True
    else:
        os.rename(target_no_audio, target_video)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False


# --- Core Dehazing Processing Function ---
def process_single_video(input_path, output_path, device_obj=device):
    """
    Processes a single video: performs CUDA MSRCR dehazing and outputs a .mp4 video.
    """
    print(f"\n[PROCESSING] Input:  '{input_path}'")
    print(f"             Output: '{output_path}'")

    meta = validate_video(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, meta['fps'], (meta['width'], meta['height']))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for target path: {output_path}")

    tot_frames = meta['total_frames']
    pbar = tqdm(total=tot_frames if tot_frames > 0 else None, desc=f"Dehazing {os.path.basename(input_path)}", unit="frame")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dehazed_frame = apply_msrcr(frame, device_obj)
            out.write(dehazed_frame)
            frame_count += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        out.release()

    print(f"[SUCCESS] Wrote {frame_count} dehazed frames to '{output_path}'.")

    # Attempt audio transfer
    audio_transferred = transfer_audio(input_path, output_path)
    if audio_transferred:
        print("[INFO] Audio track transferred successfully.")


# --- Main Batch Execution ---
def main():
    parser = argparse.ArgumentParser(description="CUDA Batch Video Dehazing & Color Enhancement Pipeline")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop processing if a video format error occurs")
    args = parser.parse_args()

    # Pre-configured Source to Target Directory Mappings
    directory_mappings = [
        {
            "source": r"C:\Users\parsa\Desktop\Code\Datasets\Parsa's dataset\2026-07-10_Field_Trip2\system_pos1\system_pos1",
            "target": r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\Dehazed Videos\system_pose1"
        },
        {
            "source": r"C:\Users\parsa\Desktop\Code\Datasets\Parsa's dataset\2026-07-10_Field_Trip2\system_position2",
            "target": r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\Dehazed Videos\system_pose2"
        },
        {
            "source": r"C:\Users\parsa\Desktop\Code\Datasets\Parsa's dataset\2026-07-10_Field_Trip2\sneak_peek",
            "target": r"C:\Users\parsa\Desktop\Code\Datasets\Team's dataset\Dehazed Videos\sneak_peek"
        }
    ]

    print("=========================================================")
    print("      CUDA Batch Video Dehazing & Color Upgrade          ")
    print("=========================================================")
    print(f"Device: {device}")

    total_processed = 0
    total_errors = 0

    for mapping in directory_mappings:
        src_dir = mapping["source"]
        tgt_dir = mapping["target"]

        print(f"\n---------------------------------------------------------")
        print(f"Scanning Source Folder: '{src_dir}'")
        print(f"Target Destination:     '{tgt_dir}'")

        if not os.path.exists(src_dir):
            print(f"[WARNING] Source directory does not exist: '{src_dir}'. Skipping.")
            continue

        os.makedirs(tgt_dir, exist_ok=True)

        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.h264', '*.264')
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(src_dir, ext)))

        video_files = sorted(list(set(video_files)))

        if not video_files:
            print(f"[INFO] No video files found in '{src_dir}'.")
            continue

        print(f"[INFO] Found {len(video_files)} video(s) to process.")

        for vid_path in video_files:
            filename = os.path.basename(vid_path)
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(tgt_dir, f"{base_name}_dehazed.mp4")

            try:
                process_single_video(vid_path, out_path, device_obj=device)
                total_processed += 1
            except VideoFormatError as vfe:
                total_errors += 1
                print(f"\n[VIDEO FORMAT ERROR] {vfe}")
                if args.stop_on_error:
                    print("Halting execution due to --stop-on-error flag.")
                    sys.exit(1)
            except Exception as e:
                total_errors += 1
                print(f"\n[ERROR] Unexpected error processing '{vid_path}': {e}")
                if args.stop_on_error:
                    raise e

    print("\n=========================================================")
    print(f" Batch processing finished. Successfully processed: {total_processed} videos. Errors/Format issues: {total_errors}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
