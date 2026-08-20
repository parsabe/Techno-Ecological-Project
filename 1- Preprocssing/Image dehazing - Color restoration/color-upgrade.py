import os
import sys
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

# --------------------------------------------
# PyTorch Model Architecture for 4xNomos8kSC
# --------------------------------------------

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    )

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = conv(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = conv(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = conv(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = conv(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = conv(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        return x + self.sub(x)

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        
        # Trunk blocks (23 RRDB modules + 1 final trunk conv)
        rrdb_blocks = []
        for _ in range(nb):
            rrdb_blocks.append(RRDB(nf, gc))
        rrdb_blocks.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        
        # Upsampling block
        upsampler = []
        if scale == 4:
            upsampler.append(nn.Upsample(scale_factor=2, mode='nearest'))
            upsampler.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
            upsampler.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            upsampler.append(nn.Upsample(scale_factor=2, mode='nearest'))
            upsampler.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
            upsampler.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif scale == 2:
            upsampler.append(nn.Upsample(scale_factor=2, mode='nearest'))
            upsampler.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
            upsampler.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            
        self.model = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True),
            ShortcutBlock(nn.Sequential(*rrdb_blocks)),
            *upsampler,
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        return self.model(x)

# --------------------------------------------
# Tiled Processing to Prevent VRAM OOM
# --------------------------------------------

def tile_process(model, img_tensor, tile_size=256, tile_pad=10, scale=4):
    """
    Processes the input image in tiles to avoid CUDA OOM.
    img_tensor: Tensor of shape [1, C, H, W]
    """
    batch, channel, height, width = img_tensor.shape
    output_shape = (batch, channel, height * scale, width * scale)
    output = torch.zeros(output_shape, device=img_tensor.device, dtype=img_tensor.dtype)

    stride = tile_size - 2 * tile_pad
    
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            # Tile coordinates with padding
            y_start = max(y - tile_pad, 0)
            y_end = min(y + tile_size - tile_pad, height)
            x_start = max(x - tile_pad, 0)
            x_end = min(x + tile_size - tile_pad, width)

            # Extract tile
            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            
            # Pad tile if needed to be a multiple of 4
            h_pad = (4 - tile.shape[2] % 4) % 4
            w_pad = (4 - tile.shape[3] % 4) % 4
            if h_pad > 0 or w_pad > 0:
                tile = F.pad(tile, (0, w_pad, 0, h_pad), mode='reflect')

            # Run model on tile
            with torch.no_grad():
                tile_out = model(tile)
                
            # Remove padding
            if h_pad > 0 or w_pad > 0:
                tile_out = tile_out[:, :, :-h_pad * scale if h_pad > 0 else None, :-w_pad * scale if w_pad > 0 else None]

            # Determine crop coordinates for output
            crop_y_start = y - y_start
            crop_y_end = min(y + stride, height) - y_start
            crop_x_start = x - x_start
            crop_x_end = min(x + stride, width) - x_start

            # Output crop boundaries (scale factor applied)
            out_crop_y_start = crop_y_start * scale
            out_crop_y_end = crop_y_end * scale
            out_crop_x_start = crop_x_start * scale
            out_crop_x_end = crop_x_end * scale

            # Output coordinates
            out_y_start = y * scale
            out_y_end = min(y + stride, height) * scale
            out_x_start = x * scale
            out_x_end = min(x + stride, width) * scale

            output[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                tile_out[:, :, out_crop_y_start:out_crop_y_end, out_crop_x_start:out_crop_x_end]
                
    return output

# --------------------------------------------
# Main Process Loop
# --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="4xNomos8kSC Video Resolution and Detail Enhancer (CUDA Accelerated)")
    parser.add_argument("--input", type=str, default="main.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video file")
    parser.add_argument("--model_path", type=str, default="models/4xNomos8kSC.pth", help="Path to the .pth model checkpoint")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for memory-safe processing (0 to disable tiling)")
    parser.add_argument("--tile_pad", type=int, default=10, help="Overlap padding size between tiles")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision (FP16) mode for faster and memory-saving inference")
    parser.add_argument("--limit_frames", type=int, default=0, help="Limit processing to a specific number of frames (0 for full video)")
    
    args = parser.parse_args()

    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA is not available. This script is optimized for and requires GPU/CUDA execution.")
        sys.exit(1)
        
    device = torch.device("cuda")
    print(f"CUDA initialized. Processing using GPU device: {torch.cuda.get_device_name(0)}")

    # Initialize model architecture and load state dict
    print(f"Loading model checkpoint from '{args.model_path}'...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4)
    
    try:
        state_dict = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)
        
    model = model.to(device)
    model.eval()

    if args.fp16:
        print("Using half precision (FP16) inference mode.")
        model = model.half()
        
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at path '{args.input}'")
        sys.exit(1)
        
    # Check if input is an image or video
    input_lower = args.input.lower()
    is_image = any(input_lower.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'])
    
    if is_image:
        print(f"Input detected as image. Processing '{args.input}'...")
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"Error: Could not read image file '{args.input}'")
            sys.exit(1)
            
        orig_height, orig_width = frame.shape[:2]
        target_width = orig_width * 4
        target_height = orig_height * 4
        
        print(f"Dimensions: {orig_width}x{orig_height} -> {target_width}x{target_height}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        frame_tensor = frame_tensor.float() / 255.0
        
        if args.fp16:
            frame_tensor = frame_tensor.half()
            
        with torch.no_grad():
            if args.tile_size > 0:
                out_tensor = tile_process(model, frame_tensor, tile_size=args.tile_size, tile_pad=args.tile_pad, scale=4)
            else:
                h_pad = (4 - frame_tensor.shape[2] % 4) % 4
                w_pad = (4 - frame_tensor.shape[3] % 4) % 4
                if h_pad > 0 or w_pad > 0:
                    frame_tensor = F.pad(frame_tensor, (0, w_pad, 0, h_pad), mode='reflect')
                out_tensor = model(frame_tensor)
                if h_pad > 0 or w_pad > 0:
                    out_tensor = out_tensor[:, :, :-h_pad * 4 if h_pad > 0 else None, :-w_pad * 4 if w_pad > 0 else None]
                    
        out_tensor = torch.clamp(out_tensor, 0.0, 1.0)
        if args.fp16:
            out_tensor = out_tensor.float()
            
        out_frame = out_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_frame = (out_frame * 255.0).astype(np.uint8)
        out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        
        if args.output is None:
            input_dir = os.path.dirname(args.input)
            input_name = os.path.splitext(os.path.basename(args.input))[0]
            args.output = os.path.join(input_dir, f"{input_name}_upgraded_4X.png")
            
        out_dir = os.path.dirname(args.output)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        cv2.imwrite(args.output, out_frame_bgr)
        print(f"Processing complete! Saved enhanced image to '{args.output}'")
        return
        
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.input}'")
        sys.exit(1)
        
    # Read video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    target_width = orig_width * 4
    target_height = orig_height * 4
    
    if args.limit_frames > 0:
        total_frames = min(total_frames, args.limit_frames)
        
    print(f"Input dimensions: {orig_width}x{orig_height} @ {fps} fps")
    print(f"Output target dimensions: {target_width}x{target_height} (4x upscale)")
    
    # Configure output file path
    if args.output is None:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(input_dir, f"{input_name}_upgraded_4X.mp4")
        
    # Create output directory if needed
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (target_width, target_height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for output file '{args.output}'")
        cap.release()
        sys.exit(1)
        
    pbar = tqdm(total=total_frames, desc="Upgrading details and resolution")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame (BGR -> RGB, normalize to [0, 1], add batch dim)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
            frame_tensor = frame_tensor.float() / 255.0
            
            if args.fp16:
                frame_tensor = frame_tensor.half()
                
            # Inference (Tiled vs Full-frame)
            with torch.no_grad():
                if args.tile_size > 0:
                    out_tensor = tile_process(model, frame_tensor, tile_size=args.tile_size, tile_pad=args.tile_pad, scale=4)
                else:
                    # Divisibility pad if needed
                    h_pad = (4 - frame_tensor.shape[2] % 4) % 4
                    w_pad = (4 - frame_tensor.shape[3] % 4) % 4
                    if h_pad > 0 or w_pad > 0:
                        frame_tensor = F.pad(frame_tensor, (0, w_pad, 0, h_pad), mode='reflect')
                        
                    out_tensor = model(frame_tensor)
                    
                    if h_pad > 0 or w_pad > 0:
                        out_tensor = out_tensor[:, :, :-h_pad * 4 if h_pad > 0 else None, :-w_pad * 4 if w_pad > 0 else None]
            
            # Postprocess frame (Clamping, Denormalize, RGB -> BGR)
            out_tensor = torch.clamp(out_tensor, 0.0, 1.0)
            if args.fp16:
                out_tensor = out_tensor.float()
                
            out_frame = out_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_frame = (out_frame * 255.0).astype(np.uint8)
            out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            
            # Write to output file
            out.write(out_frame_bgr)
            
            frame_count += 1
            pbar.update(1)
            
            if args.limit_frames > 0 and frame_count >= args.limit_frames:
                break
                
    except Exception as e:
        print(f"\nAn error occurred during video enhancement: {e}")
    finally:
        pbar.close()
        cap.release()
        out.release()
        print(f"\nProcessing complete! Enhanced video saved to '{args.output}'")

if __name__ == "__main__":
    main()
