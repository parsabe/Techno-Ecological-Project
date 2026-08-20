import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo
skvideo.setFFmpegPath("C:/ffmpeg/bin")
import skvideo.io
from queue import Queue, Empty
from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

# --- MSRCR Dehazing Implementation ---

# Separable Gaussian Blur on GPU with Downsampling for large sigmas
def gaussian_blur_gpu(tensor, sigma):
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
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = tensor_down.shape[1]
    kernel_h = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
    kernel_w = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)

    pad = ksize // 2
    h_down_curr, w_down_curr = tensor_down.shape[2], tensor_down.shape[3]
    if pad >= h_down_curr or pad >= w_down_curr:
        pad = min(h_down_curr - 1, w_down_curr - 1, pad)
        ksize = 2 * pad + 1
        x = torch.arange(ksize, dtype=torch.float32, device=tensor_down.device)
        mean = (ksize - 1) / 2.0
        kernel_1d = torch.exp(-((x - mean) ** 2) / (2.0 * variance))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_h = kernel_1d.view(1, 1, ksize, 1).repeat(channels, 1, 1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, ksize).repeat(channels, 1, 1, 1)

    padded = F.pad(tensor_down, (pad, pad, pad, pad), mode='replicate')
    blurred = F.conv2d(padded, kernel_h, groups=channels)
    blurred = F.conv2d(blurred, kernel_w, groups=channels)

    if sigma > target_sigma:
        blurred = F.interpolate(blurred, size=(h, w), mode='bilinear', align_corners=False)
    return blurred

# MSRCR pre-processing on GPU
def apply_msrcr_cuda(img, device_obj):
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device_obj, dtype=torch.float32)
    img_tensor = img_tensor + 1.0

    sigmas = [15, 80, 250]
    retinex = torch.zeros_like(img_tensor, dtype=torch.float32)

    for s in sigmas:
        blur = gaussian_blur_gpu(img_tensor, s)
        retinex += torch.log10(img_tensor) - torch.log10(blur)
    retinex /= len(sigmas)

    channel_sum = torch.sum(img_tensor, dim=1, keepdim=True)
    color_rest = torch.log10(125.0 * img_tensor) - torch.log10(channel_sum + 1.0)

    msrcr = retinex * color_rest

    min_val = msrcr.min()
    max_val = msrcr.max()
    if max_val > min_val:
        msrcr = (msrcr - min_val) / (max_val - min_val) * 255.0
    else:
        msrcr = torch.zeros_like(msrcr)

    msrcr = torch.clamp(msrcr, 0, 255)
    return msrcr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

# MSRCR pre-processing on CPU (Fallback)
def apply_msrcr_cpu(img):
    img_float = np.float32(img) + 1.0
    sigmas = [15, 80, 250]
    retinex = np.zeros_like(img_float)
    for s in sigmas:
        blur = cv2.GaussianBlur(img_float, (0, 0), s)
        retinex += np.log10(img_float) - np.log10(blur)
    retinex /= len(sigmas)
    color_rest = np.log10(125.0 * img_float) - np.log10(np.sum(img_float, axis=2, keepdims=True) + 1.0)
    msrcr = retinex * color_rest
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX)
    return msrcr.astype(np.uint8)

# Unified apply_msrcr function that redirects based on device availability
def apply_msrcr(img, device_obj=None):
    if device_obj is not None:
        device_type = device_obj.type if hasattr(device_obj, 'type') else str(device_obj)
        if 'cuda' in device_type:
            return apply_msrcr_cuda(img, device_obj)
    return apply_msrcr_cpu(img)

def transferAudio(sourceVideo, targetVideo):
    import shutil
    import moviepy.editor
    tempAudioFileName = "./temp/audio.mkv"

    # split audio from original video file and store in "temp" directory
    if True:

        # clear old "temp" directory if it exits
        if os.path.isdir("temp"):
            # remove temp directory
            shutil.rmtree("temp")
        # create new "temp" directory
        os.makedirs("temp")
        # extract audio from video
        os.system('ffmpeg -y -i "{}" -c:a copy -vn {}'.format(sourceVideo, tempAudioFileName))

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    # combine audio file and new video file
    os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))

    if os.path.getsize(targetVideo) == 0: # if ffmpeg failed to merge the video and audio together try converting the audio to aac
        tempAudioFileName = "./temp/audio.m4a"
        os.system('ffmpeg -y -i "{}" -c:a aac -b:a 160k -vn {}'.format(sourceVideo, tempAudioFileName))
        os.system('ffmpeg -y -i "{}" -i {} -c copy "{}"'.format(targetNoAudio, tempAudioFileName, targetVideo))
        if (os.path.getsize(targetVideo) == 0): # if aac is not supported by selected format
            os.rename(targetNoAudio, targetVideo)
            print("Audio transfer failed. Interpolated video will have no audio")
        else:
            print("Lossless audio transfer failed. Audio was transcoded to AAC (M4A) instead.")

            # remove audio-less video
            os.remove(targetNoAudio)
    else:
        os.remove(targetNoAudio)

    # remove temp directory
    shutil.rmtree("temp")

parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)
parser.add_argument('--dehaze', action='store_true', help='apply MSRCR dehazing before interpolation')

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (not args.video is None or not args.img is None)
if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale==1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if not args.img is None:
    args.png = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script is configured to only run on GPU.")

device = torch.device("cuda")
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if(args.fp16):
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
print("Loaded 3.x/4.x HD model.")
model.eval()
model.device()

original_video = args.video
if not args.video is None and args.dehaze:
    print(f"Applying MSRCR dehazing pre-processing to {args.video}...")
    dehazed_video_path = os.path.splitext(args.video)[0] + "_dehazed.mp4"
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open input video {args.video} for dehazing.")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc_dehaze = cv2.VideoWriter_fourcc(*'mp4v')
    out_dehaze = cv2.VideoWriter(dehazed_video_path, fourcc_dehaze, fps, (frame_width, frame_height))
    if not out_dehaze.isOpened():
        cap.release()
        raise RuntimeError(f"Error: Could not open video writer for {dehazed_video_path}")
        
    pbar_dehaze = tqdm(total=tot_frames, desc="Dehazing video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dehazed_frame = apply_msrcr(frame, device)
        out_dehaze.write(dehazed_frame)
        pbar_dehaze.update(1)
        
    cap.release()
    out_dehaze.release()
    pbar_dehaze.close()
    print(f"Dehazing completed. Saved intermediate video to {dehazed_video_path}")
    args.video = dehazed_video_path

if not args.video is None:
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
    else:
        fpsNotAssigned = False
    videogen = skvideo.io.vreader(args.video)
    lastframe = next(videogen)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path_wo_ext, ext = os.path.splitext(args.video)
    print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, args.ext, tot_frame, fps, args.fps))
    if args.png == False and fpsNotAssigned == True:
        print("The audio will be merged after interpolation process")
    else:
        print("Will not merge audio because using png or fps flag!")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key= lambda x:int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]
h, w, _ = lastframe.shape
vid_out_name = None
vid_out = None
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
else:
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, args.fps, (w, h))

def clear_write_buffer(user_args, write_buffer):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            vid_out.write(item[:, :, ::-1])

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        for frame in videogen:
            if not user_args.img is None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, n):    
    global model
    if model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(model.inference(I0, I1, (i+1) * 1. / (n+1), args.scale))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n//2)
        second_half = make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

def pad_image(img):
    if(args.fp16):
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)

if args.montage:
    left = w // 4
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=tot_frame)
if args.montage:
    lastframe = lastframe[:, left: left + w]
write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
I1 = pad_image(I1)
temp = None # save lastframe when processing static frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get() # read a new frame
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I1 = model.inference(I0, I1, scale=args.scale)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        
    if ssim < 0.2:
        output = []
        for i in range(args.multi - 1):
            output.append(I0)
        '''
        output = []
        step = 1 / args.multi
        alpha = 0
        for i in range(args.multi - 1):
            alpha += step
            beta = 1-alpha
            output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        '''
    else:
        output = make_inference(I0, I1, args.multi - 1)

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(np.concatenate((lastframe, mid[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
    pbar.update(1)
    lastframe = frame
    if break_flag:
        break

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
write_buffer.put(None)

import time
while(not write_buffer.empty()):
    time.sleep(0.1)
pbar.close()
if not vid_out is None:
    vid_out.release()

# move audio to new video file if appropriate
if args.png == False and fpsNotAssigned == True and not args.video is None:
    try:
        transferAudio(original_video, vid_out_name)
    except:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
