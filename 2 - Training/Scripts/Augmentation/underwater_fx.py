#!/usr/bin/env python3
"""
underwater_fx.py
-----------------
Applies a realistic underwater visual effect to images.
(Uses only Pillow + NumPy; OpenCV is not required)

Applied effects:
  1. Color cast (attenuates red, boosts blue/green) -> Controlled via --color
  2. Turbidity / haze simulation (suspended particles)
     -> Color tone linked to --color, contrast reduction linked to --intensity
  3. Gaussian + film-grain noise (suspended particles) -> --intensity
  4. Subtle box-blur (underwater light scattering)     -> --intensity
  5. Vignette (edge darkening)                          -> --intensity

--color and --intensity are COMPLETELY INDEPENDENT:
  --color     : Controls color shift + haze hue (0.0-1.0)
  --intensity : Controls noise, blur, vignette, depth gradient,
                and haze contrast reduction (0.0-1.0)

In directory mode, subdirectories are scanned recursively, preserving
the original folder structure in the output destination.

Usage:
  Single file:
    python underwater_fx.py --input image.jpg --output underwater_image.jpg

  Entire directory (including subdirectories):
    python underwater_fx.py --input /path/to/input --output /path/to/output

  Customizing color shift and effect intensity separately:
    python underwater_fx.py --input /path/to/input --output /path/to/output --color 0.5 --intensity 0.8

Requirements:
  pip install pillow numpy
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def apply_color_cast(img, color):
    """Attenuate the red channel and boost green/blue (deep water hue). img: RGB float32."""
    out = img.copy()
    out[:, :, 0] *= (1.0 - 0.35 * color)   # R
    out[:, :, 1] *= (1.0 - 0.05 * color)   # G
    out[:, :, 2] *= (1.0 + 0.06 * color)   # B
    return np.clip(out, 0, 255)


def apply_turbidity_haze(img, color, intensity):
    """Turbid water / suspended particle feel: lowers contrast and blends 
    a semi-transparent teal 'water layer' on top. img: RGB float32."""
    haze_color = np.array([70, 115, 125], dtype=np.float32)  # RGB (soft teal tone)
    haze_layer = np.full_like(img, haze_color)

    alpha = 0.45 * color  # Layer opacity -> COLOR
    blended = img * (1 - alpha) + haze_layer * alpha

    # Slightly lower contrast (clarity drops in turbid water) -> INTENSITY
    mean = blended.mean()
    contrast_factor = 1.0 - 0.25 * intensity
    blended = (blended - mean) * contrast_factor + mean

    return np.clip(blended, 0, 255)


def apply_depth_gradient(img, intensity):
    """Slightly brighter/clearer top, darker bottom -> simulates depth progression."""
    h, w = img.shape[:2]
    gradient = np.linspace(0, 1, h, dtype=np.float32).reshape(h, 1, 1)
    darken = 1.0 - 0.25 * intensity * gradient
    return img * darken


def apply_noise(img, intensity, seed=None):
    """Gaussian noise + film grain -> simulates suspended particles/plankton in water."""
    rng = np.random.default_rng(seed)
    h, w, c = img.shape

    sigma = 20 * intensity
    gauss_noise = rng.normal(0, sigma, (h, w, c)).astype(np.float32)

    # Sparse, larger "particle" specks (floating debris/plankton feel)
    particle_mask = (rng.random((h, w)) < (0.0015 * intensity)).astype(np.float32)
    particle_layer = np.zeros_like(img, dtype=np.float32)
    for ch in range(c):
        particle_layer[:, :, ch] = particle_mask * rng.uniform(120, 220)

    out = img + gauss_noise + particle_layer * 0.5
    return np.clip(out, 0, 255)


def _box_blur(img, ksize):
    """Simple separable box blur (pure NumPy, no cv2/scipy required)."""
    if ksize < 3:
        return img
    if ksize % 2 == 0:
        ksize += 1
    pad = ksize // 2

    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w, c = img.shape

    # Horizontal cumulative sum for sliding window average
    cs = np.cumsum(padded, axis=1)
    cs = np.concatenate([np.zeros((cs.shape[0], 1, c), dtype=cs.dtype), cs], axis=1)
    horiz = (cs[:, ksize:, :] - cs[:, :-ksize, :]) / ksize

    cs2 = np.cumsum(horiz, axis=0)
    cs2 = np.concatenate([np.zeros((1, cs2.shape[1], c), dtype=cs2.dtype), cs2], axis=0)
    blurred = (cs2[ksize:, :, :] - cs2[:-ksize, :, :]) / ksize

    return blurred[:h, :w, :]


def apply_light_scatter_blur(img, intensity):
    """Light blur mimicking underwater light scattering (blended with original)."""
    ksize = max(3, int(3 + 6 * intensity))
    blurred = _box_blur(img, ksize)
    mix = 0.35 * intensity
    return img * (1 - mix) + blurred * mix


def apply_vignette(img, intensity):
    """Darkens edges to simulate light falloff in deeper water."""
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1 - (dist / max_dist) * (0.35 * intensity)
    vignette = np.clip(vignette, 0, 1)[:, :, None]
    return img * vignette


def underwater_effect(img_rgb_uint8, color=0.6, intensity=0.6, seed=None):
    """Applies all underwater effects sequentially. Input/Output: RGB uint8 numpy array.

    color     : Color cast + haze color intensity (0.0-1.0)
    intensity : Noise, blur, vignette, depth gradient, and haze contrast intensity (0.0-1.0)
    """
    color = float(np.clip(color, 0.0, 1.0))
    intensity = float(np.clip(intensity, 0.0, 1.0))

    out = img_rgb_uint8.astype(np.float32)
    out = apply_color_cast(out, color)
    out = apply_turbidity_haze(out, color, intensity)
    out = apply_depth_gradient(out, intensity)
    out = apply_light_scatter_blur(out, intensity)
    out = apply_noise(out, intensity, seed=seed)
    out = apply_vignette(out, intensity)

    return np.clip(out, 0, 255).astype(np.uint8)


def process_file(input_path, output_path, color, intensity, seed=None):
    try:
        with Image.open(input_path) as im:
            im = im.convert("RGB")
            arr = np.array(im)
    except Exception as e:
        print(f"  [SKIPPED] Could not read image: {input_path} ({e})")
        return False

    result_arr = underwater_effect(arr, color=color, intensity=intensity, seed=seed)

    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(result_arr, mode="RGB").save(output_path)
    except Exception as e:
        print(f"  [ERROR] Could not save image: {output_path} ({e})")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Applies an underwater visual effect to images.")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory path")
    parser.add_argument("--output", "-o", required=True, help="Output image file or directory path")
    parser.add_argument("--color", "-c", type=float, default=0.6,
                        help="Color shift intensity range 0.0-1.0 (default: 0.6)")
    parser.add_argument("--intensity", "-s", type=float, default=0.6,
                        help="Noise/blur/vignette/haze contrast intensity range 0.0-1.0 (default: 0.6)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for noise generation (optional)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if os.path.isdir(input_path):
        # Recursively find all images while preserving folder structure
        found = []
        for root, dirs, files in os.walk(input_path):
            for fname in sorted(files):
                if fname.lower().endswith(SUPPORTED_EXT):
                    in_f = os.path.join(root, fname)
                    rel_path = os.path.relpath(in_f, input_path)
                    out_f = os.path.join(output_path, rel_path)
                    found.append((in_f, out_f))

        if not found:
            print("No supported images found in the specified directory (or its subdirectories).")
            sys.exit(1)

        os.makedirs(output_path, exist_ok=True)
        print(f"Found {len(found)} image(s) (including subdirectories). "
              f"Processing (color={args.color}, intensity={args.intensity})...")
        success = 0
        for in_f, out_f in found:
            rel = os.path.relpath(in_f, input_path)
            print(f"  -> {rel}")
            if process_file(in_f, out_f, args.color, args.intensity, seed=args.seed):
                success += 1
        print(f"Completed: {success}/{len(found)} images processed -> {output_path}")

    elif os.path.isfile(input_path):
        if process_file(input_path, output_path, args.color, args.intensity, seed=args.seed):
            print(f"Completed -> {output_path}")
    else:
        print(f"Input path not found: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()