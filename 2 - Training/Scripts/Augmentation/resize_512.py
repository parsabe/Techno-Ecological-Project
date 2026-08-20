"""
Resizes images to 512x512 while maintaining their original aspect ratio.
Images are fitted inside a 512x512 bounding box, and the remaining space 
is filled with padding (letterbox method).

Recursively scans subdirectories and mirrors the original folder structure 
in the target output directory.

Usage:
    pip install pillow
    python resize_512.py /path/to/input /path/to/output

Note: You can change PAD_COLOR to any RGB color you prefer.
      If you need a transparent background (PNG), set PAD_COLOR to (0, 0, 0, 0)
      and change the image mode to "RGBA".
"""

import sys
from pathlib import Path
from PIL import Image

TARGET_SIZE = 512
PAD_COLOR = (114, 114, 114)  # Default padding color (RGB)
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def resize_with_padding(img: Image.Image, size: int, pad_color) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (size, size), pad_color)
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def main():
    if len(sys.argv) != 3:
        print("Usage: python resize_512.py <input_folder> <output_folder>")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Search for all supported image files recursively
    files = [f for f in src_dir.rglob("*") if f.suffix.lower() in VALID_EXT]
    print(f"Found {len(files)} images (including subdirectories).")

    for i, f in enumerate(files, 1):
        try:
            # Preserve folder structure relative to the source directory
            rel_path = f.relative_to(src_dir)
            out_path = dst_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(f) as img:
                out = resize_with_padding(img, TARGET_SIZE, PAD_COLOR)
                out.save(out_path)
        except Exception as e:
            print(f"ERROR ({f}): {e}")
            
        if i % 50 == 0 or i == len(files):
            print(f"Progress: {i}/{len(files)} completed")

    print("Processing complete.")


if __name__ == "__main__":
    main()