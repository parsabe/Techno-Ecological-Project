import os
import cv2
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================
DATASET_DIR = "fish_dataset/"    # Root directory containing class subfolders
TEXT_PROMPT = "fish."            # Object prompt for the model
BOX_THRESHOLD = 0.30             # Confidence threshold (0.0 - 1.0)

MODEL_ID = "Idea-Research/grounding-dino-tiny" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[INFO] Using device: {device.upper()}\n")

# ==========================================
# MODEL LOADING
# ==========================================
print("[INFO] Loading model, please wait...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
print("[INFO] Model loaded successfully.\n")

# List classes alphabetically
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

print("--- DETECTED CLASSES AND CLASS IDs ---")
for class_id, class_name in enumerate(classes):
    print(f"ID {class_id}: {class_name}")
print("---------------------------------------\n")

# ==========================================
# MAIN ANNOTATION LOOP
# ==========================================
for class_id, class_name in enumerate(classes):
    class_path = os.path.join(DATASET_DIR, class_name)
    print(f"[PROCESSING] Processing class: {class_name.upper()} (Class ID: {class_id})")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images = [f for f in os.listdir(class_path) if f.lower().endswith(valid_extensions)]
    
    for img_name in tqdm(images, desc=f"Annotating {class_name}"):
        img_path = os.path.join(class_path, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            
            inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Post-processing step
            results = processor.post_process_grounded_object_detection(
                outputs, 
                inputs.input_ids, 
                target_sizes=[(height, width)]
            )[0]
            
            scores = results["scores"].cpu().numpy()
            all_boxes = results["boxes"].cpu().numpy()
            
            # Filter by confidence threshold
            boxes = all_boxes[scores > BOX_THRESHOLD]
            
            yolo_lines = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                
                # YOLO Normalization Calculations
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                
                x_center = min(max(x_center, 0.0), 1.0)
                y_center = min(max(y_center, 0.0), 1.0)
                box_width = min(max(box_width, 0.0), 1.0)
                box_height = min(max(box_height, 0.0), 1.0)
                
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                yolo_lines.append(line)
            
            # Save location: directly alongside the image file (in the same directory)
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(class_path, txt_name)
            
            with open(txt_path, "w") as f:
                f.write("\n".join(yolo_lines))
                
        except Exception as e:
            print(f"\n[ERROR] An error occurred while processing {img_name}: {str(e)}")

print(f"\n[COMPLETED] All annotations have been saved alongside the corresponding images (as .txt)!")