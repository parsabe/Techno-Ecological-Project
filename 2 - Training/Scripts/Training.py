from ultralytics import YOLO


def main():
    # --- 1. Model Selection ---
    # n (nano)   -> fastest, lowest accuracy, suitable for small datasets/CPU
    # s (small)  -> good balance of speed and accuracy
    # m (medium) -> great starting point for ~10k images (recommended)
    # l (large)  -> higher accuracy, requires more GPU memory and time
    # x (xlarge) -> highest accuracy potential, slowest
    model = YOLO("yolov8m.pt")  # Start with pretrained weights (transfer learning)

    # --- 2. Training ---
    results = model.train(
        data="data.yaml",
        epochs=400,            # Can be set high since early stopping is controlled by patience
        imgsz=512,             # Try 960-1280 if fish are small or shot from a distance
        batch=16,              # Adjust based on GPU memory (lower if VRAM is insufficient)
        patience=30,           # Stop training if val metric doesn't improve for 30 epochs
        device=0,              # GPU id; use "cpu" if no GPU is available
        workers=8,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,           # Cosine learning rate scheduler -> more stable convergence
        project="fish_detection",
        name="yolov8m_fish_v1",

        # --- Augmentation (Aggressive augmentation is fine for single-class) ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,          # Slight rotation
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.3,            # Vertical flip enabled as fish can appear from any orientation
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,       # Turn off mosaic in the last 10 epochs -> helpful for fine-tuning

        save=True,
        save_period=10,
        val=True,
        plots=True,
        seed=42,
        deterministic=True,
        verbose=True,
    )

if __name__ == "__main__":
    main()