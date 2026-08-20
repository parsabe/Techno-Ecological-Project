"""
Fish Detection and Tracking Script (BoT-SORT)
-----------------------------------------------
- Detects fish in a video using a YOLO model
- Assigns a unique ID to each fish using the BoT-SORT tracking algorithm
- Displays the running total of unique detected fish on screen
- Prints and saves a per-species summary (unique count per class) to a CSV file
- Saves the annotated output to a new video file

Usage:
    python botsort.py --model best.pt --source video.mp4 --output output.mp4

    example:
    python botsort.py --model small.pt --conf 0.25  --iou 0.8 --source input.mp4 --output output3.mp4 --device 0 --imgsz 1080 --half --csv test.csv

Requirements:
    pip install ultralytics opencv-python
"""

import argparse
import csv
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO


class ReIDTrackStitcher:
    """
    Feature Re-ID Track Stitcher.
    Extracts visual appearance embedding vectors (color distribution in LAB/HSV + spatial aspect ratio) for targets.
    Maintains a lost track memory gallery (`ReID_Gallery`) to match new tracks to recently lost ones.
    If cosine similarity > 0.82 within a memory window (e.g. 15s / 450 frames), maps new track ID to original track ID.
    """
    def __init__(self, memory_frames=450, similarity_threshold=0.82):
        self.memory_frames = memory_frames
        self.similarity_threshold = similarity_threshold
        self.gallery = {}   # lost_track_id: {"embedding": norm_vec, "last_frame": frame_idx, "cls_name": cls_name}
        self.active_tracks = {}  # track_id: {"embedding": norm_vec, "last_frame": frame_idx, "cls_name": cls_name}
        self.id_mapping = {}  # new_track_id -> original_track_id

    def extract_embedding(self, img, box):
        x1, y1, x2, y2 = map(int, box)
        h, w, c = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(64, dtype=np.float32)
        crop = img[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hist_l = cv2.calcHist([lab], [0], None, [16], [0, 256])
        
        vec = np.vstack([hist_h, hist_s, hist_v, hist_l]).flatten()
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def process_frame_tracks(self, frame_idx, current_frame_targets, img):
        """
        current_frame_targets: list of dicts {"track_id": int, "box": [x1,y1,x2,y2], "cls_name": str}
        Returns updated targets with re-mapped persistent track_ids.
        """
        current_active_ids = set()
        updated_targets = []
        
        for tgt in current_frame_targets:
            raw_id = tgt["track_id"]
            box = tgt["box"]
            cls_name = tgt["cls_name"]
            
            mapped_id = self.id_mapping.get(raw_id, raw_id)
            emb = self.extract_embedding(img, box)
            
            if mapped_id not in self.active_tracks and mapped_id not in self.gallery:
                best_match_id = None
                best_sim = -1.0
                
                for g_id, g_info in list(self.gallery.items()):
                    if frame_idx - g_info["last_frame"] <= self.memory_frames and g_info["cls_name"] == cls_name:
                        sim = float(np.dot(emb, g_info["embedding"]))
                        if sim > best_sim and sim >= self.similarity_threshold:
                            best_sim = sim
                            best_match_id = g_id
                            
                if best_match_id is not None:
                    self.id_mapping[raw_id] = best_match_id
                    mapped_id = best_match_id
                    if best_match_id in self.gallery:
                        del self.gallery[best_match_id]
                    
            self.active_tracks[mapped_id] = {
                "embedding": emb,
                "last_frame": frame_idx,
                "cls_name": cls_name
            }
            current_active_ids.add(mapped_id)
            
            tgt_copy = dict(tgt)
            tgt_copy["track_id"] = mapped_id
            updated_targets.append(tgt_copy)
            
        for a_id in list(self.active_tracks.keys()):
            if a_id not in current_active_ids:
                if frame_idx - self.active_tracks[a_id]["last_frame"] <= self.memory_frames:
                    self.gallery[a_id] = self.active_tracks[a_id]
                del self.active_tracks[a_id]
                
        for g_id in list(self.gallery.keys()):
            if frame_idx - self.gallery[g_id]["last_frame"] > self.memory_frames:
                del self.gallery[g_id]
                
        return updated_targets



def run_tracking(model_path: str, source_path: str, output_path: str,
                  conf: float = 0.25, iou: float = 0.5,
                  tracker_cfg: str = "botsort.yaml",
                  show: bool = False,
                  device: str = None, imgsz: int = 640, half: bool = False,
                  csv_output: str = "detections_summary.csv"):

    # Load the YOLO model
    model = YOLO(model_path)

    # Open the source video (to read width/height/fps metadata)
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    cap.release()  # ultralytics will open its own stream; this was only for metadata

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set of all unique track IDs seen so far (overall total)
    unique_ids = set()

    # Per-class unique track IDs: {cls_id: set(track_id, ...)}
    class_unique_ids = {}
    # Class id -> class name mapping (used for reporting/CSV)
    class_names = {}

    # Run model.track() as a frame-by-frame stream (persist=True keeps IDs consistent across frames)
    results_generator = model.track(
        source=source_path,
        conf=conf,
        iou=iou,
        tracker=tracker_cfg,   # "botsort.yaml" -> use BoT-SORT
        persist=True,
        stream=True,           # memory-efficient, processes frame by frame
        verbose=False,
        device=device,         # None -> auto, "0" -> GPU 0, "cpu" -> CPU
        imgsz=imgsz,           # inference resolution (smaller = faster)
        half=half,              # FP16 (only effective on GPU, speeds things up)
    )

    frame_idx = 0
    start_time = time.time()

    for result in results_generator:
        frame = result.orig_img.copy()

        boxes = result.boxes
        current_frame_count = 0

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()

            current_frame_count = len(ids)

            for box, track_id, score, cls_id in zip(xyxy, ids, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                unique_ids.add(track_id)

                label_name = model.names.get(cls_id, "fish") if hasattr(model, "names") else "fish"

                # Track unique IDs per class
                class_names[cls_id] = label_name
                class_unique_ids.setdefault(cls_id, set()).add(track_id)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} {label_name} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Top info panel: current frame fish count + running total of unique fish
        info_text_1 = f"Current fish count: {current_frame_count}"
        info_text_2 = f"Total unique fish: {len(unique_ids)}"

        cv2.rectangle(frame, (0, 0), (330, 60), (0, 0, 0), -1)
        cv2.putText(frame, info_text_1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, info_text_2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        writer.write(frame)

        if show:
            cv2.imshow("Fish Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

        # --- Progress is printed to the terminal ---
        elapsed = time.time() - start_time
        proc_fps = frame_idx / elapsed if elapsed > 0 else 0.0

        if total_frames:
            percent = (frame_idx / total_frames) * 100
            remaining = (total_frames - frame_idx) / proc_fps if proc_fps > 0 else 0
            bar_len = 30
            filled = int(bar_len * frame_idx / total_frames)
            bar = "#" * filled + "-" * (bar_len - filled)
            progress_line = (
                f"\r[{bar}] {percent:5.1f}% "
                f"({frame_idx}/{total_frames} frames) "
                f"| {proc_fps:5.1f} fps | "
                f"Total fish: {len(unique_ids)} | "
                f"ETA: {remaining:5.1f}s"
            )
        else:
            # Fallback if total frame count could not be read (can happen with some video formats)
            progress_line = (
                f"\rProcessed frames: {frame_idx} | "
                f"{proc_fps:5.1f} fps | "
                f"Total fish: {len(unique_ids)}"
            )

        sys.stdout.write(progress_line)
        sys.stdout.flush()

    sys.stdout.write("\n")  # move to a new line after the progress bar

    writer.release()
    if show:
        cv2.destroyAllWindows()

    print(f"[DONE] Frames processed: {frame_idx}")
    print(f"[DONE] Total unique fish count: {len(unique_ids)}")
    print(f"[DONE] Output video saved to: {output_path}")

    # --- Per-species summary printed to the terminal ---
    print("\n[PER-SPECIES SUMMARY]")
    print(f"{'Species':<25}{'Unique Count':>15}")
    print("-" * 40)
    for cls_id, id_set in sorted(class_unique_ids.items(), key=lambda x: class_names[x[0]]):
        print(f"{class_names[cls_id]:<25}{len(id_set):>15}")

    # --- Per-species summary saved to CSV ---
    with open(csv_output, mode="w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["species", "unique_count"])
        for cls_id, id_set in sorted(class_unique_ids.items(), key=lambda x: class_names[x[0]]):
            csv_writer.writerow([class_names[cls_id], len(id_set)])
        csv_writer.writerow(["TOTAL", len(unique_ids)])

    print(f"\n[DONE] Per-species summary saved to CSV: {csv_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fish detection and tracking with BoT-SORT")
    parser.add_argument("--model", type=str, required=True, help="YOLO model file (e.g. best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Input video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IOU threshold")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config (botsort.yaml)")
    parser.add_argument("--show", action="store_true", help="Show the video window while processing")
    parser.add_argument("--device", type=str, default=None,
                         help="Device to run on: '0' (GPU 0), 'cpu', '0,1' (multi-GPU). Auto-selected if omitted.")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference resolution (smaller = higher FPS, e.g. 480, 416)")
    parser.add_argument("--half", action="store_true",
                         help="Use FP16 precision (only effective on GPU, speeds up inference)")
    parser.add_argument("--csv", type=str, default="detections_summary.csv",
                         help="CSV file to save the per-species unique count summary")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tracking(
        model_path=args.model,
        source_path=args.source,
        output_path=args.output,
        conf=args.conf,
        iou=args.iou,
        tracker_cfg=args.tracker,
        show=args.show,
        device=args.device,
        imgsz=args.imgsz,
        half=args.half,
        csv_output=args.csv,
    )
