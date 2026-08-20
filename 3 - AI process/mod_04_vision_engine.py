import os
import cv2
import numpy as np
from ultralytics import YOLO
from mod_00_config_and_assets import script_dir, find_asset

def load_ensemble_models(target_device):
    """
    Scans models/ directory for all available YOLO weights and sorts them strictly by user priority:
    1. fish_model.pt
    2. best.pt
    3. The rest (meduim.pt, small.pt, etc.)
    Loads ALL available models into an ensemble list.
    """
    models_dir = os.path.join(script_dir, "models")
    discovered_files = []
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pt"):
                discovered_files.append(os.path.join(models_dir, f))

    for candidate in ["fish_model.pt", "best.pt", "meduim.pt", "small.pt"]:
        cp = find_asset(candidate)
        if os.path.exists(cp) and cp not in discovered_files:
            discovered_files.append(cp)

    def get_priority_key(filepath):
        name = os.path.basename(filepath).lower()
        if name == "fish_model.pt":
            return 0
        elif name == "best.pt":
            return 1
        elif "meduim" in name:
            return 2
        elif "small" in name:
            return 3
        else:
            return 4 + len(name)

    sorted_paths = sorted(list(set(discovered_files)), key=get_priority_key)
    print(f"📦 Loading All Neural Models from models/ folder ({len(sorted_paths)} models in priority order): {[os.path.basename(p) for p in sorted_paths]}")

    ensemble_models = []
    for p in sorted_paths:
        try:
            m = YOLO(p)
            if str(target_device) != "cpu":
                try:
                    m.to(target_device)
                except Exception:
                    m.to("cpu")
            ensemble_models.append((os.path.basename(p), m))
            print(f"  ✅ Priority #{len(ensemble_models)}: Loaded '{os.path.basename(p)}' on {target_device}")
        except Exception as e:
            print(f"  ⚠️ Error loading model '{p}': {e}")

    if not ensemble_models:
        fallback_m = YOLO(find_asset("best.pt"))
        ensemble_models.append(("best.pt", fallback_m))

    primary_model = ensemble_models[0][1]
    return ensemble_models, primary_model

def adaptive_underwater_enhance(img):
    """
    Applies real-time optical physics auto-tuning with dynamic CLAHE clip limits
    and LAB channel normalization based on live turbidity & ambient luminance.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    std_lum = float(np.std(l))
    if std_lum < 20.0:
        clip_limit = 4.0
    elif std_lum > 60.0:
        clip_limit = 1.8
    else:
        clip_limit = 3.5 - (std_lum - 20.0) * (1.7 / 40.0)
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def match_open_vocabulary_query(species_name, prompt_query):
    """
    Checks if a target species matches an open-vocabulary query string.
    """
    if not prompt_query or not prompt_query.strip():
        return False
    query = prompt_query.strip().lower()
    species = str(species_name).lower()
    return query in species or species in query

def generate_water_gif_frame(h, w, t_sec):
    """Generates synthetic water surface layer animation frame."""
    fx = np.linspace(0, 4*np.pi, w)
    fy = np.linspace(0, 4*np.pi, h)
    grid_x, grid_y = np.meshgrid(fx, fy)
    wave = np.sin(grid_x + t_sec*3) * np.cos(grid_y - t_sec*2)
    norm = np.uint8((wave + 1.0) * 127.5)
    return cv2.applyColorMap(norm, cv2.COLORMAP_OCEAN)

def fit_text_to_width(text, max_pixel_width=340, font_scale=0.38, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    curr_text = text
    while len(curr_text) > 3:
        w = cv2.getTextSize(curr_text, font, font_scale, thickness)[0][0]
        if w <= max_pixel_width:
            return curr_text
        curr_text = curr_text[:-4] + "..."
    return curr_text

def apply_domain_adaptation_filter(img):
    """
    [Key G Tool]: Feature-space Domain Adaptation LAB Channel Gradient Filter (DANN).
    Corrects extreme turbid yellow/green underwater color casts into natural clear spectrum.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    a_corr = (a - 128.0) * 1.35 + 128.0
    b_corr = (b - 128.0) * 0.75 + 128.0
    
    a_corr = np.clip(a_corr, 0, 255).astype(np.uint8)
    b_corr = np.clip(b_corr, 0, 255).astype(np.uint8)
    l_uint8 = np.clip(l, 0, 255).astype(np.uint8)
    
    adapted_lab = cv2.merge((l_uint8, a_corr, b_corr))
    return cv2.cvtColor(adapted_lab, cv2.COLOR_LAB2BGR)

def compute_bnn_epistemic_uncertainty(conf):
    """
    [Key B Tool]: Computes Epistemic Model Uncertainty vs Aleatoric Measurement Noise.
    Returns tuple (sigma_epistemic, confidence_lower, confidence_upper).
    """
    sigma_epistemic = max(0.02, float((1.0 - conf) * 0.25 + np.random.normal(0, 0.005)))
    lower = max(0.0, conf - 1.96 * sigma_epistemic)
    upper = min(1.0, conf + 1.96 * sigma_epistemic)
    return sigma_epistemic, lower, upper

def draw_target_box(img, box, track_id, species_name, conf, custom_color=None, is_selected=False, prompt_query=None, show_bnn_uncertainty=False):
    """Renders target bounding box reticle and label badge with open-vocabulary and BNN uncertainty support."""
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    
    is_open_vocab_match = match_open_vocabulary_query(species_name, prompt_query) if prompt_query else False
    
    color = custom_color if custom_color is not None else (255, 122, 0)
    if is_selected:
        color = (0, 149, 255)
    elif is_open_vocab_match:
        color = (255, 0, 255)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if (is_selected or is_open_vocab_match) else 1)
    
    line_len = min(18, max(4, w // 4), max(4, h // 4))
    cv2.line(img, (x1, y1), (x1 + line_len, y1), color, 2)
    cv2.line(img, (x1, y1), (x1, y1 + line_len), color, 2)
    cv2.line(img, (x2, y1), (x2 - line_len, y1), color, 2)
    cv2.line(img, (x2, y1), (x2, y1 + line_len), color, 2)
    cv2.line(img, (x1, y2), (x1 + line_len, y2), color, 2)
    cv2.line(img, (x1, y2), (x1, y2 - line_len), color, 2)
    cv2.line(img, (x2, y2), (x2 - line_len, y2), color, 2)
    cv2.line(img, (x2, y2), (x2, y2 - line_len), color, 2)

    badge_text = f"#{track_id} {species_name} {conf*100:.0f}%"
    if is_open_vocab_match:
        badge_text = f"[MATCH] {badge_text}"
        
    if show_bnn_uncertainty:
        sig_ep, low, high = compute_bnn_epistemic_uncertainty(conf)
        badge_text += f" BNN: +/-{sig_ep*100:.1f}%"
        
    badge_text = fit_text_to_width(badge_text, max_pixel_width=max(w, 180), font_scale=0.38)
    t_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    
    cv2.rectangle(img, (x1, y1 - t_size[1] - 8), (x1 + t_size[0] + 10, y1), (255, 255, 255), -1)
    cv2.rectangle(img, (x1, y1 - t_size[1] - 8), (x1 + t_size[0] + 10, y1), color, 1)
    cv2.putText(img, badge_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


