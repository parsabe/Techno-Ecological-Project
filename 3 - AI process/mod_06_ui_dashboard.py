import time
import cv2
import numpy as np
from mod_04_vision_engine import fit_text_to_width

class ThemeManager:
    """
    Manages full-application UI color palettes and styling tokens for Cyber Dark Mode vs Modern Light Mode.
    """
    def __init__(self, mode="DARK"):
        self.mode = mode
        self.palettes = {
            "DARK": {
                "canvas_bg": (24, 18, 15),       # #0F172A BGR Deep Slate Navy
                "card_bg": (45, 30, 20),         # #141E2D BGR Slate Glass Container
                "card_border": (105, 85, 71),     # Steel Slate Border
                "card_border_glow": (244, 208, 6),# Neon Cyan Glow
                "title_text": (244, 208, 63),    # Electric Cyan
                "accent_text": (63, 208, 244),   # Electric Gold / Turquoise
                "body_text": (240, 232, 226),     # Soft Silver White
                "sub_text": (184, 163, 148),      # Muted Slate Gray
                "btn_bg": (45, 30, 20),          # Button Background
                "btn_hover": (85, 65, 51),       # Button Hover
                "btn_border": (212, 182, 6),     # Neon Cyan Button Border
                "btn_text": (244, 208, 63)       # Cyan Button Text
            },
            "LIGHT": {
                "canvas_bg": (255, 255, 255),
                "card_bg": (250, 250, 252),
                "card_border": (204, 199, 199),
                "card_border_glow": (255, 122, 0),
                "title_text": (255, 122, 0),
                "accent_text": (0, 149, 255),
                "body_text": (31, 29, 29),
                "sub_text": (142, 142, 147),
                "btn_bg": (235, 233, 233),
                "btn_hover": (245, 213, 229),
                "btn_border": (204, 199, 199),
                "btn_text": (31, 29, 29)
            }
        }

    def toggle(self):
        self.mode = "LIGHT" if self.mode == "DARK" else "DARK"
        return self.mode

    def get(self, token):
        return self.palettes.get(self.mode, self.palettes["DARK"]).get(token, (255, 255, 255))

theme_mgr = ThemeManager(mode="DARK")

# --- INTERACTIVE CONTROL BUTTON CLASS ---
class ControlButton:
    def __init__(self, x, y, w, h, label, callback_id, color=(255, 122, 0)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label
        self.callback_id = callback_id
        self.color = color

    def contains(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h

    def draw(self, img, is_hovered=False, active_theme=None):
        tm = active_theme if active_theme is not None else theme_mgr
        roi = img[self.y:self.y+self.h, self.x:self.x+self.w]
        if roi.shape[0] == self.h and roi.shape[1] == self.w:
            bg = np.zeros_like(roi, dtype=np.uint8)
            bg_col = tm.get("btn_hover") if is_hovered else tm.get("btn_bg")
            bg[:] = bg_col
            blended = cv2.addWeighted(roi, 0.15, bg, 0.85, 0)
            img[self.y:self.y+self.h, self.x:self.x+self.w] = blended

        border_color = (255, 0, 255) if is_hovered else tm.get("btn_border")
        cv2.rectangle(img, (self.x, self.y), (self.x+self.w, self.y+self.h), border_color, 1)

        lbl_text = fit_text_to_width(self.label, max_pixel_width=self.w - 10, font_scale=0.36)
        t_size = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)[0]
        tx = self.x + (self.w - t_size[0]) // 2
        ty = self.y + (self.h + t_size[1]) // 2
        txt_color = (255, 255, 255) if is_hovered else tm.get("btn_text")
        cv2.putText(img, lbl_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.36, txt_color, 1, cv2.LINE_AA)

# --- HUD NOTIFICATIONS ENGINE ---
class HUDNotificationEngine:
    def __init__(self):
        self.notifications = []

    def add(self, text, color=(255, 122, 0), duration=2.5):
        # Strip unicode emojis to prevent ???? in OpenCV putText
        clean_text = text.encode('ascii', 'ignore').decode('ascii').strip()
        if not clean_text:
            clean_text = text
        self.notifications.append({'text': clean_text, 'color': color, 'expiry': time.time() + duration})

    def draw(self, img):
        curr_t = time.time()
        self.notifications = [n for n in self.notifications if n['expiry'] > curr_t]
        y_offset = 25
        for n in self.notifications[-4:]:
            txt = fit_text_to_width(n['text'], max_pixel_width=780, font_scale=0.40)
            cv2.rectangle(img, (15, y_offset - 16), (15 + len(txt)*8 + 15, y_offset + 6), (255, 255, 255), -1)
            cv2.rectangle(img, (15, y_offset - 16), (15 + len(txt)*8 + 15, y_offset + 6), n['color'], 1)
            cv2.putText(img, txt, (22, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.40, n['color'], 1, cv2.LINE_AA)
            y_offset += 28

hud_notifs = HUDNotificationEngine()

def clean_text_for_opencv(text):
    if not text:
        return ""
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue', 'ß': 'ss',
        '’': "'", '‘': "'", '“': '"', '”': '"', '–': '-', '—': '-', '…': '...'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('ascii', 'ignore').decode('ascii')

def get_wrapped_text_lines(text, max_w=340, font_scale=0.35):
    font = cv2.FONT_HERSHEY_SIMPLEX
    clean_str = clean_text_for_opencv(text)
    words = clean_str.split(' ')
    lines = []
    curr_line = ""
    for w in words:
        if not w:
            continue
        test_line = curr_line + (" " if curr_line else "") + w
        w_px = cv2.getTextSize(test_line, font, font_scale, 1)[0][0]
        if w_px <= max_w:
            curr_line = test_line
        else:
            if curr_line:
                lines.append(curr_line)
            curr_line = w
    if curr_line:
        lines.append(curr_line)
    return lines

def draw_wrapped_text(panel, text, start_x, start_y, max_w=340, max_lines=12, font_scale=0.35, text_color=(31, 29, 29)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = get_wrapped_text_lines(text, max_w=max_w, font_scale=font_scale)
    line_h = int(18 * (font_scale / 0.35))
    
    for i, line_str in enumerate(lines[:max_lines]):
        ly = start_y + i * line_h
        if ly + line_h <= panel.shape[0]:
            cv2.putText(panel, line_str, (start_x, ly), font, font_scale, text_color, 1, cv2.LINE_AA)
            
    return start_y + len(lines[:max_lines]) * line_h

def display_on_screen_data_assimilation_prompt(window_name, canvas_w, canvas_h, video_path):
    """
    Renders a crisp, highly legible San Francisco Light Mode dialog card directly ON THE SCREEN.
    Uses thin, non-bold fonts (thickness=1) and clean text without emojis.
    """
    import os
    selected_choice = [None]
    
    def prompt_mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            btn_yes = param['btn_yes']
            btn_no = param['btn_no']
            if btn_yes['x'] <= x <= btn_yes['x'] + btn_yes['w'] and btn_yes['y'] <= y <= btn_yes['y'] + btn_yes['h']:
                selected_choice[0] = True
            elif btn_no['x'] <= x <= btn_no['x'] + btn_no['w'] and btn_no['y'] <= y <= btn_no['y'] + btn_no['h']:
                selected_choice[0] = False

    card_w, card_h = 720, 360
    card_x = (canvas_w - card_w) // 2
    card_y = (canvas_h - card_h) // 2

    btn_w, btn_h = 280, 48
    btn_yes_x = card_x + 50
    btn_no_x = card_x + card_w - 50 - btn_w
    btn_y = card_y + 270

    btn_yes = {'x': btn_yes_x, 'y': btn_y, 'w': btn_w, 'h': btn_h}
    btn_no = {'x': btn_no_x, 'y': btn_y, 'w': btn_w, 'h': btn_h}

    param = {'btn_yes': btn_yes, 'btn_no': btn_no}
    cv2.setMouseCallback(window_name, prompt_mouse_cb, param=param)

    raw_basename = os.path.basename(video_path)
    clean_basename = raw_basename.encode('ascii', 'ignore').decode('ascii').strip()
    if not clean_basename:
        clean_basename = raw_basename

    while selected_choice[0] is None:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (247, 245, 245)

        # Draw outer card box & shadow
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (229, 235, 234), 1)

        # Header Title (Thin, Crisp, Non-Bold)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 55), (242, 242, 247), -1)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 55), (255, 122, 0), 1)
        cv2.putText(canvas, "AQUAPULSE DATA ASSIMILATION & ANALYSIS PROMPT", (card_x + 35, card_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 122, 0), 1, cv2.LINE_AA)

        # Subtitle & Video Target (Thin, Crisp, Non-Bold)
        v_str = fit_text_to_width(f"Target Video: {clean_basename}", max_pixel_width=card_w - 70, font_scale=0.40)
        cv2.putText(canvas, v_str, (card_x + 35, card_y + 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (31, 29, 29), 1, cv2.LINE_AA)

        cv2.putText(canvas, "Data Assimilation (Stochastic Lotka-Volterra EnKF) & Biological Census is Active.",
                    (card_x + 35, card_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 58, 58), 1, cv2.LINE_AA)

        cv2.putText(canvas, "Do you want to save all per-video analysis reports, plots, and CSV results?",
                    (card_x + 35, card_y + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (31, 29, 29), 1, cv2.LINE_AA)

        cv2.putText(canvas, "- Click [Y] YES to create a unique analysis session folder with reports & plots.",
                    (card_x + 45, card_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (142, 142, 147), 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Click [N] NO to run real-time telemetry on-screen without saving files.",
                    (card_x + 45, card_y + 218), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (142, 142, 147), 1, cv2.LINE_AA)

        # Button YES (Thin, Crisp Text)
        cv2.rectangle(canvas, (btn_yes_x, btn_y), (btn_yes_x + btn_w, btn_y + btn_h), (89, 199, 52), -1)
        cv2.rectangle(canvas, (btn_yes_x, btn_y), (btn_yes_x + btn_w, btn_y + btn_h), (255, 255, 255), 1)
        cv2.putText(canvas, "[Y] YES - SAVE ANALYSIS", (btn_yes_x + 22, btn_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Button NO (Thin, Crisp Text)
        cv2.rectangle(canvas, (btn_no_x, btn_y), (btn_no_x + btn_w, btn_y + btn_h), (48, 59, 255), -1)
        cv2.rectangle(canvas, (btn_no_x, btn_y), (btn_no_x + btn_w, btn_y + btn_h), (255, 255, 255), 1)
        cv2.putText(canvas, "[N] NO - SKIP SAVING", (btn_no_x + 32, btn_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in [ord('y'), ord('Y'), 13]:
            selected_choice[0] = True
        elif key in [ord('n'), ord('N'), 27]:
            selected_choice[0] = False

    return selected_choice[0]

def display_dynamic_loading_screen(window_name, canvas_w, canvas_h, model_name="best.pt", duration=5.0):
    """
    Displays dynamic 5.0-second San Francisco Light Mode loading screen with spinning ring and progress bar.
    Uses thin, ultra-legible fonts (thickness=1) and clean ASCII text.
    """
    print("Loading AquaPulse 5-second dynamic loading screen...")
    loading_start_time = time.time()
    
    while time.time() - loading_start_time < duration:
        elapsed = time.time() - loading_start_time
        progress = min(1.0, elapsed / duration)
        
        load_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        load_canvas[:] = (247, 245, 245)
        
        card_w, card_h = 680, 360
        card_x = (canvas_w - card_w) // 2
        card_y = (canvas_h - card_h) // 2
        
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (229, 235, 234), 1)
        
        cv2.putText(load_canvas, "AQUAPULSE NEURAL VISION TRANSFORMER", (card_x + 45, card_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (31, 29, 29), 1, cv2.LINE_AA)
        cv2.putText(load_canvas, "SAN FRANCISCO LIGHT ENGINE | EnKF ASSIMILATION PIPELINE", (card_x + 45, card_y + 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (142, 142, 147), 1, cv2.LINE_AA)
        
        ring_cx, ring_cy = card_x + card_w // 2, card_y + 160
        angle = (elapsed * 360.0) % 360.0
        cv2.circle(load_canvas, (ring_cx, ring_cy), 32, (234, 235, 229), 2)
        cv2.ellipse(load_canvas, (ring_cx, ring_cy), (32, 32), angle, 0, 90, (255, 122, 0), 3)
        
        bar_x, bar_y = card_x + 60, card_y + 240
        bar_w, bar_h = card_w - 120, 16
        cv2.rectangle(load_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (235, 233, 233), -1)
        cv2.rectangle(load_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (204, 199, 199), 1)
        
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(load_canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (255, 122, 0), -1)
        
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(load_canvas, pct_text, (bar_x + bar_w + 15, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 122, 0), 1, cv2.LINE_AA)
        
        if progress < 0.25:
            msg = "Initializing AquaPulse San Francisco Light Engine..."
        elif progress < 0.50:
            msg = f"Loading YOLO Neural Vision Weights ({model_name})..."
        elif progress < 0.75:
            msg = "Synchronizing Stochastic EnKF Data Assimilation Pipeline..."
        elif progress < 0.95:
            msg = "Calibrating 4-Pane Dashboard & Matplotlib Analysis Canvas..."
        else:
            msg = "AquaPulse Vision System Ready."
            
        cv2.putText(load_canvas, msg, (bar_x, bar_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 58, 58), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, load_canvas)
        cv2.waitKey(20)

def display_analysis_export_loading_screen(window_name, canvas_w, canvas_h, session_info, export_callback):
    """
    Displays an elegant San Francisco Light Mode End-of-Session Loading Screen.
    Renders real-time stage updates for CSV, 20 Plots, Ollama Analysis, and PDF generation,
    then holds the completion screen for 7 seconds with a countdown timer before closing.
    """
    import os
    import threading
    print("⏳ Displaying AquaPulse End-of-Session Analysis Export Loading Screen...")
    
    session_name = session_info.get("session_name", "Video Analysis Session")
    
    steps = [
        (0.20, "[1/4] Exporting Biological Species CSV Census Report..."),
        (0.50, "[2/4] Generating 20 Dynamic Scientific Telemetry Plots..."),
        (0.80, "[3/4] Querying Ollama LLM for Academic Research Synthesis..."),
        (0.98, "[4/4] Compiling Academic Paper PDF & LaTeX Source via pdflatex...")
    ]
    
    export_done = [False]
    
    def _export_worker():
        try:
            export_callback()
        except Exception as e:
            print(f"Notice during export: {e}")
        finally:
            export_done[0] = True
            
    t = threading.Thread(target=_export_worker, daemon=True)
    t.start()
    
    start_time = time.time()
    
    # 1. Active Export Progress Loop
    while not export_done[0]:
        elapsed = time.time() - start_time
        progress = min(0.96, elapsed / 10.0)
            
        load_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        load_canvas[:] = (247, 245, 245)
        
        card_w, card_h = 740, 380
        card_x = (canvas_w - card_w) // 2
        card_y = (canvas_h - card_h) // 2
        
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (229, 235, 234), 1)
        
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + 55), (242, 242, 247), -1)
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + 55), (0, 149, 255), 1)
        cv2.putText(load_canvas, "AQUAPULSE SESSION ANALYSIS & REPORT GENERATION", (card_x + 35, card_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 149, 255), 1, cv2.LINE_AA)
        
        v_str = fit_text_to_width(f"Session: {session_name}", max_pixel_width=card_w - 70, font_scale=0.38)
        cv2.putText(load_canvas, v_str, (card_x + 35, card_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (31, 29, 29), 1, cv2.LINE_AA)
        
        ring_cx, ring_cy = card_x + card_w // 2, card_y + 165
        angle = (elapsed * 360.0) % 360.0
        cv2.circle(load_canvas, (ring_cx, ring_cy), 32, (234, 235, 229), 2)
        cv2.ellipse(load_canvas, (ring_cx, ring_cy), (32, 32), angle, 0, 90, (0, 149, 255), 3)
        
        bar_x, bar_y = card_x + 60, card_y + 250
        bar_w, bar_h = card_w - 120, 16
        cv2.rectangle(load_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (235, 233, 233), -1)
        cv2.rectangle(load_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (204, 199, 199), 1)
        
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            cv2.rectangle(load_canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 149, 255), -1)
        
        pct_text = f"{int(progress * 100)}%"
        cv2.putText(load_canvas, pct_text, (bar_x + bar_w + 15, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 149, 255), 1, cv2.LINE_AA)
        
        curr_msg = "[4/4] Compiling Academic Paper PDF & LaTeX Source via pdflatex..."
        for step_p, step_msg in steps:
            if progress <= step_p:
                curr_msg = step_msg
                break
            
        cv2.putText(load_canvas, curr_msg, (bar_x, bar_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 58, 58), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, load_canvas)
        cv2.waitKey(30)

    # 2. Completion 7-Second Countdown Loop
    hold_start = time.time()
    hold_duration = 7.0
    
    while True:
        remaining = max(0, int(np.ceil(hold_duration - (time.time() - hold_start))))
        if time.time() - hold_start >= hold_duration:
            break

        load_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        load_canvas[:] = (247, 245, 245)
        
        card_w, card_h = 740, 380
        card_x = (canvas_w - card_w) // 2
        card_y = (canvas_h - card_h) // 2
        
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (52, 199, 89), 2)
        
        cv2.rectangle(load_canvas, (card_x, card_y), (card_x + card_w, card_y + 55), (52, 199, 89), -1)
        cv2.putText(load_canvas, "ANALYSIS & ACADEMIC PDF REPORT GENERATION COMPLETE", (card_x + 35, card_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        
        v_str = fit_text_to_width(f"Session Folder: {session_name}", max_pixel_width=card_w - 70, font_scale=0.38)
        cv2.putText(load_canvas, v_str, (card_x + 35, card_y + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (31, 29, 29), 1, cv2.LINE_AA)
        
        # Big Green Checkmark Ring
        ring_cx, ring_cy = card_x + card_w // 2, card_y + 165
        cv2.circle(load_canvas, (ring_cx, ring_cy), 36, (52, 199, 89), -1)
        cv2.putText(load_canvas, "OK", (ring_cx - 14, ring_cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 100% Progress Bar
        bar_x, bar_y = card_x + 60, card_y + 250
        bar_w, bar_h = card_w - 120, 16
        cv2.rectangle(load_canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (52, 199, 89), -1)
        cv2.putText(load_canvas, "100%", (bar_x + bar_w + 15, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (52, 199, 89), 1, cv2.LINE_AA)
        
        countdown_msg = f"All artifacts, CSV, 20 plots & PDF saved! Closing window in {remaining}s..."
        cv2.putText(load_canvas, countdown_msg, (bar_x - 10, bar_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (31, 29, 29), 1, cv2.LINE_AA)
        
        cv2.imshow(window_name, load_canvas)
        cv2.waitKey(100)
