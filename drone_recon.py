import cv2
import torch
import numpy as np
import time
import datetime
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# ==========================================
# [CONFIG] SYSTEM SETUP
# ==========================================
CAMERA_INDEX = 0  
RECORD_MISSION = True 


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available(): DEVICE = "mps"
MODEL_NAME = "./local_model"

# ADE20K Class Definitions & Labels
SAFE_CLASSES = [3, 9, 6, 13, 29]
THREAT_CLASSES = [0, 1, 4, 7, 12, 19, 20]

CLASS_NAMES = {
    0: "WALL", 1: "BLDG", 4: "TREE", 7: "BED", 
    12: "HUMAN", 19: "CHAIR", 20: "CAR", 
    3: "FLOOR", 9: "GRASS", 6: "ROAD"
}

# Colors
COLOR_LOCK = (0, 0, 255)    # Red
COLOR_SAFE = (0, 255, 0)    # Green
COLOR_CMD  = (0, 255, 255)  # Yellow
COLOR_INFO = (255, 150, 0)  # Blue/Cyan

# ==========================================
# INITIALIZATION
# ==========================================
print(f"[INFO] Sentinel System v3.1 Booting on {DEVICE}...")

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    exit()

# Force AVFoundation for Mac
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ==========================================
# NEW: CAMERA WARM-UP ROUTINE
# ==========================================
print("[INFO] Warming up sensors...")
for i in range(10):
    ret, frame = cap.read()
    if ret:
        print(f"   Sensor Check {i+1}/10: OK")
    else:
        print(f"   Sensor Check {i+1}/10: ...waiting")
    time.sleep(0.1)

# Check if camera is actually alive after warm-up
if not cap.isOpened():
    print("[CRITICAL FAILURE] Camera did not initialize. Check USB/Permissions.")
    exit()

# Initialize Video Writer
writer = None
if RECORD_MISSION:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # MJPG is safer for Mac
    filename = f"mission_log_{int(time.time())}.avi"
    writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    print(f"[INFO] Recording Mission to: {filename}")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_flight_command(center_x, width):
    error = center_x - (width // 2)
    if abs(error) < 60: return "LOCKED", COLOR_SAFE
    elif error < 0: return f"<< LEFT [{abs(error)}]", COLOR_CMD
    else: return f"RIGHT [{abs(error)} >>", COLOR_CMD

def estimate_range(box_height):
    if box_height == 0: return 99.9
    dist = 3000 / box_height 
    return round(dist, 1)

# ==========================================
# MAIN LOOP
# ==========================================
prev_time = 0
error_count = 0 # Keep track of consecutive errors

while True:
    ret, frame = cap.read()
    
    # Safety Logic
    if not ret:
        error_count += 1
        print(f"[WARNING] Dropped Frame {error_count}/10")
        if error_count > 10:
            print("[ERROR] Camera signal lost completely.")
            break
        continue # Try next frame
    
    # Reset error count if we got a good frame
    error_count = 0

    # Resize
    process_frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)

    # 1. AI INFERENCE
    inputs = processor(images=rgb_frame, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    
    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits, size=(480, 640), mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # 2. CREATE HUD OVERLAY
    overlay = process_frame.copy()
    
    # Paint Zones
    threat_mask = np.isin(pred_seg, THREAT_CLASSES).astype(np.uint8) * 255
    safe_mask = np.isin(pred_seg, SAFE_CLASSES).astype(np.uint8) * 255
    
    overlay[threat_mask == 255] = [0, 0, 100] # Red Tint
    overlay[safe_mask == 255] = [0, 100, 0]   # Green Tint
    cv2.addWeighted(overlay, 0.6, process_frame, 0.4, 0, process_frame)

    # 3. TARGETING SYSTEM
    contours, _ = cv2.findContours(threat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_threat = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_threat) > 1500:
            x, y, w, h = cv2.boundingRect(largest_threat)
            cx, cy = x + w//2, y + h//2
            
            # Identify Target
            # Ensure we don't go out of bounds
            cy = min(cy, 479)
            cx = min(cx, 639)
            
            target_id = pred_seg[cy, cx]
            target_name = CLASS_NAMES.get(target_id, "UNK THREAT")
            rng = estimate_range(h)

            # Draw HUD Box
            l = int(w/4)
            cv2.line(process_frame, (x, y), (x+l, y), COLOR_LOCK, 2)
            cv2.line(process_frame, (x, y), (x, y+l), COLOR_LOCK, 2)
            cv2.line(process_frame, (x+w, y), (x+w-l, y), COLOR_LOCK, 2)
            cv2.line(process_frame, (x+w, y), (x+w, y+l), COLOR_LOCK, 2)
            cv2.line(process_frame, (x, y+h), (x+l, y+h), COLOR_LOCK, 2)
            cv2.line(process_frame, (x, y+h), (x, y+h-l), COLOR_LOCK, 2)
            cv2.line(process_frame, (x+w, y+h), (x+w-l, y+h), COLOR_LOCK, 2)
            cv2.line(process_frame, (x+w, y+h), (x+w, y+h-l), COLOR_LOCK, 2)

            # Flight Director
            cmd_text, cmd_color = calculate_flight_command(cx, 640)
            cv2.line(process_frame, (320, 480), (cx, cy), cmd_color, 1)
            cv2.circle(process_frame, (cx, cy), 4, cmd_color, -1)

            # Data
            cv2.putText(process_frame, cmd_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cmd_color, 2)
            cv2.putText(process_frame, f"ID: {target_name}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(process_frame, f"RNG: {rng}m", (x, y+h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_INFO, 1)

    # 4. GLOBAL STATUS DISPLAY
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    
    cv2.putText(process_frame, f"SENTINEL v3.1 | FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if RECORD_MISSION:
        cv2.circle(process_frame, (600, 30), 8, (0, 0, 255), -1) 
        cv2.putText(process_frame, "REC", (550, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(process_frame)

    cv2.imshow("Sentinel Drone HUD", process_frame)
    
    # 5. CONTROLS
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'): 
        ts = datetime.datetime.now().strftime("%H-%M-%S")
        fname = f"intel_snapshot_{ts}.jpg"
        cv2.imwrite(fname, process_frame)
        print(f"[INFO] Snapshot saved: {fname}")

cap.release()
if writer: writer.release()
cv2.destroyAllWindows()
print("[INFO] Mission Complete. Log saved.")