# gestureSend.py — Option 1: Right hand = Increase, Left hand = Decrease
# Finger-count mapping (UPDATED - 4 FINGERS ONLY, NO THUMB):
#   0 (Fist)    -> LOOP        (/shimon/loop/add      +/- LOOP_STEP)
#   1 (Point)   -> TEMPERATURE (/shimon/temperature/add +/- TEMP_STEP)
#   2 (V-Sign)  -> VELOCITY    (/shimon/velocity/add  +/- VEL_STEP)
#   3 (W-Shape) -> TEMPO       (/shimon/tempo/add     +/- TEMPO_STEP)  <-- [SWAPPED]
#   4 (Open)    -> STOP LOOP   (/shimon/loop/stop     1)               <-- [SWAPPED]
#
# Keyboard Control:
#   'q'         -> QUIT PROGRAM
#
# Stability (debounce) and 500ms cooldown per hand included.

import time
from collections import deque

import cv2
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient

# =========================
# OSC / STEP CONFIG
# =========================
UDP_IP    = "127.0.0.1"   # IP of the OSC server in main.py
UDP_PORT  = 9000          # Must match the port of the OSC server in main.py

LOOP_STEP = 1             # Integer increment/decrement for loop count
VEL_STEP  = 0.05          # Increment/decrement for velocity scale
TEMPO_STEP= 0.1           # Increment/decrement for BPM scale
TEMP_STEP = 0.05          # Increment/decrement for temperature

COOLDOWN_SEC  = 0.50      # 500ms cooldown per hand
STABLE_FRAMES = 4         # Number of frames required for stability

# =========================
# MediaPipe Hands
# =========================
mp_hands   = mp.solutions.hands
mp_draw    = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =========================
# Helpers
# =========================
def finger_extended(lms, tip_idx, pip_idx):
    """Returns True if Tip is above PIP (y is smaller)."""
    tip = lms[tip_idx]; pip = lms[pip_idx]
    return tip.y < pip.y

def count_extended_total(lms):
    """Counts extended fingers (0~4), EXCLUDING the thumb."""
    m = mp_hands.HandLandmark
    cnt = 0
    cnt += 1 if finger_extended(lms, m.INDEX_FINGER_TIP,  m.INDEX_FINGER_PIP)  else 0
    cnt += 1 if finger_extended(lms, m.MIDDLE_FINGER_TIP, m.MIDDLE_FINGER_PIP) else 0
    cnt += 1 if finger_extended(lms, m.RING_FINGER_TIP,   m.RING_FINGER_PIP)   else 0
    cnt += 1 if finger_extended(lms, m.PINKY_TIP,         m.PINKY_PIP)         else 0
    return cnt

def select_param_by_count(total_extended):
    """Mapping: 0=LOOP, 1=TEMP, 2=VEL, 3=TEMPO, 4=STOP."""
    if total_extended == 0:
        return "LOOP"  # Fist
    if total_extended == 1:
        return "TEMP"  # One finger
    if total_extended == 2:
        return "VEL"   # V-sign
    if total_extended == 3:
        return "TEMPO" # [SWAPPED] 3 Fingers (W-shape) -> TEMPO
    if total_extended == 4:
        return "STOP"  # [SWAPPED] Open palm (4 fingers) -> STOP
    return "NONE"

# =========================
# Main
# =========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERR] Cannot open camera")
        return

    client = SimpleUDPClient(UDP_IP, UDP_PORT)

    # Per-hand state: stability buffer & last sent time
    per_hand_buf = { "Left": deque(maxlen=STABLE_FRAMES),
                     "Right": deque(maxlen=STABLE_FRAMES) }
    per_hand_stable = { "Left": "NONE", "Right": "NONE" }
    last_sent_time  = { "Left": 0.0,    "Right": 0.0 }

    print("Gesture Controller Started.")
    print("Gestures (No Thumb): 0=Loop, 1=Temp, 2=Vel, 3=Tempo, 4=STOP")
    print("Press 'q' to QUIT.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Flip image horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            result = hands.process(image)
            image.flags.writeable = True
            
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            now = time.time()
            hud_lines = []
            current_counts = {"Left": -1, "Right": -1} # For debugging
            
            # Track hands detected in the current frame
            detected_hands_in_frame = set()

            if result.multi_hand_landmarks and result.multi_handedness:
                # 1. Update status for visible hands
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    label = result.multi_handedness[i].classification[0].label
                    lms = hand_landmarks.landmark
                    
                    detected_hands_in_frame.add(label) # Mark as visible

                    total_ext = count_extended_total(lms)
                    current_counts[label] = total_ext
                    target = select_param_by_count(total_ext)

                    per_hand_buf[label].append(target)
                    # Stability check: all frames in buffer must match
                    if len(per_hand_buf[label]) == STABLE_FRAMES and len(set(per_hand_buf[label])) == 1:
                        per_hand_stable[label] = per_hand_buf[label][0]
                    else:
                        per_hand_stable[label] = "NONE"

                    mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. Ghost hand removal logic
            for hand in ("Left", "Right"):
                if hand not in detected_hands_in_frame:
                    per_hand_buf[hand].clear()
                    per_hand_stable[hand] = "NONE"

            # 3. Send OSC messages
            for label in ("Right", "Left"):
                target = per_hand_stable[label]
                if target == "NONE":
                    continue
                
                # 3-1. STOP Logic (Trigger immediately, separate visual feedback)
                if target == "STOP":
                    if now - last_sent_time[label] > 1.0: # 1 second cooldown for STOP
                        client.send_message("/shimon/loop/stop", 1)
                        arm_client = SimpleUDPClient(UDP_IP, 6000)
                        arm_client.send_message("/shimon/loop/stop", 1)
                        print(f">> STOP SIGNAL SENT via {label} Hand (4 Fingers)")
                        last_sent_time[label] = now
                        # Visual effect for STOP
                        cv2.putText(image, "!!! STOP !!!", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
                    continue

                # 3-2. Parameter Control Logic
                if now - last_sent_time[label] < COOLDOWN_SEC:
                    continue

                if label == "Right": # Increase
                    if target == "LOOP":
                        client.send_message("/shimon/loop/add", int(+LOOP_STEP))
                        hud_lines.append("RIGHT: LOOP +")
                    elif target == "VEL":
                        client.send_message("/shimon/velocity/add", float(+VEL_STEP))
                        hud_lines.append("RIGHT: VEL +")
                    elif target == "TEMPO":
                        client.send_message("/shimon/tempo/add", float(+TEMPO_STEP))
                        hud_lines.append("RIGHT: TEMPO +")
                    elif target == "TEMP":
                        client.send_message("/shimon/temperature/add", float(+TEMP_STEP))
                        hud_lines.append("RIGHT: TEMP +")
                    last_sent_time[label] = now

                elif label == "Left": # Decrease
                    if target == "LOOP":
                        client.send_message("/shimon/loop/add", int(-LOOP_STEP))
                        hud_lines.append("LEFT: LOOP -")
                    elif target == "VEL":
                        client.send_message("/shimon/velocity/add", float(-VEL_STEP))
                        hud_lines.append("LEFT: VEL -")
                    elif target == "TEMPO":
                        client.send_message("/shimon/tempo/add", float(-TEMPO_STEP))
                        hud_lines.append("LEFT: TEMPO -")
                    elif target == "TEMP":
                        client.send_message("/shimon/temperature/add", float(-TEMP_STEP))
                        hud_lines.append("LEFT: TEMP -")
                    last_sent_time[label] = now

            # HUD Display
            hud = " | ".join(hud_lines) if hud_lines else "..."
            cv2.putText(image, hud, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(image, "R=Inc, L=Dec | 0:Loop, 1:Temp, 2:Vel, 3:Tempo, 4:STOP", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            
            debug_text = f"L: {current_counts['Left']}  R: {current_counts['Right']}"
            cv2.putText(image, debug_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2.imshow("Gesture -> OSC (Both hands count | R:+ L:-)", image)
            
            # Keyboard Input Handling
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()