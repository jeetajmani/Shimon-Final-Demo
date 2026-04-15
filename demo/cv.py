import cv2
import time
import threading
import math
import numpy as np
import mediapipe as mp
from collections import deque

tempo_detection_enabled = threading.Event()
eye_contact_enabled = threading.Event()
eye_contact_enabled.set()

SHOW_WINDOW = False
DEBUG_FPS = False

# ── Eye outline landmarks (no iris) ──────────────────────────────────────────
LEFT_EYE_OUTLINE  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                     466, 388, 387, 386, 385, 384, 398]

# ── Tempo nod thresholds ──────────────────────────────────────────────────────
TEMPO_DOWN_THR   = -1.5
TEMPO_UP_THR     = -0.5
TEMPO_MIN_NODS   = 4
TEMPO_WINDOW_SEC = 6.0



# ── Finger-count gesture config ──────────────────────────────────────────────
# Finger-count mapping (4 fingers only, no thumb):
#   0 (Fist)    -> LOOP
#   1 (Point)   -> TEMP
#   2 (V-Sign)  -> VEL
#   3 (W-Shape) -> TEMPO
#   4 (Open)    -> STOP
# Right hand = increase, Left hand = decrease
GESTURE_COOLDOWN_SEC  = 0.50
GESTURE_STABLE_FRAMES = 4
GESTURE_STOP_COOLDOWN = 1.0

# === MediaPipe ===
mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


# ========= FINGER-COUNT GESTURE HELPERS ========= #

def _finger_extended(lms, tip_idx, pip_idx):
    return lms[tip_idx].y < lms[pip_idx].y

def _count_extended_fingers(lms):
    m = mp_hands.HandLandmark
    cnt = 0
    cnt += 1 if _finger_extended(lms, m.INDEX_FINGER_TIP,  m.INDEX_FINGER_PIP)  else 0
    cnt += 1 if _finger_extended(lms, m.MIDDLE_FINGER_TIP, m.MIDDLE_FINGER_PIP) else 0
    cnt += 1 if _finger_extended(lms, m.RING_FINGER_TIP,   m.RING_FINGER_PIP)   else 0
    cnt += 1 if _finger_extended(lms, m.PINKY_TIP,         m.PINKY_PIP)         else 0
    return cnt

def _select_param_by_count(total_extended):
    if total_extended == 0: return "LOOP"
    if total_extended == 1: return "TEMP"
    if total_extended == 2: return "VEL"
    if total_extended == 3: return "TEMPO"
    if total_extended == 4: return "STOP"
    return "NONE"


# ========= FACE UTILITIES ========= #

def get_head_angles(face_landmarks):
    lm        = face_landmarks.landmark
    left_eye  = np.array([lm[33].x,  lm[33].y,  lm[33].z])
    right_eye = np.array([lm[263].x, lm[263].y, lm[263].z])
    chin      = np.array([lm[152].x, lm[152].y, lm[152].z])
    fore      = np.array([lm[10].x,  lm[10].y,  lm[10].z])

    eye_vec  = right_eye - left_eye
    eye_vec /= np.linalg.norm(eye_vec)

    vert_vec  = chin - fore
    vert_vec /= np.linalg.norm(vert_vec)

    yaw   = math.degrees(math.atan2(eye_vec[2],  eye_vec[0]))
    pitch = math.degrees(math.atan2(vert_vec[2], vert_vec[1]))
    return pitch, yaw


def estimate_tempo(nod_times):
    now    = time.time()
    recent = [t for t in nod_times if now - t < TEMPO_WINDOW_SEC]
    if len(recent) < TEMPO_MIN_NODS:
        return None
    intervals = [t2 - t1 for t1, t2 in zip(recent[:-1], recent[1:])]
    return int(60 / np.median(intervals)) if intervals else None


def face_is_looking_at_camera(face_landmarks):
    lm   = face_landmarks.landmark
    nose = lm[1]
    lc   = lm[234]
    rc   = lm[454]
    fore = lm[10]
    chin = lm[152]

    dl = abs(nose.x - lc.x)
    dr = abs(nose.x - rc.x)
    if dr == 0:
        return False
    yaw_ok = 0.5 < dl / dr < 1.5

    denom = chin.y - nose.y
    if denom == 0:
        return False
    pitch_ok = 0.8 < (nose.y - fore.y) / denom < 1.5

    return yaw_ok and pitch_ok


# ── Eye drawing ───────────────────────────────────────────────────────────────
def get_outline_pts(lm, outline_ids, w, h):
    return np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in outline_ids])


def draw_eye(frame, lm, outline_ids, w, h, draw_state):
    if draw_state == "nod":
        outline_col, fill_col = (0, 255, 0),     (0, 60, 0)
    elif draw_state == "armed":
        outline_col, fill_col = (0, 220, 255),   (0, 40, 80)
    elif draw_state == "contact":
        outline_col, fill_col = (180, 255, 180), (20, 50, 20)
    else:
        outline_col, fill_col = (160, 160, 160), (20, 20, 30)

    pts = get_outline_pts(lm, outline_ids, w, h)
    cv2.fillPoly(frame, [pts], fill_col)
    cv2.polylines(frame, [pts], isClosed=True, color=outline_col, thickness=2)


# ========= MAIN LOOP ========== #

def start_gestures_monitor(on_turn_take_callback,
                           on_tempo_callback,
                           on_gesture_callback=None):
    """
    on_turn_take_callback(source)       — fired when a turn-taking gesture is detected.
                                          source is "nod" or "eye_contact".
    on_tempo_callback(bpm)              — fired when a tempo estimate is ready.
    on_gesture_callback(param, direction) — fired when a finger-count gesture is detected.
                                          param is "LOOP", "TEMP", "VEL", "TEMPO", or "STOP".
                                          direction is "+" or "-" (Right=+, Left=-).
                                          For STOP, direction is always "+".
    """

    def camera_loop():
        # ── Tempo nod state ───────────────────────────────────────────────────
        tempo_nodding = False
        nod_times     = []

        # ── Eye contact state ─────────────────────────────────────────────────
        prev_face_looking = False
        state_label       = "Looking for eye contact…"

        # ── Finger-count gesture state ────────────────────────────────────────
        per_hand_buf    = { "Left": deque(maxlen=GESTURE_STABLE_FRAMES),
                            "Right": deque(maxlen=GESTURE_STABLE_FRAMES) }
        per_hand_stable = { "Left": "NONE", "Right": "NONE" }
        gesture_last_sent = { "Left": 0.0, "Right": 0.0 }

        # ── FPS ───────────────────────────────────────────────────────────────
        frame_count     = 0
        last_fps_time   = time.time()
        prev_tempo_mode = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_result = face_mesh.process(rgb)
            hand_result = hands.process(rgb)

            tempo_mode = tempo_detection_enabled.is_set()
            now        = time.time()

            if tempo_mode and not prev_tempo_mode:
                tempo_nodding = False
                nod_times.clear()
                print(">>> ENTER TEMPO MODE (reset nod buffers)")
            prev_tempo_mode = tempo_mode

            if DEBUG_FPS:
                frame_count += 1
                if now - last_fps_time >= 1:
                    print(f"FPS: {frame_count}")
                    frame_count   = 0
                    last_fps_time = now

            # ====== HANDS (finger-count gestures) ============================
            detected_hands = set()
            gesture_hud_lines = []

            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for i, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    cls   = hand_result.multi_handedness[i].classification[0]
                    if cls.score < 0.80:
                        continue  # skip low-confidence detections
                    label = cls.label
                    lms   = hand_landmarks.landmark
                    detected_hands.add(label)

                    # — finger-count gesture —
                    if on_gesture_callback is not None:
                        total_ext = _count_extended_fingers(lms)
                        target    = _select_param_by_count(total_ext)
                        per_hand_buf[label].append(target)
                        if (len(per_hand_buf[label]) == GESTURE_STABLE_FRAMES
                                and len(set(per_hand_buf[label])) == 1):
                            per_hand_stable[label] = per_hand_buf[label][0]
                        else:
                            per_hand_stable[label] = "NONE"

            # ghost-hand removal
            for hand in ("Left", "Right"):
                if hand not in detected_hands:
                    per_hand_buf[hand].clear()
                    per_hand_stable[hand] = "NONE"

            # ====== FINGER-COUNT GESTURE DISPATCH =============================
            if on_gesture_callback is not None:
                for label in ("Right", "Left"):
                    target = per_hand_stable[label]
                    if target == "NONE":
                        continue
                    if target == "STOP":
                        if now - gesture_last_sent[label] > GESTURE_STOP_COOLDOWN:
                            print(f"✋ STOP via {label} hand (4 fingers)")
                            on_gesture_callback("STOP", "+")
                            gesture_last_sent[label] = now
                            gesture_hud_lines.append(f"{label[0]}: STOP")
                        continue
                    if now - gesture_last_sent[label] < GESTURE_COOLDOWN_SEC:
                        continue
                    direction = "+" if label == "Right" else "-"
                    print(f"🖐 {label}: {target} {direction}")
                    on_gesture_callback(target, direction)
                    gesture_last_sent[label] = now
                    gesture_hud_lines.append(f"{label[0]}: {target} {direction}")

            # ====== FACE =====================================================
            if not face_result.multi_face_landmarks:
                prev_face_looking = False
                state_label       = "No face detected"
                cv2.waitKey(1)
                continue

            face       = face_result.multi_face_landmarks[0]
            lm         = face.landmark
            pitch, yaw = get_head_angles(face)
            looking    = face_is_looking_at_camera(face)

            ec_mode = eye_contact_enabled.is_set()

            if ec_mode:
                if looking and not prev_face_looking:
                    print("👀 Eye contact detected")
                    on_turn_take_callback(source="eye_contact")
            prev_face_looking = looking

            # ====== EYE CONTACT DISPLAY =======================================
            draw_state = "contact" if (ec_mode and looking) else "idle"

            # ====== TEMPO NOD DETECTION ========================================
            if tempo_mode:
                if not tempo_nodding and pitch < TEMPO_DOWN_THR:
                    tempo_nodding = True
                elif tempo_nodding and pitch > TEMPO_UP_THR:
                    tempo_nodding = False
                    print("💛 TEMPO — Completed nod gesture")
                    nod_times.append(time.time())
                    tempo = estimate_tempo(nod_times)
                    if tempo:
                        print(f"💛 TEMPO — Final Tempo = {tempo} BPM")
                        on_tempo_callback(tempo)

            # ====== SHOW WINDOW ================================================
            if SHOW_WINDOW:
                draw_eye(frame, lm, LEFT_EYE_OUTLINE,  w, h, draw_state)
                draw_eye(frame, lm, RIGHT_EYE_OUTLINE, w, h, draw_state)

                hud_col = (0, 255, 0)   if "NOD"        in state_label else \
                          (0, 220, 255) if "Listening"   in state_label else \
                          (0, 220, 255) if "nod now"     in state_label else \
                          (180, 180, 180)
                cv2.putText(frame, state_label, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, hud_col, 2)

                cv2.putText(frame, f"PITCH {pitch:.1f}",
                            (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
                dl    = abs(lm[1].x - lm[234].x)
                dr    = abs(lm[1].x - lm[454].x)
                ratio = dl / dr if dr > 0 else 0
                cv2.putText(frame,
                            f"pitch:{pitch:.1f}  yaw:{yaw:.1f}  dl/dr:{ratio:.2f}  looking:{looking}",
                            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)

                # ── Finger-count gesture HUD ──────────────────────────────────
                if gesture_hud_lines:
                    gesture_hud = " | ".join(gesture_hud_lines)
                    cv2.putText(frame, gesture_hud, (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "R=Inc L=Dec | 0:Loop 1:Temp 2:Vel 3:Tempo 4:Stop",
                            (20, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Gesture Monitor", frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=camera_loop, daemon=True).start()