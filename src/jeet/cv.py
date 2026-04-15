import cv2
import time
import threading
import math
import numpy as np
import mediapipe as mp
from collections import deque

tempo_detection_enabled = threading.Event()

SHOW_WINDOW = False
DEBUG_FPS = False

# ── Eye outline landmarks (no iris) ──────────────────────────────────────────
LEFT_EYE_OUTLINE  = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                     173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                     466, 388, 387, 386, 385, 384, 398]

# ── Tempo nod thresholds ──────────────────────────────────────────────────────
TEMPO_DOWN_THR   = -5
TEMPO_UP_THR     = -0.5
TEMPO_MIN_NODS   = 3
TEMPO_WINDOW_SEC = 3.45

# ── Eye contact + nod thresholds ─────────────────────────────────────────────
EYE_CONTACT_HOLD_SEC = 0.001
NOD_WINDOW_SEC       = 5.0
NOD_DOWN_THR         = -1
NOD_UP_THR           =  -5

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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


# ========= HAND GESTURE HELPERS ========= #

WRIST = 0
TH_MCP, TH_IP, TH_TIP = 2, 3, 4
IX_MCP, IX_PIP, IX_DIP = 5, 6, 7
MI_MCP, MI_PIP, MI_DIP = 9, 10, 11
RI_MCP, RI_PIP, RI_DIP = 13, 14, 15
PI_MCP, PI_PIP, PI_DIP = 17, 18, 19

def _pip_angle(pts, mcp, pip, dip):
    v1 = pts[mcp] - pts[pip]
    v2 = pts[dip] - pts[pip]
    a, b = np.linalg.norm(v1), np.linalg.norm(v2)
    if a == 0 or b == 0:
        return 180
    cosang = np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _curl_score(pts, mcp, pip, dip):
    return float(np.clip((180 - _pip_angle(pts, mcp, pip, dip)) / 120, 0, 1))

def _others_mostly_folded(pts, tol=0.45):
    curls = [_curl_score(pts, IX_MCP, IX_PIP, IX_DIP),
             _curl_score(pts, MI_MCP, MI_PIP, MI_DIP),
             _curl_score(pts, RI_MCP, RI_PIP, RI_DIP),
             _curl_score(pts, PI_MCP, PI_PIP, PI_DIP)]
    return sum(c >= tol for c in curls) >= 3

def _thumb_extended_and_dir(pts):
    v1 = pts[TH_MCP] - pts[TH_IP]
    v2 = pts[TH_TIP] - pts[TH_IP]
    a, b = np.linalg.norm(v1), np.linalg.norm(v2)
    ang = 180 if a == 0 or b == 0 else math.degrees(math.acos(np.clip(np.dot(v1, v2) / (a * b), -1, 1)))
    dir_vec = pts[TH_TIP] - pts[TH_MCP]
    n = np.linalg.norm(dir_vec)
    dir_vec = dir_vec / n if n > 0 else np.array([0.0, 0.0])
    return (ang > 150, -dir_vec[1])

def is_thumbs_up(pts):
    if not _others_mostly_folded(pts):
        return False
    th_ext, upness = _thumb_extended_and_dir(pts)
    return th_ext and upness > 0.35


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
                           on_tempo_callback):
    """
    on_turn_take_callback(source)  — fired when a turn-taking gesture is detected.
                                     source is "nod" or "thumbs_up".
    on_tempo_callback(bpm)         — fired when a tempo estimate is ready.
    """

    def camera_loop():
        # ── Tempo nod state ───────────────────────────────────────────────────
        tempo_nodding = False
        nod_times     = []

        # ── Eye contact + nod state ───────────────────────────────────────────
        prev_face_looking = False
        eye_contact_start = None
        nod_listen_start  = None
        nod_display_until = 0
        eye_nod_active    = False
        state_label       = "Looking for eye contact…"

        # ── Thumbs up state ───────────────────────────────────────────────────
        prev_thumbs_up = False

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

            # ====== THUMBS UP ================================================
            thumbs_up = False
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                    if is_thumbs_up(pts):
                        thumbs_up = True
                        break

            if thumbs_up and not prev_thumbs_up:
                print("👍 Thumbs up — turn take")
                on_turn_take_callback(source="thumbs_up")
            prev_thumbs_up = thumbs_up

            # ====== FACE =====================================================
            if not face_result.multi_face_landmarks:
                eye_contact_start = None
                nod_listen_start  = None
                eye_nod_active    = False
                prev_face_looking = False
                state_label       = "No face detected"
                cv2.waitKey(1)
                continue

            face       = face_result.multi_face_landmarks[0]
            lm         = face.landmark
            pitch, yaw = get_head_angles(face)
            looking    = face_is_looking_at_camera(face)

            if looking and not prev_face_looking:
                print("👀 Eye contact detected")
                on_turn_take_callback(source="eye_contact")
            prev_face_looking = looking

            # ====== EYE CONTACT + NOD STATE MACHINE ===========================
            draw_state = "idle"

            if nod_listen_start is None:
                if looking:
                    draw_state = "contact"
                    if eye_contact_start is None:
                        eye_contact_start = now
                    elif now - eye_contact_start >= EYE_CONTACT_HOLD_SEC:
                        nod_listen_start = now
                        eye_nod_active   = False
                        state_label      = "Eye contact ✓ — nod now!"
                else:
                    eye_contact_start = None
                    state_label = "Looking for eye contact…"
            else:
                elapsed    = now - nod_listen_start
                draw_state = "armed"

                if elapsed <= NOD_WINDOW_SEC:
                    state_label = f"Listening for nod… ({NOD_WINDOW_SEC - elapsed:.1f}s)"

                    if not eye_nod_active and pitch < NOD_DOWN_THR:
                        eye_nod_active = True

                    elif eye_nod_active and pitch > NOD_UP_THR:
                        eye_nod_active    = False
                        nod_display_until = now + 2.0
                        state_label       = "NOD DETECTED ✓"
                        draw_state        = "nod"
                        nod_listen_start  = None
                        eye_contact_start = None
                        print("✅ Eye-contact nod — turn take")
                        on_turn_take_callback(source="nod")
                else:
                    nod_listen_start  = None
                    eye_contact_start = None
                    eye_nod_active    = False
                    state_label       = "Looking for eye contact…"
                    draw_state        = "idle"

            if now < nod_display_until:
                draw_state = "nod"

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

                if now < nod_display_until:
                    cv2.putText(frame, "NOD DETECTED", (w // 2 - 150, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)

                pitch_col = (0, 255, 255) if eye_nod_active else (0, 80, 255)
                cv2.putText(frame, f"PITCH {pitch:.1f}  (down<{NOD_DOWN_THR}, up>{NOD_UP_THR})",
                            (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pitch_col, 2)
                dl    = abs(lm[1].x - lm[234].x)
                dr    = abs(lm[1].x - lm[454].x)
                ratio = dl / dr if dr > 0 else 0
                cv2.putText(frame,
                            f"pitch:{pitch:.1f}  yaw:{yaw:.1f}  dl/dr:{ratio:.2f}  looking:{looking}",
                            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1)

                cv2.imshow("Gesture Monitor", frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

    threading.Thread(target=camera_loop, daemon=True).start()