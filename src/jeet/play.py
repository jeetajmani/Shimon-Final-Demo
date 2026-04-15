import time
import rtmidi
from pythonosc import udp_client
from cv import start_gestures_monitor
from cv import tempo_detection_enabled
import threading
import numpy as np
import mido
import copy

accepting_eye_contact = threading.Event()
shimon_turn = False
recording_done = threading.Event()

accepting_tempo = threading.Event()
tempo = None
tempo_lock = threading.Lock()

part_1_flag = -1
phrase_num = 0

NOTE_LATENCY = 0.5

basepan_pos = None
neck_pos = None
headtile_pos = None

temperature = 0.75

free_play = False

SHIMON_ARM = udp_client.SimpleUDPClient("192.168.1.1", 9010)
SHIMON_HEAD = udp_client.SimpleUDPClient("192.168.1.1", 9000)
MAX_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", 7402)

def send_to_max(message, value):
    MAX_CLIENT.send_message(message, value)

def send_note_to_shimon(note, velocity):
    SHIMON_ARM.send_message("/arm", [note, velocity])

def send_gesture_to_shimon(part, pos, vel):
    SHIMON_HEAD.send_message("/head-commands", [part, pos, vel])
    
def quick_nod():
    send_gesture_to_shimon("NECK", 0, 15)
    send_gesture_to_shimon("HEADTILT", 0.2, 15)
    time.sleep(0.2)
    send_gesture_to_shimon("NECK", -0.2, 15)
    send_gesture_to_shimon("HEADTILT", -0.5, 20)
    time.sleep(0.2)
    send_gesture_to_shimon("NECK", -0.1, 10)
    send_gesture_to_shimon("HEADTILT", 0, 8)

def look_left():
    idle(0)
    send_gesture_to_shimon("BASEPAN", -1, 10)
    send_gesture_to_shimon("NECK", -0.1, 2)
    send_gesture_to_shimon("HEADTILT", 0, 8)
    time.sleep(1)
    quick_nod()
    idle(1)
    
def look_forward():
    idle(0)
    send_gesture_to_shimon("BASEPAN", 0, 10)
    send_gesture_to_shimon("NECK", 0.0, 10)
    idle(1)

def shimon_nod():
    send_gesture_to_shimon("NECK", 0.3, 10)
    send_gesture_to_shimon("HEADTILT", -1, 20)
    time.sleep(0.3)
    send_gesture_to_shimon("NECK", -0.3, 10)
    send_gesture_to_shimon("HEADTILT", 0.4, 20)
    time.sleep(0.3)

def idle(status):   
    send_to_max("/idle", status)
    
# def nod_to_tempo(bpm, stop_event):
#     """Nods Shimon's head on each beat until stop_event is set."""
#     beat_interval = 60.0 / bpm
#     direction = 1
#     i = 0
#     while not stop_event.is_set():
#         neck_pos  =  0.05 * direction
#         head_pos  = -0.2  * direction
#         send_gesture_to_shimon("NECK",      neck_pos, 15)
#         send_gesture_to_shimon("HEADTILT",  head_pos, 20)
#         direction *= -1
        
#         if i >= 5:
#             swing = min(0.2, (i - 5) * 0.05)
#             basepan_pos = swing * direction
#             send_gesture_to_shimon("BASEPAN", basepan_pos, 5)
        
#         i += 1
#         stop_event.wait(beat_interval)  # interruptible sleep

# ========= MUSICAL VARIATION ========= #

# Major scale intervals (semitones from root)
MAJOR_SCALE    = [0, 2, 4, 5, 7, 9, 11]
# Natural minor
MINOR_SCALE    = [0, 2, 3, 5, 7, 8, 10]
# Pentatonic major — very safe, always sounds consonant
PENTATONIC     = [0, 2, 4, 7, 9]
# Pentatonic minor
PENTATONIC_MIN = [0, 3, 5, 7, 10]

# Musical intervals to transpose by (in scale steps, not semitones)
# Weighted toward smaller, consonant movements
INTERVAL_WEIGHTS = {
    1: 0.35,   # step — most common melodic motion
    2: 0.30,   # skip (third) — very musical
    3: 0.15,   # fourth
    4: 0.10,   # fifth
    5: 0.07,   # sixth
    7: 0.03,   # octave
}


def estimate_key_and_scale(notes):
    """
    Estimate the most likely root and scale (major or minor) from a list of MIDI notes.
    Returns (root_pitch_class, scale_intervals).
    """
    pitch_classes = [n % 12 for n in notes]
    counts = np.zeros(12)
    for pc in pitch_classes:
        counts[pc] += 1

    best_root  = 0
    best_score = -1
    best_scale = MAJOR_SCALE

    for root in range(12):
        for scale in [MAJOR_SCALE, MINOR_SCALE, PENTATONIC, PENTATONIC_MIN]:
            scale_pcs = set((root + interval) % 12 for interval in scale)
            score = sum(counts[pc] for pc in scale_pcs)
            if score > best_score:
                best_score = best_root  = root
                best_score = score
                best_root  = root
                best_scale = scale

    return best_root, best_scale


def build_scale_notes(root, scale_intervals, lo=36, hi=96):
    """Return all MIDI note numbers in the given scale between lo and hi."""
    scale_notes = []
    for midi in range(lo, hi + 1):
        if (midi - root) % 12 in scale_intervals:
            scale_notes.append(midi)
    return scale_notes


def snap_to_scale(note, scale_notes):
    """Snap a MIDI note to the nearest note in scale_notes."""
    scale_arr = np.array(scale_notes)
    idx = np.argmin(np.abs(scale_arr - note))
    return int(scale_arr[idx])


def transpose_in_scale(note, scale_notes, steps):
    """
    Move a note by `steps` scale degrees up or down.
    Clamps to the available scale range.
    """
    scale_arr = np.array(scale_notes)
    # find closest scale position
    idx = int(np.argmin(np.abs(scale_arr - note)))
    new_idx = np.clip(idx + steps, 0, len(scale_arr) - 1)
    return int(scale_arr[new_idx])


def get_contour(notes):
    """Return list of +1 (up), -1 (down), 0 (same) between consecutive notes."""
    return [np.sign(notes[i+1] - notes[i]) for i in range(len(notes) - 1)]


def process_midi_phrase_dict(events, temperature):
    """
    Musically varies a MIDI phrase:
    - Estimates key/scale from the phrase itself
    - Only moves notes to scale degrees (no chromatic clashes)
    - Uses scale-step transposition (thirds, fourths, etc.) not random semitones
    - Low temperature = fewer changes, smaller intervals
    - High temperature = more changes, larger intervals allowed
    - Always protects first and last notes (phrase anchors)
    - Biases changes to preserve melodic contour
    """
    temperature = float(np.clip(temperature, 0, 1))

    active_indices = [i for i, e in enumerate(events) if e["velocity"] > 0]
    n_active = len(active_indices)
    if n_active < 2:
        return events

    active_notes = [events[i]["note"] for i in active_indices]

    # Estimate key
    root, scale_intervals = estimate_key_and_scale(active_notes)
    scale_notes = build_scale_notes(root, scale_intervals)
    print(f"🎼 Estimated root: {root}, scale size: {len(scale_intervals)}")

    # Snap all active notes to scale first (fixes any out-of-scale input notes)
    for i in active_indices:
        events[i]["note"] = snap_to_scale(events[i]["note"], scale_notes)

    # Get original contour (after snapping)
    snapped_notes = [events[i]["note"] for i in active_indices]
    contour = get_contour(snapped_notes)

    # How many notes to change — temperature controls density
    # Low temp: ~10–20% of notes. High temp: up to ~60%
    max_changes = max(1, int(n_active * (0.1 + 0.5 * temperature)))
    n_to_change = np.random.randint(0, max_changes + 1)
    if n_to_change == 0:
        return events

    # Available interval sizes — temperature controls how large steps can get
    # Low temp: only steps and skips. High temp: up to fifths/sixths
    max_interval = max(1, int(1 + temperature * 6))  # 1–7 scale steps
    available_intervals = [k for k in INTERVAL_WEIGHTS if k <= max_interval]
    interval_probs = np.array([INTERVAL_WEIGHTS[k] for k in available_intervals])
    interval_probs /= interval_probs.sum()

    # Protect first and last note — they anchor the phrase
    protected = {active_indices[0], active_indices[-1]}

    # Weight middle notes more likely to change (hanning window), skip protected
    weights = np.hanning(n_active) + 1e-6
    weights[0]  = 0
    weights[-1] = 0
    weights /= weights.sum()

    n_to_change = min(n_to_change, n_active - 2)  # can't change protected notes
    if n_to_change == 0:
        return events

    chosen_positions = np.random.choice(
        range(n_active), size=n_to_change, replace=False, p=weights
    )

    for pos in chosen_positions:
        event_idx  = active_indices[pos]
        old_note   = events[event_idx]["note"]

        # Pick interval size
        interval = int(np.random.choice(available_intervals, p=interval_probs))

        # Bias direction to match original contour where possible
        if pos < len(contour):
            contour_dir = contour[pos]
        else:
            contour_dir = 0

        if contour_dir != 0 and np.random.random() < 0.65:
            # 65% chance to follow original contour direction
            direction = int(contour_dir)
        else:
            direction = int(np.random.choice([-1, 1]))

        new_note = transpose_in_scale(old_note, scale_notes, direction * interval)

        # Keep within playable range
        new_note = int(np.clip(new_note, 48, 95))

        events[event_idx]["note"] = new_note
        # print(f"  ♪ {old_note} → {new_note} ({"+" if direction > 0 else ""}{direction * interval} scale steps)")

    return events

# def play_sequence(events, temperature):
#     events = process_midi_phrase_dict(events, temperature)
#     recording_done.clear()
#     events[0]["delta"] = 0.01

#     note_on_deltas = [e["delta"] for e in events if e["velocity"] > 0 and e["delta"] > 0.05]
#     if note_on_deltas:
#         median_delta  = float(np.median(note_on_deltas))
#         estimated_bpm = float(np.clip(60.0 / median_delta, 40, 200))
#     else:
#         estimated_bpm = 120.0
#     print(f"🎵 Estimated playback tempo: {estimated_bpm:.1f} BPM")

#     nod_stop   = threading.Event()
#     nod_thread = threading.Thread(
#         target=nod_to_tempo, args=(estimated_bpm, nod_stop), daemon=True
#     )
#     nod_thread.start()

#     for event in events:
#         time.sleep(event["delta"])
#         if event["velocity"] != 0:
#             send_note_to_shimon(event["note"], 120)
#             print(f"Sent: {event}")

#     nod_stop.set()
#     nod_thread.join(timeout=1.0)
#     look_left()

def play_sequence(events, temperature):
    events = process_midi_phrase_dict(events, temperature)
    recording_done.clear()
    
    # Adjust deltas that are too small
    # for event in events:
    #     while event["delta"] < 0.1:
    #         event["delta"] *= 2

    events[0]["delta"] = 0.01
    
    direction  = 1
    i          = 0
    last_nod   = -1.0  # force first onset to always nod
    NOD_MIN_MS = 0.15

    for event in events:
        time.sleep(event["delta"])

        if event["velocity"] != 0:
            send_note_to_shimon(event["note"], 120)
            print(f"Sent: {event}")

            now = time.time()
            if now - last_nod >= NOD_MIN_MS:
                neck_pos = 0.05 * direction
                head_pos = -0.2 * direction
                send_gesture_to_shimon("NECK",      neck_pos, 15)
                send_gesture_to_shimon("HEADTILT",  head_pos, 20)

                if i >= 5:
                    swing = min(0.2, (i - 5) * 0.05)
                    send_gesture_to_shimon("BASEPAN", swing * direction, 5)

                direction *= -1
                last_nod   = now
                i += 1

    look_left()


def on_turn_take(source="nod"):
    if not accepting_eye_contact.is_set() or shimon_turn:
        print(f"Turn-take ignored (not ready) — source: {source}")
        return

    # gate by mode
    if part_1_flag == 1 and source != "nod":
        return
    if part_1_flag == 2 and source != "eye_contact":
        return
    if part_1_flag == 3 and source != "thumbs_up":
        return

    print(f"Turn-take triggered via: {source}")
    recording_done.set()
    accepting_eye_contact.clear()

        
def on_tempo_detected(detected_tempo):
    global tempo
    with tempo_lock:
        tempo = detected_tempo
    print(f"🎵 Detected nod tempo: {tempo:.1f} BPM")


def keyboard_phrase(port_index):
    global phrase_num
    global temperature
    temperature = 1
    print("KEYBOARD SEND")
    
    events = []
    start_time = time.time()
    last_time = start_time
    last_note_time = time.time()
    
    midi_in = rtmidi.MidiIn()
    midi_in.open_port(port_index)
    
    i = 0
    SILENCE_TIMEOUT = 2.0
    
    while True:
        try: 
            # VISUAL MODE (part_1_flag == 1 or 3)
            if part_1_flag in (1, 2, 3):
                msg = midi_in.get_message()
                if msg:
                    now = time.time()
                    delta = now - last_time
                    last_time = now
                    event = {
                        "index": i,
                        "note": msg[0][1],
                        "velocity": msg[0][2],
                        "delta": delta
                    }
                    events.append(event)
                    print(event)
                    i += 1
                    if len(events) >= 10:
                        print("NOW WE CAN SEE - Look at Shimon when ready!")
                        accepting_eye_contact.set()
                else:
                    time.sleep(0.001)
                    
                if recording_done.is_set():
                    print("ALL DONE - Turn-take received!")
                    events_original = copy.deepcopy(events)
                    look_forward()
                    play_sequence(events, temperature)
                    events.clear()
                    i = 0
                    recording_done.clear()
                    accepting_eye_contact.clear()
                    print("READY FOR NEXT PHRASE")
                    break

            # AUDIO MODE (part_1_flag == 0)
            elif part_1_flag == 0:
                msg = midi_in.get_message()
                if msg:
                    now = time.time()
                    delta = now - last_time
                    last_time = now
                    last_note_time = now
                    event = {
                        "index": i,
                        "note": msg[0][1],
                        "velocity": msg[0][2],
                        "delta": delta
                    }
                    events.append(event)
                    print(event)
                    i += 1
                else:
                    time.sleep(0.001)
                    
                if len(events) >= 10:
                    silence_duration = time.time() - last_note_time
                    if silence_duration >= SILENCE_TIMEOUT:
                        print(f"ALL DONE - {SILENCE_TIMEOUT}s silence detected!")
                        events_original = copy.deepcopy(events)
                        look_forward()
                        play_sequence(events, temperature)
                        events.clear()
                        i = 0
                        last_note_time = time.time()
                        print("READY FOR NEXT PHRASE")
                        break
                
        except KeyboardInterrupt:
            look_forward()
            idle(0)
            exit()
  
def tempo_detect():
    global tempo
    global free_play
    print("Waiting for tempo from nods...")
    tempo_detection_enabled.set()
    with tempo_lock:
        tempo = None

    bpm = None
    while bpm is None:
        with tempo_lock:
            if tempo is not None:
                bpm = tempo
                tempo = None
        time.sleep(0.01)

    print(f"Tempo confirmed: {bpm:.1f} BPM — starting head movement")
    idle(0)
    free_play = True
    try:
        send_to_max("/tempo", bpm)
        send_to_max("/status", 1)
        print("\n-------------")
        print("LIVE TEMPO MODE: Nod 3 times to set new tempo")
        print("-------------")
        while True:
            with tempo_lock:
                if tempo is not None:
                    bpm = tempo
                    tempo = None
                    send_to_max("/tempo", bpm)
                    print(f"🎛 LIVE TEMPO UPDATED → {bpm} BPM")
            time.sleep(0.01)
    except KeyboardInterrupt:
        send_to_max("/status", 0)
        look_forward()
    
def play_notes():
    global tempo
    beat_interval = 60.0 / tempo
    t0 = time.monotonic()
    beat_count = 0
    while True:
        now = time.monotonic()
        next_beat = t0 + beat_count * beat_interval
        if now >= next_beat:
            note = int(np.random.randint(48, 96))
            send_note_to_shimon(note, 80)
            print(f"[{beat_count}] 🎹 Note {note} on beat")
            beat_count += 1

def move_neck_and_head():
    global tempo
    beat_interval = 60.0 / tempo
    t0 = time.monotonic()
    next_beat = t0
    beat_count = 0
    neck_direction = -1
    while True:
        now = time.monotonic()
        if now >= next_beat:
            neck_pos = 0.05 * neck_direction
            head_pos = -0.2 * neck_direction
            send_gesture_to_shimon("NECK", neck_pos, 15)
            send_gesture_to_shimon("HEADTILT", head_pos, 20)
            neck_direction *= -1
            beat_count += 1
            next_beat += beat_interval

if __name__ == "__main__":
    midi_in = rtmidi.MidiIn()
    ports = midi_in.get_ports()
    print("Available ports:", ports)

    port_index  = int(input("Select input port number: "))
    part_1_flag = int(input("Select AUDIO (0) / VISUAL+NOD (1) / VISUAL ONLY (2) / THUMBS UP (3): "))

    if part_1_flag == 0:
        print("------------------------------")
        print("------------------- AUDIO ONLY")
        print("------------------------------")
    elif part_1_flag == 1:
        print("------------------------------")
        print("-------- EYE CONTACT + NODDING")
        print("------------------------------")
        start_gestures_monitor(on_turn_take, on_tempo_detected)
    elif part_1_flag == 2:
        print("------------------------------")
        print("------------- EYE CONTACT ONLY")
        print("------------------------------")
        start_gestures_monitor(on_turn_take, on_tempo_detected)
    elif part_1_flag == 3:
        print("------------------------------")
        print("-------------------- THUMBS UP")
        print("------------------------------")
        start_gestures_monitor(on_turn_take, on_tempo_detected)

    look_left()
    for i in range(7):
        keyboard_phrase(port_index)
    
    idle(0)
    look_forward()