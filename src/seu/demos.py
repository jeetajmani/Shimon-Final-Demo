import os.path
import random
from copy import copy, deepcopy
import pretty_midi
from pretty_midi import Note
import soundfile as sf
from pythonosc import udp_client
from rtmidi.midiconstants import NOTE_OFF, NOTE_ON
from audioToMidi import AudioMidiConverter
from audioDevice import AudioDevice
from tempoTracker import TempoTracker
from gestureController import GestureController
import numpy as np
from threading import Thread, Lock, Event
import time
import threading
from queue import Queue
from midi import MidiOutDevice, MidiMessage
import utils

# ... (Instrument, Instruments, Phrase 클래스는 기존과 동일) ...
class Instrument:
    def __init__(self, name: str, channel: int, is_audio: bool = True, should_constrain_raga: bool = False):
        self.idx = -1; self.name = name; self.channel = channel; self.is_audio = is_audio; self.should_constrain_raga = should_constrain_raga
    def __eq__(self, other): return self.name == other.name
    def __str__(self): return self.name
    @property
    def is_midi(self): return not self.is_audio

class Instruments:
    def __init__(self, instruments: [Instrument], randomize: bool = True):
        self.instruments = instruments
        for i in range(len(self.instruments)): self.instruments[i].idx = i
        self.randomize = randomize; self.idx = 0
    def __len__(self): return len(self.instruments)
    def __next__(self):
        if self.randomize:
            if len(self.instruments) > 1: self.idx = random.choice([i for i in range(len(self.instruments)) if i != self.idx])
        else: self.idx = (self.idx + 1) % len(self.instruments)
        print(f"Now {self.current}'s turn"); return self.idx
    @property
    def current(self) -> Instrument: return self.instruments[self.idx]
    @property
    def total_channels(self):
        ch = 1
        for inst in self.instruments: ch = max(ch, inst.channel)
        return ch

class Phrase:
    def __init__(self, notes=None, onsets=None, tempo=None, name=None):
        self.name = name; self.notes = notes if notes is not None else []; self.onsets = onsets if onsets is not None else []; self.tempo = tempo; self.is_korvai = name == "korvai"; self.is_intro = name == "intro"
    def get(self): return self.notes, self.onsets
    def get_raw_notes(self):
        notes = []
        for note in self.notes: notes.append(note.pitch)
        return np.array(notes)
    def __len__(self): return len(self.notes)
    def __getitem__(self, item):
        if len(self.notes) > item: return self.notes[item], self.onsets[item]
        return None, None
    def __setitem__(self, key, value: tuple):
        if len(self.notes) > key: self.notes[key] = value[0]; self.onsets[key] = value[1]
    def append(self, note, onset): self.notes.append(note); self.onsets.append(onset)
    def __str__(self):
        ret = ""
        for i in range(len(self.notes)): ret = ret + f"{self.notes[i]}, Onset: {self.onsets[i]}\n"
        return ret

class Performer(GestureController):
    def __init__(self, osc_address: str, osc_port: int, gesture_note_mapping: dict[str, int], osc_arm_route: str,
                 osc_head_route: str, min_note_dist_ms,
                 max_notes_per_onset, tempo=None, ticks=None, virtual=False):
        self.client = udp_client.SimpleUDPClient(osc_address, osc_port)
        super().__init__(self.client, gesture_note_mapping, osc_head_route)
        self.tempo = tempo; self.osc_arm_route = osc_arm_route; self.ticks = ticks; self.min_note_dist_ms = min_note_dist_ms; self.max_notes_per_onset = max_notes_per_onset
        self.note_on_thread = Thread(); self.note_off_thread = Thread(); self.lock = Lock(); self.stop_event = threading.Event(); self.timer = None
        self.midi_out = MidiOutDevice('ShimonDemo', virtual=True) if virtual else None
        self.is_performing = False; self.perform_lock = Lock(); self.abort_signal = False; self.qna_tempo_scale = 1.0; self.velocity_scale = 1.0

    def set_file_tempo(self, tempo): self.tempo = tempo
    def add_qna_tempo(self, value_to_add):
        self.qna_tempo_scale = max(0.1, min(self.qna_tempo_scale + float(value_to_add), 3.0))
        try: udp_client.SimpleUDPClient("127.0.0.1", 5000).send_message("/tempo_scale", self.qna_tempo_scale)
        except: pass
    def add_velocity_scale(self, value_to_add): self.velocity_scale = max(0.1, min(self.velocity_scale + float(value_to_add), 2.0))
    def stop_performance(self): self.abort_signal = True

    def perform_gestures(self, gestures: Phrase, tempo=None, wait_for_measure_end=False):
        self.note_on_thread = Thread(target=self.handle_note_ons, args=(gestures.notes, tempo))
        self.note_off_thread = Thread(target=self.handle_note_offs, args=(gestures.notes, tempo))
        if self.timer and self.timer.is_alive(): self.timer.join()
        self.stop_event.set(); self.timer = threading.Timer(0.5, self.delay_start_thread); self.timer.start()

    def delay_start_thread(self):
        if self.note_on_thread.is_alive(): self.note_on_thread.join()
        if self.note_off_thread.is_alive(): self.note_off_thread.join()
        self.stop_event.clear(); self.note_on_thread.start(); self.note_off_thread.start()

    def handle_note_ons(self, notes: [Note], tempo: int):
        m = self.tempo / tempo if tempo and self.tempo else 1
        prev_start = 0; now = time.time()
        for note in notes:
            if self.stop_event.is_set(): return
            dly = max(0, ((note.start - prev_start) * m) - (time.time() - now))
            self.stop_event.wait(dly); now = time.time(); self.lock.acquire()
            self.send_gesture(note.pitch, note.velocity); self.lock.release(); prev_start = note.start

    def handle_note_offs(self, notes: [Note], tempo: int):
        m = self.tempo / tempo if tempo and self.tempo else 1
        prev_end = 0; now = time.time()
        for note in notes:
            if self.stop_event.is_set(): return
            dly = max(0, ((note.end - prev_end) * m) - (time.time() - now))
            self.stop_event.wait(dly); now = time.time(); self.lock.acquire()
            self.send_gesture(note.pitch, 0); self.lock.release(); prev_end = note.end

    def perform(self, phrase: Phrase, gestures: Phrase or None, tempo=None, wait_for_measure_end=False, repeats: int = 1, temperature : float = 0):
        try: udp_client.SimpleUDPClient("127.0.0.1", 5001).send_message("/switch", 0)
        except: pass
        with self.perform_lock:
            if self.is_performing: return
            self.is_performing = True; self.abort_signal = False
        try:
            phrase = self.filter_phrase(phrase, min_note_dist_ms=self.min_note_dist_ms, max_notes_per_onset=self.max_notes_per_onset)
            m = self.tempo / tempo if self.tempo and tempo else 1
            if gestures: self.perform_gestures(gestures=gestures, tempo=tempo, wait_for_measure_end=wait_for_measure_end)
            for loop_idx in range(repeats):
                if self.abort_signal: break 
                phrase_copy = QnADemo.process_midi_phrase(deepcopy(phrase), temperature) 
                notes, onsets = phrase_copy.get()
                i = 0
                while i < len(notes):
                    if self.abort_signal: break
                    poly_notes = []
                    poly_onsets = []
                    while i < len(onsets):
                        if len(poly_notes) > 0 and poly_onsets[-1] == onsets[i]: poly_notes.append(notes[i]); poly_onsets.append(onsets[i]); i += 1
                        elif len(poly_notes) == 0 and i < len(notes) - 1 and onsets[i] == onsets[i + 1]: poly_notes.append(notes[i]); poly_notes.append(notes[i + 1]); poly_onsets.append(onsets[i]); poly_onsets.append(onsets[i + 1]); i += 2
                        else: break
                    duration = 0
                    if len(poly_notes) > 0:
                        if i < len(notes): duration = notes[i].start - poly_notes[0].start
                        for j in range(len(poly_notes)): self.send_client(MidiMessage(NOTE_ON, 1, int(poly_notes[j].pitch), int(poly_notes[j].velocity)))
                    else:
                        if i < len(notes) - 1: duration = notes[i + 1].start - notes[i].start
                        self.send_client(MidiMessage(NOTE_ON, 1, int(notes[i].pitch), int(notes[i].velocity))); i += 1
                    sleep_time = (duration * m) / self.qna_tempo_scale if tempo is None else (duration * m)
                    time.sleep(sleep_time)
            if wait_for_measure_end and tempo and self.ticks: self.wait_for_measure_end(onsets, tempo)
        finally:
            with self.perform_lock: self.is_performing = False
            try:
                udp_client.SimpleUDPClient("127.0.0.1", 5001).send_message("/switch", 1)
                udp_client.SimpleUDPClient("127.0.0.1", 7000).send_message("/end", 1)
            except: pass

    def send_client(self, msg: MidiMessage):
        clamped_velocity = max(0, min(127, int(msg.velocity * self.velocity_scale)))
        if self.midi_out:
            self.midi_out.send([NOTE_ON + (msg.channel - 1), msg.pitch, clamped_velocity]); self.midi_out.send([NOTE_OFF + (msg.channel - 1), msg.pitch, clamped_velocity])
        else: self.client.send_message(self.osc_arm_route, [msg.pitch, clamped_velocity])

    @staticmethod
    def filter_phrase(phrase: Phrase, min_note_dist_ms: float = 50, max_notes_per_onset: int = 4):
        temp = Phrase(); notes, onsets = phrase.get(); same_onset_count = 0; min_note_dist = min_note_dist_ms / 1000
        temp.append(notes[0], onsets[0])
        for i in range(1, len(phrase)):
            d_time = abs(notes[i].start - temp[-1][0].start)
            if d_time < 1e-2:
                if same_onset_count >= max_notes_per_onset - 1: continue
                same_onset_count += 1
            elif min_note_dist > d_time: continue
            else: same_onset_count = 0
            temp.append(notes[i], onsets[i])
        return temp

class Demo: pass

class QnADemo(Demo):
    def __init__(self, performer: Performer, raga_map, sr=16000,
                 instruments=("Violin", "Keys"), frame_size=2048, activation_threshold=0.02, n_wait=16,
                 input_dev_name='Scarlett 2i2 USB', outlier_filter_coeff=2,
                 timeout_sec=2, randomness_temperature: float = 1.0,
                 auto_random_temp=False, auto_random_loop=False,
                 temp_min=0.1, temp_max=1.0): # <-- 랜덤 범위 인자 추가
        super().__init__()
        self.active = False; self.activation_threshold = activation_threshold; self.n_wait = n_wait
        self.randomness_temperature = randomness_temperature; self.wait_count = 0; self.playing = False
        self.phrase = []; self.midi_notes = []; self.midi_onsets = []; self.outlier_filter_coeff = outlier_filter_coeff
        self.repeat_count = 4; self.process_thread = Thread(); self.event = Event(); self.lock = Lock()
        self.instruments = Instruments(instruments, randomize=False)
        self.auto_random_temp = auto_random_temp; self.auto_random_loop = auto_random_loop 
        self.temp_min = temp_min; self.temp_max = temp_max # <-- 범위 저장
        try: self.audio_device = AudioDevice(self.callback, rate=sr, frame_size=frame_size, input_dev_name=input_dev_name, channels=self.instruments.total_channels)
        except AssertionError: self.audio_device = None
        if self.audio_device is None: self.instruments = Instruments([Instrument("Keys", 1, False)])
        self.audio2midi = AudioMidiConverter(raga_map=raga_map, sr=sr, frame_size=frame_size, outlier_filter_coeff=outlier_filter_coeff)
        if self.audio_device: self.audio_device.start()
        self.timeout = timeout_sec; self.last_time = time.time(); self.performer = performer

    def set_temperature(self, value_to_add): self.randomness_temperature = max(0.0, min(self.randomness_temperature + float(value_to_add), 1.0))
    def set_repeat_count(self, value_to_add): self.repeat_count = max(1, self.repeat_count + int(value_to_add))
    def stop_loop_immediately(self, unused_arg): self.performer.stop_performance(); self.reset_var()
    def reset_var(self): self.wait_count = 0; self.playing = False; self.phrase = []; self.last_time = time.time()
    def handle_midi(self, msg, dt):
        if not self.instruments.current.is_midi: return
        if msg[0] == NOTE_ON:
            self.last_time = time.time(); note = pretty_midi.Note(msg[2], msg[1], self.last_time, self.last_time + 0.1)
            self.midi_notes.append(note); self.midi_onsets.append(self.last_time)
    def callback(self, channel: int, samples: np.ndarray):
        if not self.active: self.reset_var(); return
        act = np.abs(samples).mean()
        if self.instruments.current.channel != channel or self.instruments.current.is_midi: return
        if act > self.activation_threshold: self.lock.acquire(); self.wait_count = 0; self.playing = True; self.phrase.append(samples); self.lock.release()
        else:
            self.lock.acquire()
            if self.wait_count > self.n_wait: self.wait_count = 0; self.playing = False
            else:
                if self.playing: self.phrase.append(samples)
                self.wait_count += 1
            self.lock.release()
    def start(self):
        self.reset_var()
        if self.process_thread.is_alive(): self.process_thread.join()
        self.lock.acquire(); self.active = True; self.lock.release()
        self.process_thread = Thread(target=self._process); self.process_thread.start(); self.event.clear(); self.check_timeout()
    def _process(self):
        while True:
            time.sleep(0.1); self.lock.acquire()
            if not self.active: self.lock.release(); return
            if not (self.playing or len(self.phrase) == 0): self.lock.release(); break
            self.lock.release()
        self.lock.acquire(); phrase = np.hstack(self.phrase); self.phrase = []; self.lock.release()
        if len(phrase) > 0:
            notes, onsets = self.audio2midi.convert(phrase, return_onsets=True, constrain_raga=self.instruments.current.should_constrain_raga)
            self.perform(Phrase(notes, onsets))
        self._process()
    def stop(self):
        self.lock.acquire(); self.active = False; self.lock.release()
        if self.process_thread.is_alive(): self.process_thread.join()
        if self.audio_device: self.audio_device.stop()
        self.event.set()

    def perform(self, phrase):
        # [ Image of a randomized parameter distribution range ]
        if self.auto_random_temp: 
            # 인자로 받은 범위를 사용하여 랜덤 온도 생성
            self.randomness_temperature = round(random.uniform(self.temp_min, self.temp_max), 2)
            print(f"🎲 [AUTO] Random Temp ({self.temp_min}~{self.temp_max}): {self.randomness_temperature}")
        if self.auto_random_loop: 
            self.repeat_count = random.randint(1, 8)
            print(f"🎲 [AUTO] Random Loop (1~8): {self.repeat_count}")

        self.performer.send_gesture(gesture="look", velocity=3); self.performer.send_gesture(gesture="headcircle", velocity=80)
        self.performer.perform(phrase=phrase, gestures=None, repeats=self.repeat_count, temperature=self.randomness_temperature)
        self.performer.send_gesture(gesture="headcircle", velocity=0); time.sleep(0.5)
        inst = next(self.instruments) + 1
        self.performer.send_gesture(gesture="look", velocity=inst + 6); self.performer.send_gesture(gesture="look", velocity=inst)

    @staticmethod
    def process_midi_phrase(phrase, temperature: float):
        temp = max(min(temperature, 1), 0); end = int((len(phrase)) * temp)
        if end <= 0: return phrase
        n_notes = np.random.randint(0, end, 1) if end > 0 else 0
        if n_notes <= 0: return phrase
        w = np.maximum(0, temperature - 1) + np.hanning(len(phrase)) + 1e-6; p = w / np.sum(w)
        indices = np.random.choice(np.arange(len(phrase)), n_notes, replace=False, p=p)
        options = np.unique(phrase.get_raw_notes())
        if len(options) == 0: return phrase
        for i in indices: phrase.notes[i].pitch = np.random.choice(options, 1)[0]
        return phrase

    def check_timeout(self):
        if time.time() - self.last_time > self.timeout and len(self.midi_notes) > 0:
            midi_notes = copy(self.midi_notes); midi_onsets = copy(self.midi_onsets); self.midi_notes = []; self.midi_onsets = []
            t = midi_notes[0].start
            for i in range(len(midi_notes)): midi_notes[i].start -= t; midi_notes[i].end -= t; midi_onsets[i] -= t
            if self.performer.is_performing: threading.Timer(1, self.check_timeout).start(); return
            self.perform(Phrase(midi_notes, midi_onsets))
        if not self.event.is_set(): threading.Timer(1, self.check_timeout).start()

class BeatDetectionDemo(Demo):
    def __init__(self, performer: Performer, tempo_range: tuple = (60, 120), smoothing=4, n_beats_to_track=16, timeout_sec=5, timeout_callback=None, user_data=None, default_tempo: int = None):
        super().__init__(); self.performer = performer; self.timeout_callback = timeout_callback; self.user_data = user_data
        self.tempo_tracker = TempoTracker(smoothing=smoothing, n_beats_to_track=n_beats_to_track, tempo_range=tempo_range, default_tempo=default_tempo, timeout_sec=timeout_sec, timeout_callback=self.timeout_handle)
        self._event = threading.Event(); self._first_time = True; self.beat_interval = -1
    def start(self): self._first_time = True; self.performer.send_gesture("look", 9); self.performer.send_gesture("look", 3); self.tempo_tracker.start(); self._event.clear()
    def stop(self): self.tempo_tracker.stop(); self._event.set(); self._first_time = True
    def handle_midi(self, msg, dt):
        if msg[0] == NOTE_ON:
            tempo = self.tempo_tracker.track_tempo(msg, dt)
            if tempo: self.performer.send_gesture(gesture="beatOnce", velocity=80); self.beat_interval = 60/tempo; (self.gesture_ctl() if self._first_time else None); self._first_time = False
    def get_tempo(self): return self.tempo_tracker.tempo
    def timeout_handle(self): self.timeout_callback(self.user_data)
    def gesture_ctl(self):
        if not self._event.is_set(): threading.Timer(self.beat_interval, self.gesture_ctl).start()

class SongDemo(Demo):
    def __init__(self, performer: Performer, midi_files: [[str]], gesture_midi_files: [[str]], note_mapping = (96, 98, 100), complete_callback=None, user_data=None):
        super().__init__(); self.performer = performer; self.phrase_note_map = note_mapping; self.user_data = user_data; self.phrase_idx = 0; self.variation_idx = 0; self.phrases = self._parse_midi(midi_files); self.g_phrases = self._parse_midi(gesture_midi_files)
        if not self.phrases: self.file_tempo = 120.0; self.next_phrase = None; self.tempo = 120.0
        else: self.file_tempo = self.phrases[0][0].tempo; self.next_phrase = self.phrases[0][0]; self.next_g_phrase = self.g_phrases[0][0]; self.tempo = self.file_tempo
        self.ticks = 480; self.playing = False; self.thread = Thread(); self.lock = Lock(); self.callback_queue = Queue(1); self.callback_queue.put(complete_callback)
    def set_tempo(self, tempo): self.tempo = tempo
    def start(self):
        if not self.phrases: return
        self.playing = True; self.performer.send_gesture("look", 8); self.performer.set_file_tempo(self.file_tempo)
        self.thread = Thread(target=self.perform, args=(self.next_phrase, self.next_g_phrase)); self.thread.start()
    def stop(self): self.lock.acquire(); self.playing = False; self.lock.release()
    def handle_midi(self, msg, dt):
        if msg[0] == NOTE_ON and msg[1] in self.phrase_note_map:
            self.phrase_idx = self.phrase_note_map.index(msg[1]); self.set_phrase(True)
    def set_phrase(self, reset=False):
        if 0 <= self.phrase_idx < len(self.phrases):
            self.variation_idx = 0 if reset else (self.variation_idx + 1) % len(self.phrases[self.phrase_idx])
            self.next_phrase = self.phrases[self.phrase_idx][self.variation_idx]; self.next_g_phrase = self.g_phrases[self.phrase_idx][self.variation_idx]
    def perform(self, phrase, gestures):
        if not phrase: (self.callback_queue.get()(self.user_data) if not self.callback_queue.empty() else None); return
        self.performer.perform(phrase, gestures, self.tempo, wait_for_measure_end=True)
        self.lock.acquire(); play = self.playing; self.lock.release()
        if self.next_phrase and play: self.set_phrase(); self.perform(self.next_phrase, self.next_g_phrase)
    def _parse_midi(self, files):
        if not files: return None
        res = []
        for v in files:
            temp = []
            for f in v:
                if not os.path.exists(f): continue
                m = pretty_midi.PrettyMIDI(f); notes = sorted(m.instruments[0].notes, key=lambda x: x.start); ons = [m.time_to_tick(n.start) for n in notes]
                temp.append(Phrase(notes, ons, round(m.get_tempo_changes()[1][0],3) if m.get_tempo_changes()[1].size>0 else 120.0, os.path.basename(f)))
            if temp: res.append(temp)
        return res
    def reset(self): self.stop()