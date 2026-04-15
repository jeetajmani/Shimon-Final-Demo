from midi import MidiInDevice
from rtmidi.midiconstants import NOTE_ON
import time
from demos import Performer, Demo, BeatDetectionDemo, QnADemo, SongDemo, Instrument
import threading
from pythonosc import dispatcher, osc_server
import cv2
import numpy as np

class ShimonDemo:
    def __init__(self, keyboard_name, has_note_off_support, mode_key, qna_param, bd_param, song_param, performer_param):
        self.mode_key = mode_key
        self.performer = Performer(ticks=480, **performer_param)
        self.keys = MidiInDevice(keyboard_name, has_note_off_support, callback_fn=self.keys_callback)
        self.qna_demo = QnADemo(performer=self.performer, **qna_param)
        self.bd_demo = BeatDetectionDemo(performer=self.performer, timeout_callback=self.bd_timeout_callback, **bd_param)
        self.song_demo = SongDemo(performer=self.performer, complete_callback=self.song_complete_callback, **song_param)
        self.running = False
        self.current_demo = self.qna_demo
        self.osc_dispatcher = dispatcher.Dispatcher()
        
        # All Handlers Enabled
        self.osc_dispatcher.map("/shimon/temperature/add", self._handle_temperature_add)
        self.osc_dispatcher.map("/shimon/loop/add", self._handle_loop_add)
        self.osc_dispatcher.map("/shimon/tempo/add", self._handle_tempo_add)
        self.osc_dispatcher.map("/shimon/velocity/add", self._handle_velocity_add)
        self.osc_dispatcher.map("/shimon/loop/stop", self._handle_loop_stop)

        self.osc_server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 9000), self.osc_dispatcher)
        self.osc_server_thread = threading.Thread(target=self.osc_server.serve_forever)
        self.osc_server_thread.daemon = True

    def _handle_temperature_add(self, address, *args):
        if args: self.qna_demo.set_temperature(args[0])
    def _handle_loop_add(self, address, *args):
        if args: self.qna_demo.set_repeat_count(args[0])
    def _handle_loop_stop(self, address, *args):
        self.qna_demo.stop_loop_immediately(0)
    def _handle_velocity_add(self, address, *args):
        if args: self.performer.add_velocity_scale(args[0])
    def _handle_tempo_add(self, address, *args):
        if args: self.performer.add_qna_tempo(args[0])

    def bd_timeout_callback(self, user_data): self.manage_demos()
    def keys_callback(self, msg, dt, user_data):
        if msg[0] == NOTE_ON:
            if msg[1] == self.mode_key and self.current_demo != self.song_demo: self.manage_demos()
            else: self.current_demo.handle_midi(msg, dt)
    def song_complete_callback(self, user_data): self.stop()
    def manage_demos(self):
        self.current_demo.stop()
        if self.current_demo == self.qna_demo: self.current_demo = self.bd_demo
        elif self.current_demo == self.bd_demo:
            tempo = self.bd_demo.get_tempo(); (self.song_demo.set_tempo(tempo) if tempo and tempo > 0 else None)
            self.current_demo = self.song_demo
        self.current_demo.start()

    def run(self):
        self.running = True; self.osc_server_thread.start(); self.current_demo.start()
        hud_image = np.zeros((320, 450, 3), dtype=np.uint8); window_name = "Shimon Controls - HUMAN"
        while self.running:
            hud_image.fill(0) 
            cv2.putText(hud_image, f"Mode: HUMAN-DOMINANT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(hud_image, f"Temperature: {self.qna_demo.randomness_temperature:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(hud_image, f"Loop Count: {self.qna_demo.repeat_count}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(hud_image, f"Tempo Scale: {self.performer.qna_tempo_scale:.1f}x", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(hud_image, f"Velocity Scale: {self.performer.velocity_scale:.1f}x", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, hud_image)
            if cv2.waitKey(100) & 0xFF == ord('q'): self.reset()
        cv2.destroyWindow(window_name)

    def stop(self):
        self.running = False; self.osc_server.shutdown(); self.osc_server_thread.join()
        self.qna_demo.stop(); self.bd_demo.stop(); self.song_demo.stop()
    def reset(self): self.stop(); self.keys.reset()

if __name__ == '__main__':
    # 1. 제스처 이름표 (Library) 정의
    gesture_note_mapping = {
        "beatOnce": 50, "breath": 51, "look": 52, "circle": 53,
        "nodsway": 54, "ar_sway": 55, "eyebrows": 56, "headcircle": 57,
        "cooldown": 58, "scream": 59, "headcirclefast": 60, "no": 61
    }

    performer_params = {
        "osc_address": "127.0.0.1", "osc_port": 20000,
        "gesture_note_mapping": gesture_note_mapping, # 위에서 만든 이름표 사용
        "osc_arm_route": "/arm", "osc_head_route": "/head",
        "min_note_dist_ms": 100, "max_notes_per_onset": 3, "virtual": False
    }

    qna_params = {
        "raga_map": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        "instruments": (Instrument("Violin", 1), Instrument("Keys", 1, False)),
        "input_dev_name": "Scarlett 2i2 USB",
        "randomness_temperature": 0.25, # Human 초기값
        "auto_random_temp": False, "auto_random_loop": False, "timeout_sec": 0.8
    }

    demo = ShimonDemo('Portable Grand Port 1', False, 98, qna_params, 
                      { "smoothing": 5, "n_beats_to_track": 8, "timeout_sec": 2, "tempo_range": (50, 120) }, 
                      { "midi_files": [["phrases/intro.mid"]], "gesture_midi_files": [["gestures/intro.mid"]], "note_mapping": [96, 98, 100] }, 
                      performer_params)
    demo.qna_demo.repeat_count = 1 # Human 초기 루프
    demo.run()