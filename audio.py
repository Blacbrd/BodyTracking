import json
import threading
import time
import os

import vosk
import pyaudio

def _audio_callback_factory(audio_queue):

    def audio_callback(in_data, frame_count, time_info, status_flags):
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    return audio_callback

def _speech_recognition_worker(model, state, grammar):
    recognizer = vosk.KaldiRecognizer(model, 16000, grammar)

    while True:
        data = state.audio_queue.get()

        if data is None:
            # shutdown signal
            break

        if recognizer.AcceptWaveform(data):
            result_json = recognizer.Result()
            result = json.loads(result_json)
            text = result.get("text", "")

        else:
            partial_json = recognizer.PartialResult()
            result = json.loads(partial_json)
            text = result.get("partial", "")

        lower_text = text.lower()

        if "calibrate" in lower_text and ("arm" in lower_text or "leg" in lower_text or "head" in lower_text or "box" in lower_text):
            command = ""

            if "arm" in lower_text:
                command = "calibrate_arms"

            elif "head" in lower_text:
                command = "calibrate_head"

            elif "box" in lower_text:
                command = "calibrate_box"

            elif "leg" in lower_text:
                command = "calibrate_legs"

            if command:
                print("Voice command detected:", text)
                state.command_queue.put(command)
                time.sleep(0.2)
                recognizer.Reset()

        if "change" in lower_text and ("hand" in lower_text or "body" in lower_text):

            if "hand" in lower_text:
                print("Voice command detected:", text)
                state.command_queue.put("change_hand")
                time.sleep(0.2)
                recognizer.Reset()

            elif "body" in lower_text:
                print("Voice command detected:", text)
                state.command_queue.put("change_body")
                time.sleep(0.2)
                recognizer.Reset()

def start_audio(state, vosk_model_path):
    """
    Start the audio capturing & recognition background thread.
    Returns (p, stream) which the caller should close/terminate on exit.
    """
    if not os.path.exists(vosk_model_path):
        raise FileNotFoundError(f"VOSK model path not found: {vosk_model_path}")

    model = vosk.Model(vosk_model_path)

    words = [
        "recalibrate",
        "calibrate",
        "arms",
        "arm",
        "legs",
        "leg",
        "inventory",
        "open",
        "close",
        "opened",
        "closed",
        "head",
        "change",
        "hand",
        "body",
        "box"
    ]

    grammar = json.dumps(words)

    p = pyaudio.PyAudio()

    audio_callback = _audio_callback_factory(state.audio_queue)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=3000,
                    stream_callback=audio_callback)

    stream.start_stream()

    speech_thread = threading.Thread(target=_speech_recognition_worker, args=(model, state, grammar), daemon=True)
    speech_thread.start()

    return p, stream
