#!/usr/bin/env python3
"""PROJECT ANIMUS v3 - voice brain + WebSocket bridge + local HUD server.

Run:   python3 animus.py        (keep index.html in the same folder)
Say:   "Hey Jarvis" ... "open vision"   |  or type a command + ENTER.

Env:   GEMINI_API_KEY, GROQ_API_KEY            (cloud LLMs, optional)
       GROQ_MODELS, GEMINI_MODELS              (comma lists, check provider docs)
       OLLAMA_MODEL (default gemma2:2b), MIC_INDEX (force input device)
"""
import os
import re
import sys
import json
import time
import queue
import select
import ctypes
import asyncio
import functools
import threading
import subprocess
import http.server
from collections import deque
from pathlib import Path

import numpy as np
import pyaudio
import requests
import websockets
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from groq import Groq

# =====================================================================
# 1. C-LEVEL ALSA WARNING SUPPRESSION
# =====================================================================
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                      ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def _py_error_handler(filename, line, function, err, fmt): pass
_c_error_handler = ERROR_HANDLER_FUNC(_py_error_handler)
try:
    ctypes.cdll.LoadLibrary("libasound.so.2").snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass

def silence_stderr():
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    old = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old

def restore_stderr(old):
    sys.stderr.flush()
    os.dup2(old, 2)
    os.close(old)

# =====================================================================
# 2. CONFIG
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
HTTP_PORT, WS_PORT = 8000, 8765
HUD_URL = f"http://localhost:{HTTP_PORT}/index.html"
ALLOWED_ORIGINS = [f"http://localhost:{HTTP_PORT}", f"http://127.0.0.1:{HTTP_PORT}"]

RATE, CHUNK = 16000, 1280
WAKE_THRESHOLD = 0.45
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")

GROQ_MODELS = os.environ.get("GROQ_MODELS", "llama-3.3-70b-versatile,llama-3.1-8b-instant").split(",")
GEMINI_MODELS = os.environ.get("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite").split(",")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
gemini_client = (genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=20000))
                 if GEMINI_API_KEY else None)
groq_client = Groq(api_key=GROQ_API_KEY, timeout=15) if GROQ_API_KEY else None

RUNNING = True
WAKE_EVENT = threading.Event()          # set by the HUD (peace-sign gesture)
COMMANDS = queue.Queue()                # text commands typed in the HUD
LAST_GOOD = {}                          # remembers which model last worked
HISTORY = deque(maxlen=8)               # short rolling chat memory
latest_servo_angles = {k: 180 for k in ("thumb", "index", "middle", "ring", "pinky")}

# =====================================================================
# 3. HUD BRIDGE: static server (index.html only) + WebSocket
# =====================================================================
class HudHandler(http.server.SimpleHTTPRequestHandler):
    """Serves index.html and nothing else from this folder."""
    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()
        self.send_error(404)
    def log_message(self, *a): pass

def start_http():
    handler = functools.partial(HudHandler, directory=str(BASE_DIR))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

CLIENTS = set()
ws_loop = None

def push(type_, **data):
    """Thread-safe broadcast to every connected HUD."""
    if ws_loop is None or not CLIENTS:
        return
    payload = json.dumps({"type": type_, **data})
    async def _send():
        await asyncio.gather(*[c.send(payload) for c in list(CLIENTS)], return_exceptions=True)
    asyncio.run_coroutine_threadsafe(_send(), ws_loop)

def launch(*cmds):
    for cmd in cmds:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    return False

GESTURE_ACTIONS = {
    "OPEN_TERMINAL": lambda: launch(["x-terminal-emulator"], ["garcon-url-handler", "terminal"]),
    "CLOSE_WINDOW": lambda: launch(["xdotool", "getactivewindow", "windowclose"],
                                   ["xdotool", "key", "ctrl+shift+w"]),
    "OPEN_911_PAGE": lambda: launch(["garcon-url-handler", "https://www.911memorial.org"]),
}

def clamp_angles(d):
    try:
        return {k: max(0, min(180, int(d[k]))) for k in latest_servo_angles}
    except (KeyError, TypeError, ValueError):
        return None

def handle_ws_message(msg):
    global latest_servo_angles
    kind = msg.get("type")
    if kind == "robot_hand_angles":
        angles = clamp_angles(msg.get("data") or {})
        if angles:
            latest_servo_angles = angles
    elif kind == "gesture_command":
        action = GESTURE_ACTIONS.get(msg.get("action"))
        if action:
            print(f"\n⚡ [GESTURE]: {msg.get('action')}", flush=True)
            threading.Thread(target=action, daemon=True).start()
    elif kind == "wake":
        WAKE_EVENT.set()
    elif kind == "text_command":
        text = str(msg.get("text", ""))[:300].strip()
        if text:
            COMMANDS.put(text)

async def ws_handler(ws):
    CLIENTS.add(ws)
    print("\n[ANIMUS WS]: HUD connected.", flush=True)
    try:
        await ws.send(json.dumps({"type": "hello", "wake_word": "hey jarvis"}))
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if isinstance(msg, dict):
                handle_ws_message(msg)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        CLIENTS.discard(ws)
        print("\n[ANIMUS WS]: HUD disconnected.", flush=True)

def start_ws():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    def run():
        asyncio.set_event_loop(ws_loop)
        async def serve():
            async with websockets.serve(ws_handler, "127.0.0.1", WS_PORT,
                                        origins=ALLOWED_ORIGINS, max_size=2**16):
                await asyncio.Future()
        ws_loop.run_until_complete(serve())
    threading.Thread(target=run, daemon=True).start()

# =====================================================================
# 4. SPEECH OUTPUT
# =====================================================================
def clean_for_speech(text, limit=600):
    t = re.sub(r"```.*?```", " code omitted. ", text, flags=re.S)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = re.sub(r"[*_`#>~|]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > limit:
        cut = t[:limit]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
        t = cut[:end + 1] if end > 200 else cut
    return t

def speak(text):
    print(f"\n[ANIMUS Answer]:\n{text}\n", flush=True)
    push("reply", text=text)
    spoken = clean_for_speech(text)
    if not spoken:
        return
    push("state", state="speaking")
    try:
        say = subprocess.Popen(["espeak", "--stdin", "--stdout"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        play = subprocess.Popen(["aplay", "-q"], stdin=say.stdout,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        say.stdout.close()
        say.stdin.write(spoken.encode())
        say.stdin.close()
        play.wait()
        say.wait()
    except Exception:
        pass
    finally:
        push("state", state="idle")

# =====================================================================
# 5. LLM ENGINES (history, timeouts, remembered-good-model)
# =====================================================================
SYS_PROMPT = (
    "You are ANIMUS, an advanced AI assistant powering a wearable interface. "
    "Inputs come from a Speech-to-Text system. Correct obvious phonetic errors or missing small words "
    "based on intent. Answer directly, clearly and concisely, in plain sentences with no markdown, "
    "because your reply is spoken aloud."
)

def _order(models, key):
    good = LAST_GOOD.get(key)
    return [good] + [m for m in models if m != good] if good in models else list(models)

def ask_groq(messages):
    if not groq_client:
        return None
    for m in _order(GROQ_MODELS, "groq"):
        try:
            print(f"[ANIMUS Core] Groq ({m})...", flush=True)
            r = groq_client.chat.completions.create(
                model=m, messages=[{"role": "system", "content": SYS_PROMPT}] + messages)
            LAST_GOOD["groq"] = m
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Groq Error ({m})]: {e}", flush=True)
    return None

def ask_gemini(messages):
    if not gemini_client:
        return None
    contents = [types.Content(role="user" if m["role"] == "user" else "model",
                              parts=[types.Part(text=m["content"])]) for m in messages]
    for m in _order(GEMINI_MODELS, "gemini"):
        try:
            print(f"[ANIMUS Core] Gemini ({m})...", flush=True)
            r = gemini_client.models.generate_content(
                model=m, contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYS_PROMPT))
            if r and r.text:
                LAST_GOOD["gemini"] = m
                return r.text.strip()
        except Exception as e:
            print(f"[Gemini Error ({m})]: {e}", flush=True)
    return None

COMPLEX_RE = re.compile(r"\b(explain|why|how|code|compare|analy[sz]e|derive|solve)\b")

def ask_llm(user_input):
    messages = list(HISTORY) + [{"role": "user", "content": user_input}]
    complex_q = len(user_input.split()) > 10 or COMPLEX_RE.search(user_input.lower())
    engines = (ask_gemini, ask_groq) if complex_q else (ask_groq, ask_gemini)
    reply = None
    for engine in engines:
        reply = engine(messages)
        if reply:
            break
    if not reply:
        return "Cloud engines unreachable. Check network or API keys."
    HISTORY.append({"role": "user", "content": user_input})
    HISTORY.append({"role": "assistant", "content": reply})
    return reply

# =====================================================================
# 6. ROBOT HAND CONTROL (presets first, Ollama JSON as fallback)
# =====================================================================
def _pose(**kw):
    return {k: kw.get(k, 0) for k in ("thumb", "index", "middle", "ring", "pinky")}

PRESETS = {
    "open": _pose(thumb=180, index=180, middle=180, ring=180, pinky=180),
    "fist": _pose(),
    "peace": _pose(index=180, middle=180),
    "point": _pose(index=180),
    "thumbs up": _pose(thumb=180),
    "pinch": _pose(thumb=90, index=90, middle=180, ring=180, pinky=180),
}
PRESET_WORDS = [("thumbs up", "thumbs up"), ("peace", "peace"), ("point", "point"), ("pinch", "pinch"),
                ("fist", "fist"), ("close", "fist"), ("open", "open"), ("release", "open")]
HAND_RE = re.compile(r"\b(hand|finger|fingers|servo|pinch|fist|thumbs up|peace sign)\b")
HAND_CONTROL_PROMPT = (
    "You are the Robotic Hand Controller for ANIMUS. Translate the user's request into servo angles. "
    "Return ONLY JSON with integer keys thumb, index, middle, ring, pinky; each 0 (fully closed) to "
    "180 (fully open)."
)

def hand_command(text):
    t = text.lower()
    angles = next((PRESETS[p] for w, p in PRESET_WORDS if re.search(rf"\b{w}\b", t)), None)
    if angles is None:
        print(f"[ANIMUS Robot Core] Ollama ({OLLAMA_MODEL})...", flush=True)
        try:
            res = requests.post(OLLAMA_URL, timeout=60, json={
                "model": OLLAMA_MODEL, "stream": False, "format": "json",
                "options": {"temperature": 0.1},
                "prompt": f"{HAND_CONTROL_PROMPT}\nCommand: {text}"})
            res.raise_for_status()
            angles = clamp_angles(json.loads(res.json().get("response", "{}")))
        except Exception as e:
            return f"Robot hand controller error: {e}"
    if not angles:
        return "I could not work out that hand pose."
    push("hand_command", angles=angles)
    return "Hand pose sent."

# =====================================================================
# 7. COMMAND DISPATCHER
# =====================================================================
SHUTDOWN_RE = re.compile(r"^(shut ?down|power off|quit|exit) animus$")
VISION_RE = re.compile(r"\b(open|launch|start|show)\b.*\b(vision|camera)\b|^(vision|camera)$")
ESTOP_RE = re.compile(r"\b(emergency stop|e ?stop)\b")
TIME_RE = re.compile(r"^(what time is it|what time is|what time|time)( now)?$")

def process_command(text, typed=False):
    global RUNNING
    if len(text.strip()) < 2:
        return None
    cmd = re.sub(r"[^\w\s]", "", text.lower()).strip()
    print("\n==========================================")
    print(f"[INPUT RECEIVED]: '{text}'")
    print("==========================================", flush=True)
    push("transcript", text=text)
    push("state", state="thinking")

    if SHUTDOWN_RE.match(cmd) or (typed and cmd in ("shutdown", "quit", "exit")):
        RUNNING = False
        return "Shutting down ANIMUS systems."
    if ESTOP_RE.search(cmd):
        push("estop")
        return "Emergency stop sent."
    if cmd in ("status", "system status"):
        return f"All core systems nominal. HUD {'connected' if CLIENTS else 'not connected'}."
    if TIME_RE.match(cmd):
        return f"The current time is {time.strftime('%I:%M %p')}."
    if VISION_RE.search(cmd):
        if CLIENTS:
            return "Vision feed already active."
        ok = launch(["garcon-url-handler", HUD_URL], ["xdg-open", HUD_URL])
        return "Dispatching vision interface." if ok else f"Could not open a browser. Visit {HUD_URL}"
    if HAND_RE.search(cmd):
        return hand_command(text)
    return ask_llm(text)

def handle_text(text, typed=False):
    reply = process_command(text, typed)
    if reply:
        speak(reply)
    else:
        push("state", state="idle")

# =====================================================================
# 8. AUDIO HELPERS
# =====================================================================
HALLUCINATIONS = {"thank you", "thanks for watching", "you", "bye", "thank you."}

def rms(raw):
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0

def measure_ambient(stream, seconds=1.0):
    vals = [rms(stream.read(CHUNK, exception_on_overflow=False)) for _ in range(int(RATE / CHUNK * seconds))]
    return float(np.median(vals))

def drain_mic_buffer(stream, oww):
    time.sleep(0.2)
    while stream.get_read_available() > 0:
        stream.read(CHUNK, exception_on_overflow=False)
    oww.reset()

def check_keyboard_input():
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip()
    return None

def record_adaptive_audio(stream, silence_rms):
    frames, silent = [], 0
    max_silent = int(RATE / CHUNK * 0.8)
    for _ in range(int(RATE / CHUNK * 10)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        silent = silent + 1 if rms(data) < silence_rms else 0
        if len(frames) > int(RATE / CHUNK * 1.0) and silent >= max_silent:
            break
    return b"".join(frames)

def transcribe(whisper, raw):
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 0.02:                       # basically silence: don't let Whisper hallucinate
        return ""
    audio = audio * min(10.0, 0.9 / peak)  # capped gain so room noise is not blown up
    segments, _ = whisper.transcribe(
        audio, language="en", beam_size=5,
        initial_prompt="Voice commands for an assistant named ANIMUS.",
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=400),
        condition_on_previous_text=False)
    text = " ".join(s.text for s in segments if s.no_speech_prob < 0.6).strip()
    return "" if text.lower() in HALLUCINATIONS else text

def pick_mic(audio):
    if os.environ.get("MIC_INDEX"):
        return int(os.environ["MIC_INDEX"])
    return int(audio.get_default_input_device_info()["index"])

# =====================================================================
# 9. MAIN LOOP
# =====================================================================
def main():
    print("=======================================================")
    print("  PROJECT ANIMUS v3: VOICE + HUD BRIDGE                ")
    print("=======================================================")

    start_http()
    start_ws()
    print(f"[ANIMUS Core] HUD: {HUD_URL}   WS: ws://localhost:{WS_PORT}")

    openwakeword.utils.download_models(["hey_jarvis"])
    print("[ANIMUS Core] Loading Faster-Whisper base.en...")
    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")
    oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    old_err = silence_stderr()
    audio = mic = None
    try:
        audio = pyaudio.PyAudio()
        mic = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                         input_device_index=pick_mic(audio), frames_per_buffer=CHUNK)
    finally:
        restore_stderr(old_err)

    speak("ANIMUS online and listening.")
    drain_mic_buffer(mic, oww)
    ambient = measure_ambient(mic)
    silence_rms = max(0.004, ambient * 2.5)
    print(f"[ANIMUS Core] Ambient RMS {ambient:.4f} -> silence threshold {silence_rms:.4f}")
    print("[INFO]: Say 'Hey Jarvis', make a peace sign in the HUD, or type a prompt + ENTER.\n")

    try:
        while RUNNING:
            typed = check_keyboard_input()
            if not typed and not COMMANDS.empty():
                typed = COMMANDS.get_nowait()
            if typed:
                handle_text(typed, typed=True)
                drain_mic_buffer(mic, oww)
                continue

            frame = np.frombuffer(mic.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            oww.predict(frame)
            score = max((float(b[-1]) for b in oww.prediction_buffer.values() if len(b)), default=0.0)
            print(f"\r[Listening...] Wake Score: {score:.4f}   ", end="", flush=True)

            if score > WAKE_THRESHOLD or WAKE_EVENT.is_set():
                WAKE_EVENT.clear()
                print(f"\n\n⚡ [ANIMUS]: Wake! (score {score:.3f}) Listening...", flush=True)
                oww.reset()
                push("state", state="listening")
                raw = record_adaptive_audio(mic, silence_rms)
                push("state", state="thinking")
                text = transcribe(whisper, raw)
                if text:
                    handle_text(text)
                else:
                    push("state", state="idle")
                drain_mic_buffer(mic, oww)
    except KeyboardInterrupt:
        print("\n[ANIMUS Core] Stopping audio capture...")
    finally:
        if mic:
            mic.stop_stream()
            mic.close()
        if audio:
            audio.terminate()
        print("[ANIMUS Core] Systems offline.")

if __name__ == "__main__":
    main()
