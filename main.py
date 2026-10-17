import os
import sys
import time
import select
import ctypes
import subprocess
import threading
import json
import requests
import asyncio
import numpy as np
import pyaudio
import websockets
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from groq import Groq

# =====================================================================
# 1. C-LEVEL WARNING SUPPRESSION
# =====================================================================
ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
def py_error_handler(filename, line, function, err, fmt): pass
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
try:
    asound = ctypes.cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception: pass

def silence_stderr():
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return old_stderr

def restore_stderr(old_stderr):
    sys.stderr.flush()
    os.dup2(old_stderr, 2)
    os.close(old_stderr)

# =====================================================================
# 2. CONFIGURATION & API INITIALIZATION
# =====================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

VISION_LAUNCHED = False
RUNNING = True

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def speak(text):
    """Pipes espeak through aplay for non-overlapping execution."""
    print(f"\n[ANIMUS Answer]:\n{text}\n", flush=True)
    try:
        p1 = subprocess.Popen(["espeak", text, "--stdout"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        subprocess.run(["aplay", "-q"], stdin=p1.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p1.wait()
    except Exception:
        pass

# =====================================================================
# 3. WEBSOCKET SERVER THREAD (Communicates with index.html)
# =====================================================================
latest_servo_angles = {"thumb": 180, "index": 180, "middle": 180, "ring": 180, "pinky": 180}

async def websocket_handler(websocket):
    global latest_servo_angles
    print("\n[ANIMUS Vision WS]: Web Vision interface connected!", flush=True)
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "robot_hand_angles":
                latest_servo_angles = data.get("data", {})

            elif msg_type == "gesture_command":
                action = data.get("action")
                print(f"\n⚡ [GESTURE TRIGGER]: Received action -> {action}", flush=True)
                
                if action == "OPEN_TERMINAL":
                    try:
                        subprocess.Popen(["x-terminal-emulator"])
                    except Exception:
                        subprocess.Popen(["garcon-url-handler", "terminal"])

                elif action == "CLOSE_WINDOW":
                    try:
                        subprocess.Popen(["xdotool", "getactivewindow", "windowclose"])
                    except Exception:
                        subprocess.Popen(["xdotool", "key", "ctrl+shift+w"])

                elif action == "OPEN_911_PAGE":
                    subprocess.Popen(["garcon-url-handler", "https://www.911memorial.org"])

    except websockets.exceptions.ConnectionClosed:
        print("\n[ANIMUS Vision WS]: Interface disconnected.", flush=True)

async def start_websocket_server():
    async with websockets.serve(websocket_handler, "localhost", 8765):
        await asyncio.Future()

def run_ws_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_websocket_server())

# Launch WebSocket Server thread AFTER defining the helper functions
ws_loop = asyncio.new_event_loop()
ws_thread = threading.Thread(target=run_ws_loop, args=(ws_loop,), daemon=True)
ws_thread.start()

# =====================================================================
# 4. ROUTING ENGINES & LLM POST-CORRECTION
# =====================================================================
SYS_PROMPT = (
    "You are ANIMUS, an advanced AI assistant powering a wearable interface. "
    "Inputs come from a Speech-to-Text system. Correct obvious phonetic errors or missing small words "
    "based on intent. Provide direct, clear, accurate, and concise answers."
)

HAND_CONTROL_PROMPT = (
    "You are the Robotic Hand Controller for ANIMUS. Translate the user's "
    "gesture or movement request into precise mechanical joint coordinates and servo actions."
)

def ask_groq_fastest(user_input):
    if not groq_client:
        return None
    models = ["llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    for model_id in models:
        try:
            print(f"[ANIMUS Core] Fast response via Groq ({model_id})...", flush=True)
            completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                model=model_id
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Groq Error ({model_id})]: {e}", flush=True)
            continue
    return None

def ask_gemini_smartest(user_input):
    if not client:
        return None
    models = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    for model_name in models:
        try:
            print(f"[ANIMUS Core] Complex reasoning via {model_name}...", flush=True)
            res = client.models.generate_content(
                model=model_name,
                contents=user_input,
                config=types.GenerateContentConfig(system_instruction=SYS_PROMPT)
            )
            if res and hasattr(res, 'text') and res.text:
                return res.text.strip()
        except Exception as e:
            print(f"[Gemini Error ({model_name})]: {e}", flush=True)
            continue
    return None

def ask_ollama_hand_control(user_input):
    print("[ANIMUS Robot Core] Processing hand command via Ollama (gemma2:2b)...", flush=True)
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma2:2b", 
                "prompt": f"{HAND_CONTROL_PROMPT}\nUser Command: {user_input}", 
                "stream": False
            },
            timeout=120
        )
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        else:
            return f"Ollama Error: Status Code {res.status_code}"
    except Exception as e:
        return f"Robot Hand Controller Error: {e}."

def route_query(user_input):
    cmd_lower = user_input.lower()
    hand_keywords = ["hand", "finger", "grip", "robot hand", "servo", "gesture", "pinch", "fist"]
    
    if any(k in cmd_lower for k in hand_keywords):
        return ask_ollama_hand_control(user_input)

    complex_triggers = ["explain", "why", "how", "code", "compare", "analyze", "derive", "solve"]
    is_complex = len(user_input.split()) > 10 or any(w in cmd_lower for w in complex_triggers)

    if is_complex:
        response = ask_gemini_smartest(user_input) or ask_groq_fastest(user_input)
    else:
        response = ask_groq_fastest(user_input) or ask_gemini_smartest(user_input)

    return response or "Cloud engines unreachable. Check network or API keys."

# =====================================================================
# 5. COMMAND DISPATCHER
# =====================================================================
def process_command(command_text):
    global RUNNING, VISION_LAUNCHED
    cmd = command_text.lower().strip()

    if not cmd or len(cmd) < 2:
        return None

    cmd_clean = cmd.rstrip('.?!')

    print(f"\n==========================================")
    print(f"[INPUT RECEIVED]: '{command_text}'")
    print(f"==========================================")

    if cmd_clean in ["status", "system status"]:
        return "All core systems nominal."
    elif cmd_clean in ["time", "what time is it", "what time is", "what time"]:
        return f"The current time is {time.strftime('%I:%M %p')}."
    elif any(k in cmd_clean for k in ["vision", "camera", "open vision"]):
        if not VISION_LAUNCHED:
            VISION_LAUNCHED = True
            try:
                subprocess.Popen(["garcon-url-handler", "/home/mvamathevan0/animus_v2/index.html"])
                return "Dispatching ChromeOS MediaPipe vision interface."
            except Exception as e:
                VISION_LAUNCHED = False
                return f"Failed to launch vision interface: {e}"
        else:
            return "Vision feed active."
    elif cmd_clean in ["shutdown", "exit", "quit", "stop"]:
        RUNNING = False
        return "Shutting down ANIMUS systems."
    else:
        return route_query(command_text)

def drain_mic_buffer(mic_stream, oww_model, chunk_size):
    time.sleep(0.2)
    while mic_stream.get_read_available() > 0:
        mic_stream.read(chunk_size, exception_on_overflow=False)
    oww_model.reset()

def check_keyboard_input():
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip()
    return None

def record_adaptive_audio(mic_stream, rate, chunk):
    frames = []
    silent_chunks = 0
    max_silent_chunks = int(rate / chunk * 0.8)
    max_total_chunks = int(rate / chunk * 10)

    for _ in range(max_total_chunks):
        data = mic_stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
        
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        rms_energy = np.sqrt(np.mean(audio_chunk ** 2))

        if rms_energy < 0.005:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if len(frames) > int(rate / chunk * 1.0) and silent_chunks >= max_silent_chunks:
            break

    for _ in range(4):
        try:
            frames.append(mic_stream.read(chunk, exception_on_overflow=False))
        except Exception:
            pass

    return b''.join(frames)

# =====================================================================
# 6. MAIN VOICE LOOP
# =====================================================================
def main():
    global RUNNING

    print("=======================================================")
    print("  PROJECT ANIMUS v2: ADVANCED STT + WEBSOCKET PIPELINE  ")
    print("=======================================================")

    openwakeword.utils.download_models(['hey_jarvis'])

    print("\n[ANIMUS Core] Loading Faster-Whisper base.en model...")
    whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8", local_files_only=False)

    print("[ANIMUS Core] Loading openWakeWord engine...")
    oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1280

    old_err = silence_stderr()
    try:
        audio = pyaudio.PyAudio()
        mic_stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=1,
            frames_per_buffer=CHUNK
        )
    finally:
        restore_stderr(old_err)

    speak("ANIMUS online and listening.")
    print("\n[INFO]: Say 'Hey Jarvis' or press ENTER/type a prompt in the terminal anytime!\n")
    drain_mic_buffer(mic_stream, oww_model, CHUNK)

    INITIAL_PROMPT = "User voice command to assistant ANIMUS: What time is it? Open vision. System status."

    try:
        while RUNNING:
            text_input = check_keyboard_input()
            if text_input:
                response = process_command(text_input)
                if response:
                    speak(response)
                drain_mic_buffer(mic_stream, oww_model, CHUNK)
                continue

            pcm_data = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_frame = np.frombuffer(pcm_data, dtype=np.int16)

            oww_model.predict(audio_frame)

            score = 0.0
            for key in oww_model.prediction_buffer:
                if len(oww_model.prediction_buffer[key]) > 0:
                    score = max(score, float(oww_model.prediction_buffer[key][-1]))

            print(f"\r[Listening...] Wake Score: {score:.4f} (or type prompt + ENTER)", end="", flush=True)

            if score > 0.45:
                print(f"\n\n⚡ [ANIMUS]: Wake word detected! (Score: {score:.3f}) Listening...", flush=True)
                oww_model.reset()

                raw_audio = record_adaptive_audio(mic_stream, RATE, CHUNK)
                print("[ANIMUS Core] Transcribing audio with Faster-Whisper...", flush=True)

                audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

                max_val = np.max(np.abs(audio_np))
                if max_val > 0:
                    audio_np = audio_np / max_val

                segments, _ = whisper_model.transcribe(
                    audio_np, 
                    language="en",
                    beam_size=5,
                    initial_prompt=INITIAL_PROMPT,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=400),
                    condition_on_previous_text=False
                )
                transcription = " ".join([segment.text for segment in segments]).strip()

                response = process_command(transcription)
                if response:
                    speak(response)

                drain_mic_buffer(mic_stream, oww_model, CHUNK)

    except KeyboardInterrupt:
        print("\n[ANIMUS Core] Stopping audio capture...")
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        audio.terminate()
        print("[ANIMUS Core] Systems offline.")

if __name__ == "__main__":
    main()
