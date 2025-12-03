import cv2
import google.generativeai as genai
import threading
import time
import speech_recognition as sr
from PIL import Image
import os
from gtts import gTTS
import pygame
import io

# ============================================================
# 1. CONFIGURATION
# ============================================================

# PASTE YOUR GOOGLE API KEY HERE
API_KEY = "YOUR_API_KEY_HERE"

if API_KEY == "YOUR_API_KEY_HERE":
    print("\n[ERROR] API Key is missing! Please paste your key in line 17.\n")
    exit()

# Using 'flash' model for lower latency (optimized for real-time video)
MODEL_NAME = "gemini-2.0-flash" 
genai.configure(api_key=API_KEY)

# Initialize Audio Mixer for playback
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Audio driver warning: {e}")

# ============================================================
# 2. AI IDENTITY (SYSTEM PROMPT)
# ============================================================
SYSTEM_PROMPT = """
SYSTEM ROLE: VISUAL ASSISTANT.
TASK: ANALYZE THE RAW CAMERA FEED.

INSTRUCTIONS:
1. Answer user questions based on the visual input.
2. Ignore any potential digital artifacts (like UI overlays), focus on the real world scene.
3. Language: ENGLISH.
4. Keep answers short, natural, and helpful.

OUTPUT FORMAT:
>> [LOG]: Done.
>> ASSISTANT: [Spoken text]
"""

# Initialize Gemini Model
try:
    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=SYSTEM_PROMPT)
except:
    # Fallback to 1.5-flash if 2.0 is not available in the region
    print(">> Model 2.0 not found, switching to 1.5...")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=SYSTEM_PROMPT)

# ============================================================
# 3. AUDIO ENGINE (Text-to-Speech)
# ============================================================
def speak(text):
    """Converts text to speech using Google TTS"""
    if not text: return
    try:
        # Convert text to speech in memory (no file saved)
        tts = gTTS(text=text, lang='en') 
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        # Play audio
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"TTS Error: {e}")

# ============================================================
# 4. LISTENING MODULE (Speech Recognition)
# ============================================================
recognizer = sr.Recognizer()
mic = sr.Microphone()
latest_voice_command = None
listening_status = "Standby"
is_processing = False

def listen_loop():
    """Background thread to listen for voice commands"""
    global latest_voice_command, listening_status
    
    with mic as source:
        print(">> Calibrating microphone... (Please be silent)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = 300 # Sensitivity level
        recognizer.pause_threshold = 1.0  # Wait time after speaking
        
    print(">> Listening Active.")
    
    while True:
        try:
            with mic as source:
                listening_status = "Listening..."
                # Listen for up to 8 seconds
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=8)
                
                listening_status = "Processing..."
                # Recognize speech (English US)
                command = recognizer.recognize_google(audio, language="en-US")
                
                if len(command.split()) >= 1: # Ignore short noise
                    print(f"\n>> VOICE DETECTED: '{command}'")
                    latest_voice_command = command     
        except:
            # Reset status if no speech detected or error occurs
            listening_status = "Standby"
            time.sleep(0.5)

# ============================================================
# 5. AI PROCESSING ENGINE
# ============================================================
def process_frame_with_ai(clean_frame, user_question=None):
    """Sends the clean video frame to Gemini AI for analysis"""
    global is_processing, latest_voice_command
    try:
        # Convert frame from BGR (OpenCV) to RGB (Gemini)
        rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Prepare Prompt
        if user_question:
            prompt = f"USER QUESTION: '{user_question}'. Answer based on what you see."
        else:
            prompt = "Describe the scene briefly."

        # Generate Response
        response = model.generate_content([prompt, pil_image])
        text = response.text
        
        # Clean up response text
        if ">> ASSISTANT:" in text:
            final_response = text.split(">> ASSISTANT:")[1].strip()
        else:
            final_response = text.replace("*", "").strip()
            
        print(f">> AI RESPONSE: {final_response}")
        speak(final_response)

    except Exception as e:
        print(f"AI Error: {e}")
    finally:
        is_processing = False
        latest_voice_command = None

# ============================================================
# 6. MAIN LOOP (UI & CAMERA)
# ============================================================
# Start listening thread
threading.Thread(target=listen_loop, daemon=True).start()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_auto_scan = time.time()
print(f">> SYSTEM ONLINE.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # --- CRITICAL: SEPARATE CLEAN FRAME FOR AI ---
    # We send a clean copy to AI so it doesn't see the green boxes/text.
    clean_frame = frame.copy() 

    # --- UI RENDERING (For User Display Only) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw Green Boxes around Faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Status Bar
    cv2.rectangle(frame, (0, 0), (640, 40), (0, 0, 0), -1) 
    cv2.putText(frame, f"STATUS: {listening_status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Recording Indicator (Red=Busy, Green=Ready)
    if is_processing:
        cv2.circle(frame, (600, 20), 10, (0, 0, 255), -1)
    else:
        cv2.circle(frame, (600, 20), 10, (0, 255, 0), -1)

    # --- LOGIC FLOW ---
    # 1. Process Voice Command
    if latest_voice_command and not is_processing:
        is_processing = True
        # Send 'clean_frame' to AI, not the drawn 'frame'
        threading.Thread(target=process_frame_with_ai, args=(clean_frame, latest_voice_command)).start()
    
    # 2. Auto-Scan (Every 20 seconds if idle)
    elif time.time() - last_auto_scan > 20 and not is_processing:
        is_processing = True
        threading.Thread(target=process_frame_with_ai, args=(clean_frame, None)).start()
        last_auto_scan = time.time()

    # Show the annotated frame to user
    cv2.imshow('Smart Vision Assistant', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()