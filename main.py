import sys
import subprocess

# Install missing packages automatically
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import cvzone
except ImportError:
    install("cvzone==1.5.6")
    import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
from gtts import gTTS
import os
import base64
from io import BytesIO
import tempfile

# Set the page configuration with custom styling
st.set_page_config(layout="wide")

# Add custom CSS for the layout with neon effects
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://images3.alphacoders.com/134/1349491.jpeg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .webcam-container {
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            position: relative;
        }
        .author {
            color: #0ff;
            font-weight: bold;
            font-size: 1.3em;
            font-family: Arial Black, sans-serif;
            text-shadow: 0 0 5px #0ff, 0 0 10px #0ff;
            animation: fadeIn 2s ease-in, neonGlowAuthor 1.5s infinite alternate;
            opacity: 0;
            animation-fill-mode: forwards;
        }
        .answer-text {
            font-weight: bold;
            font-family: Arial Black, sans-serif;
            font-size: 20px;
            color: black;
            background-color: white; /* Changed to solid white */
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .answer-title {
            font-family: Arial Black, sans-serif;
            color: #2c3e50;
        }
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 4px;
            font-weight: bold;
            font-family: Arial Black, sans-serif;
        }
        .stButton>button:hover {
            background-color: #c0392b;
        }
        .audio-container {
            margin-top: 10px;
        }
        @keyframes neonGlowAuthor {
            from {
                text-shadow: 0 0 5px #0ff, 0 0 10px #0ff;
            }
            to {
                text-shadow: 0 0 10px #0ff, 0 0 20px #0ff, 0 0 30px #0ff;
            }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# Function to convert text to speech with children's voice
def text_to_speech(text):
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_filename = fp.name

        # Generate speech with child's voice (using 'co.uk' domain which often has clearer voices)
        tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)

        # Save to temporary file
        tts.save(temp_filename)

        # Read the audio file into BytesIO
        audio_bytes = BytesIO()
        with open(temp_filename, 'rb') as f:
            audio_bytes.write(f.read())
        audio_bytes.seek(0)

        # Clean up the temporary file
        os.unlink(temp_filename)

        # Create audio HTML with autoplay
        audio_str = "data:audio/mp3;base64," + base64.b64encode(audio_bytes.read()).decode()
        audio_html = f"""
            <audio autoplay class="audio-container">
                <source src="{audio_str}" type="audio/mp3">
            </audio>
        """
        return audio_html
    except Exception as e:
        st.error(f"Error in text-to-speech: {e}")
        return None


# Create columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    # Webcam container with black background
    st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
    stop_button = st.button("Stop Webcam")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<h1 class="answer-title">Answer</h1>', unsafe_allow_html=True)
    output_text_area = st.markdown("", unsafe_allow_html=True)
    audio_placeholder = st.empty()
    st.markdown('<p class="author">Made by RATNAPRAVA MOHAPATRA</p>', unsafe_allow_html=True)

# Configure the generative AI model
genai.configure(api_key="AIzaSyABuF2mybjDu1KSixpoeEO7BakU7NI7uyA")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0,cv2.CAP_ANDROID)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        print(fingers)
        return fingers, lmList
    else:
        return None


def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
    return current_pos, canvas


def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text


prev_pos = None
canvas = None
output_text = ""
last_spoken_text = ""

# Continuously get frames from the webcam
while run:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text and output_text != last_spoken_text:
        output_text_area.markdown(f'<div class="answer-text">{output_text}</div>', unsafe_allow_html=True)

        # Generate and play audio with child's voice
        audio_html = text_to_speech(output_text)
        if audio_html:
            audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

        last_spoken_text = output_text

    if stop_button:
        run = False

    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()