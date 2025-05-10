import cv2
import cvzone
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.markdown("""
<style>
    button[title^=Exit]+button[title^=Exit] {
        display: none
    }
</style>
""", unsafe_allow_html=True)

# Set the page configuration
st.set_page_config(layout="wide")
# Add custom CSS for the layout
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
        }
        .author {
            color: #0ff;
            font-weight: bold;
            font-size: 1.5em;
            font-family: 'Arial Black', sans-serif;
            text-shadow: 0 0 5px #0ff, 0 0 10px #0ff, 0 0 15px #0ff, 0 0 20px #0ff;
            animation: wave 2s ease-in-out infinite;
        }
        .answer-text {
            font-weight: bold;
            font-family: Arial, sans-serif;
            font-size: 24px;
            color: black;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border: 2px solid #000;
        }
        .answer-title {
            font-family: 'Arial Black', sans-serif;
            color: #2c3e50;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .stButton>button {
            background-color: #e74c3c;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 4px;
            font-weight: bold;
            font-family: 'Arial Black', sans-serif;
        }
        .stButton>button:hover {
            background-color: #c0392b;
        }
        .audio-container {
            margin-top: 10px;
        }
        @keyframes wave {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
    </style>
""", unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Run', value=True, key='run_checkbox')
    FRAME_WINDOW = st.image([])
    stop_button = st.button("Stop Webcam")
with col2:
    st.markdown('<h1 class="answer-title">Answer</h1>', unsafe_allow_html=True)
    output_text_area = st.markdown('<div class="answer-text"></div>', unsafe_allow_html=True)
    st.markdown('<p class="author">Made by RATNAPRAVA MOHAPATRA</p>', unsafe_allow_html=True)

# Configure the generative AI model
genai.configure(api_key="")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
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

    if output_text:
        output_text_area.markdown(f'<div class="answer-text">{output_text}</div>', unsafe_allow_html=True)

    # Check if the stop button was pressed
    if stop_button:
        run = False  # Exit the loop

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)

# Release the webcam and clean up
cap.release()
cv2.destroyAllWindows()
