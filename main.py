import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import streamlit as st
from time import sleep
import random

# Set the page configuration
st.set_page_config(layout="wide")

# Custom CSS with bold black answer text
st.markdown("""
    <style>
        .stApp {
            background-image: url('https://images3.alphacoders.com/134/1349491.jpeg');
            background-size: cover;
        }
        .keyboard-container {
            background-color: rgba(255, 255, 255, 0.7);
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .answer-text {
            font-size: 24px;
            font-weight: bold !important;
            color: #000000 !important;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            line-height: 1.5;
        }
        .quiz-container {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 10px;
            margin-top: 30px;
            border: 2px solid #2c3e50;
        }
        .quiz-question {
            font-size: 22px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }
        .quiz-answer {
            font-size: 20px;
            background-color: rgba(234, 229, 219, 0.7);
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            cursor: pointer;
        }
        .quiz-answer:hover {
            background-color: rgba(200, 230, 255, 0.7);
        }
        .author {
            color: #0ff;
            font-weight: bold;
            font-size: 1.5em;
            font-family: 'Arial Black', sans-serif;
            text-shadow: 0 0 5px #0ff, 0 0 10px #0ff, 0 0 15px #0ff, 0 0 20px #0ff;
            animation: wave 2s ease-in-out infinite;
        }
        @keyframes wave {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize Gemini
genai.configure(api_key="Add your api key")
model = genai.GenerativeModel('gemini-1.5-flash')


# Virtual Keyboard Class
class Button:
    def __init__(self, pos, text, size=[60, 60]):
        self.pos = pos
        self.size = size
        self.text = text

    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (234, 229, 219), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), (128, 0, 0), 2)
        cv2.putText(img, self.text, (x + 15, y + 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (128, 0, 0), 3)
        return img


# Keyboard layout
keys = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'],
    [' ', '(', ')', '+', '-', '*', '/', '=', 'DEL', 'SOLVE']
]
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        pos = [70 + j * 70, 100 + i * 70]
        size = [60, 60] if key != ' ' else [200, 60]
        buttonList.append(Button(pos, key, size))

# Streamlit layout
col1, col2 = st.columns([3, 2])
with col1:
    run = st.checkbox('Enable Camera', True)
    FRAME_WINDOW = st.image([])
    input_mode = st.radio("Input Mode:", ["Gesture Drawing", "Virtual Keyboard"])
with col2:
    st.markdown("### Solution")
    solution_area = st.empty()
    st.markdown("### Current Input")
    input_display = st.empty()
    st.markdown('<p class="author">Made by RATNAPRAVA MOHAPATRA</p>', unsafe_allow_html=True)

# Initialize webcam and detector
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

final_text = ""
canvas = None
prev_pos = None
drawing = False  # Track drawing state

# GK Quiz Game State
if 'quiz_question' not in st.session_state:
    st.session_state.quiz_question = ""
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = []
if 'quiz_correct' not in st.session_state:
    st.session_state.quiz_correct = ""
if 'quiz_feedback' not in st.session_state:
    st.session_state.quiz_feedback = ""
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0


def generate_gk_question():
    """Generate a new GK question using Gemini"""
    topics = [
        "science trivia", "history facts", "geography questions",
        "fun math puzzles", "pop culture trivia", "animal kingdom facts"
    ]
    topic = random.choice(topics)
    prompt = f"Generate an interesting general knowledge question about {topic} with 4 multiple choice answers. " \
             "Provide the correct answer. Format your response like this:\n" \
             "Question: [question text]\n" \
             "A) [option 1]\n" \
             "B) [option 2]\n" \
             "C) [option 3]\n" \
             "D) [option 4]\n" \
             "Correct Answer: [letter]"

    response = model.generate_content(prompt)
    return response.text


def parse_quiz_response(response_text):
    """Parse the Gemini response into question, answers, and correct answer"""
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    question = ""
    answers = []
    correct = ""

    for line in lines:
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
        elif line.startswith(("A)", "B)", "C)", "D)")):
            answers.append(line)
        elif line.startswith("Correct Answer:"):
            correct = line.replace("Correct Answer:", "").strip()

    return question, answers, correct


def check_answer(selected):
    """Check if the selected answer is correct"""
    if selected == st.session_state.quiz_correct:
        st.session_state.quiz_feedback = "✅ Correct! Well done!"
        st.session_state.quiz_score += 1
    else:
        st.session_state.quiz_feedback = f"❌ Incorrect. The correct answer was {st.session_state.quiz_correct}"


def new_question():
    """Generate a new question"""
    quiz_response = generate_gk_question()
    question, answers, correct = parse_quiz_response(quiz_response)
    st.session_state.quiz_question = question
    st.session_state.quiz_answers = answers
    st.session_state.quiz_correct = correct
    st.session_state.quiz_feedback = ""


# Generate first question
if not st.session_state.quiz_question:
    new_question()

# GK Quiz Section
st.markdown("---")
with st.container():
    st.markdown("### 🧠 Fun GK Quiz Challenge (BY RATNAPRAVA)")
    st.markdown(f"<div class='quiz-container'>"
                f"<div class='quiz-question'>{st.session_state.quiz_question}</div>",
                unsafe_allow_html=True)

    for answer in st.session_state.quiz_answers:
        if st.button(answer, key=answer, use_container_width=True,
                     on_click=check_answer, args=(answer[0],)):
            pass

    if st.session_state.quiz_feedback:
        st.markdown(f"**{st.session_state.quiz_feedback}**")
        st.markdown(f"**Score:** {st.session_state.quiz_score}")

    if st.button("Next Question ➡️", use_container_width=True):
        new_question()

# Main application loop
while run:
    success, img = cap.read()
    if not success:
        continue
    img = cv2.flip(img, 1)

    if input_mode == "Gesture Drawing":
        if canvas is None:
            canvas = np.zeros_like(img)

        hands, img = detector.findHands(img, draw=False)

        if hands:
            hand = hands[0]
            fingers = detector.fingersUp(hand)
            lmList = hand['lmList']

            # Get the index finger tip position
            index_finger = lmList[8][0:2]

            if fingers == [0, 1, 0, 0, 0]:  # Drawing gesture
                if not drawing:
                    # Start new drawing
                    prev_pos = index_finger
                    drawing = True
                else:
                    # Continue drawing
                    if prev_pos:
                        distance = np.linalg.norm(np.array(index_finger) - np.array(prev_pos))
                        if distance < 50:  # Only draw if movement is small
                            cv2.line(canvas, prev_pos, index_finger, (255, 0, 255), 10)
                        prev_pos = index_finger
            else:
                # Not in drawing gesture mode
                drawing = False
                prev_pos = None

            if fingers == [1, 0, 0, 0, 0]:  # Clear canvas
                canvas = np.zeros_like(img)
                drawing = False
                prev_pos = None

            if fingers == [1, 1, 1, 1, 0]:  # Solve
                pil_image = Image.fromarray(canvas)
                response = model.generate_content(["Solve this math problem:", pil_image])
                solution_area.markdown(
                    f'<div class="answer-text"><strong>{response.text}</strong></div>',
                    unsafe_allow_html=True
                )
                sleep(1)
        else:
            # No hands detected
            drawing = False
            prev_pos = None

        img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

    elif input_mode == "Virtual Keyboard":
        hands, img = detector.findHands(img, draw=True)

        for button in buttonList:
            img = button.draw(img)

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 15, y + 40),
                                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

                    p1 = lmList[8][0:2]
                    p2 = lmList[12][0:2]
                    distance = np.linalg.norm(np.array(p1) - np.array(p2))

                    if distance < 30:
                        if button.text == "DEL":
                            final_text = final_text[:-1]
                        elif button.text == "SOLVE":
                            response = model.generate_content(f"Solve: {final_text}")
                            solution_area.markdown(
                                f'<div class="answer-text"><strong>{response.text}</strong></div>',
                                unsafe_allow_html=True
                            )
                            final_text = ""
                        else:
                            final_text += button.text
                        sleep(0.3)

        cv2.rectangle(img, (70, 20), (700, 80), (234, 229, 219), cv2.FILLED)
        cv2.putText(img, final_text, (80, 70), cv2.FONT_HERSHEY_PLAIN, 4, (128, 0, 0), 4)
        input_display.text(final_text)

    FRAME_WINDOW.image(img, channels="BGR")

cap.release()
cv2.destroyAllWindows()
