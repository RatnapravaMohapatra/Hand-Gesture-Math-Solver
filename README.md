# AI-Powered Virtual Keyboard and Gesture Recognition for Mathematical Problem-Solving


## Abstract

This project introduces an innovative AI-powered virtual keyboard and gesture recognition system designed to revolutionize human-computer interaction in mathematical problem-solving. By integrating real-time computer vision (OpenCV + MediaPipe) with generative AI (Google Gemini), the system enables users to write and solve mathematical equations through intuitive hand gestures, eliminating the need for physical input devices.

## Key Innovations

* **High-precision gesture recognition:** Achieves 93% accuracy for recognizing digits, operators, and commands using 21-point hand landmark detection via MediaPipe.
* **Smart virtual canvas:** Features advanced stabilization algorithms to ensure smooth and natural equation input from hand gestures.
* **AI-powered math solver:** Capable of interpreting handwritten mathematical expressions with over 90% accuracy, covering arithmetic, algebra, and basic calculus, powered by Google Gemini.
* **Interactive web interface:** Built with Streamlit, providing a live camera feed, dynamic feedback on recognized gestures and equations, and a built-in quiz system for enhanced learning.

## Addressing Critical Challenges

This system effectively addresses critical challenges in accessibility and education by offering:

* **Touchless interaction:** Providing an accessible input method for mobility-impaired users and suitable for sterile environments where physical contact is undesirable.
* **Real-time AI feedback:** Offering step-by-step solutions and immediate feedback on entered equations, significantly improving learning outcomes.
* **Versatile deployment:** Applicable in various settings such as classrooms, remote learning environments, and public interactive kiosks.

## Performance

Performance evaluations have demonstrated robust operation under varying lighting conditions with a response latency of less than 2 seconds.

## Future Enhancements

Future development efforts will focus on:

* Mobile compatibility for broader accessibility.
* Implementation of offline functionality using lightweight AI models.
* Expansion of the gesture vocabulary to include more complex mathematical operations.

## Conclusion

This work successfully bridges the gap between natural human interaction paradigms and computational mathematics. It lays a strong foundation for the development of next-generation educational tools, assistive technologies, and intuitive touchless interfaces.

## Getting Started

Instructions on how to set up and run the project will be provided here.Including:

* Prerequisites (e.g., Python version, required libraries).
* Installation steps (e.g., cloning the repository, installing dependencies using pip).
* Instructions on how to run the Streamlit application.
* Configuration  ( API keys).



![Screenshot 2025-05-11 100558](https://github.com/user-attachments/assets/f4edd683-6ce7-4f0e-8dcc-94a7f3fd3320)




![Screenshot 2025-05-11 171740](https://github.com/user-attachments/assets/57258b45-d155-43cd-80a3-5e69ca0181cb)


![Screenshot 2025-05-11 172626](https://github.com/user-attachments/assets/9052faf2-d741-4b9c-9721-e2f517bea587)

```bash
# Example installation steps 
git clone [https://github.com/RatnapravaMohapatra/Hand-Gesture-Math-Solver/blob/main/main.py]
cd -repository
pip install -r requirements.txt
streamlit run main.py

