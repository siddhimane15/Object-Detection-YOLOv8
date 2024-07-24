# Object Detection with YOLOv8n and Speech Recognition

This project leverages the YOLOv8n model for real-time object detection, integrated with OpenCV for video processing, Pyttsx3 for text-to-speech, and SpeechRecognition for voice command recognition. The system captures video from a webcam, detects objects, and announces detected objects upon receiving a specific voice command.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install the required Python libraries:**
    ```bash
    pip install opencv-python numpy pyttsx3 SpeechRecognition ultralytics
    ```

3. **Yolo v8 model will be installed in the folder automatically

## Usage

1. **Run the script:**
    ```bash
    python3 speech_object_detect_v8.py
    ```

2. The application will start capturing video from your webcam.

3. To activate object detection, say "Jordan" into the microphone. The system will announce the detected objects for 5 seconds.

4. Press `q` to exit the application.

## Project Structure

