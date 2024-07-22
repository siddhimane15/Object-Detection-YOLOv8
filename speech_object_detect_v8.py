#!/usr/bin/python3
import cv2
import numpy as np
import pyttsx3
import threading
import speech_recognition as sr
import time
from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")

# Load the classes and colors
#classes_path = r'C:\Users\ADMIN\Desktop\python_scripts\yolov8n.txt'
#with open(classes_path, 'r') as f:
    #classes = [line.strip() for line in f.readlines()]

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Initialize text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

# Flag to determine whether to handle detected objects
handle_objects = False

def draw_prediction(img, class_id, confidence, box):
    x, y, w, h = map(int, box)
    label = f"{classes[class_id]}: {confidence:.2f}"
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    text = f"There is a {classes[class_id]} in front of you."
    print(text)
    engine.say(text)
    engine.runAndWait()
    time.sleep(0.1)

def handle_detected_objects(results):
    detected_objects = []
    if handle_objects:
        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            confidences = result.boxes.conf  # Confidence scores
            class_ids = result.boxes.cls.int()  # Class IDs

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                draw_prediction(image, class_id, confidence, box)
                detected_objects.append(classes[class_id])
    return detected_objects

def speech_recognition_thread():
    global handle_objects
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        with microphone as source:
            print("Listening for the command...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print("You said:", command)
            if "watson" in command:
                engine.say("Observing the surrounding area. Please wait.")
                engine.runAndWait()
                handle_objects = True
                time.sleep(5)  # Wait for 5 seconds for observation
                handle_objects = False
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"Speech recognition request failed: {e}")

# Start the thread for speech recognition
speech_thread = threading.Thread(target=speech_recognition_thread)
speech_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.resize(frame, (1280, 720))
    results = model(image)

    detected_objects = handle_detected_objects(results)

    cv2.imshow("object detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
