import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json, Sequential
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Load the model architecture from JSON file
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Explicitly register the Sequential class
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({"Sequential": Sequential})

# Load the model weights
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Function to process video frames and detect emotions
def detect_emotions():
    global running
    if not running:
        return

    ret, frame = cap.read()
    if not ret:
        return

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.expand_dims(np.expand_dims(roi_gray, -1), 0)
        img_pixels = img_pixels / 255.0

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if running:
        video_label.after(10, detect_emotions)

# Function to close the application
def close_application():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("Real-Time Emotion Detection")
window.geometry("800x600")

# Create a label to display the video stream
video_label = tk.Label(window)
video_label.pack()

# Create a button to close the application
close_button = tk.Button(window, text="Close", command=close_application)
close_button.pack()

# Start the emotion detection process in a separate thread
running = True
threading.Thread(target=detect_emotions).start()

# Run the Tkinter event loop
window.mainloop()
