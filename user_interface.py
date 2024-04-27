import cv2
import tkinter as tk
from tkinter import messagebox
from keras.models import load_model
import numpy as np
import threading

# Load the trained models
model_isl = load_model('model_isl.h5')
model_asl = load_model('model_asl.h5')

# Initialize Tkinter
root = tk.Tk()
root.title("Sign Language Detection")

# Function to detect signs
def detect_signs(language):
    cap = cv2.VideoCapture(0)

    if language == 'ISL':
        model = model_isl
    elif language == 'ASL':
        model = model_asl
    else:
        messagebox.showwarning("Warning", "Invalid language selection.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)

        # Predict the sign
        sign = predict_sign(model, processed_frame)

        # Display the sign on the frame
        cv2.putText(frame, sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sign Language Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to preprocess the frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (400, 400))
    normalized_frame = resized_frame.astype('float32') / 255.0
    expanded_frame = np.expand_dims(normalized_frame, axis=0)
    return expanded_frame

# Function to predict the sign
def predict_sign(model, frame):
    predictions = model.predict(frame)
    sign_index = np.argmax(predictions)
    sign = chr(ord('A') + sign_index)
    return sign

# Start detection
def start_detection(language):
    threading.Thread(target=detect_signs, args=(language,)).start()

# Create GUI elements
language_var = tk.StringVar()
language_var.set("Select Language")
language_menu = tk.OptionMenu(root, language_var, "ISL", "ASL")
language_menu.pack()

start_button = tk.Button(root, text="Start Detection", command=lambda: start_detection(language_var.get()))
start_button.pack()

# Start the GUI
root.mainloop()

