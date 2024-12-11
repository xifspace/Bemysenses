import tkinter as tk
from tkinter import Text
import os
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import pickle
import numpy as np
from translate import Translator
import pyttsx3
import openai

# Set your OpenAI API key
openai.api_key = 'sk-b2L_R449xOCDuRldZmS129C0RYfCwmCJXq1IVhmcC2T3BlbkFJXRzwpB6TdgTFtQIFpMhEvUfb12Fmb2DYeCZ1Vtx1IA'

# Load pre-trained model for real-time sign language prediction
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Sign language dictionary for translation images
images_folder = r'E:\Miniproject\www.bemysenses.com\Dataset ASL Hand Gestures\grayscale_frames'
sign_language_dict = {
    'A': 'a_1-19.png', 'B': 'b_1-19.png', 'C': 'c_1-19.png', 'D': 'd_1-19.png', 'E': 'e_1-19.png',
    'F': 'f_1-19.png', 'G': 'g_1-19.png', 'H': 'h_1-19.png', 'I': 'i_1-19.png', 'J': 'j_1-19.png',
    'K': 'k_1-19.png', 'L': 'l_1-19.png', 'M': 'm_1-19.png', 'N': 'n_1-19.png', 'O': 'o_1-19.png',
    'P': 'p_1-19.png', 'Q': 'q_1-19.png', 'R': 'r_1-19.png', 'S': 's_1-19.png', 'T': 't_1-19.png',
    'U': 'u_1-19.png', 'V': 'v_1-19.png', 'W': 'w_1-19.png', 'X': 'x_1-19.png', 'Y': 'y_1-19.png',
    'Z': 'z_1-19.png', ' ': 'space.png',
}

# Dictionary for real-time prediction labels
labels_dict = {i: chr(65 + i) for i in range(26)}

# Initialize the speech engine
engine = pyttsx3.init()

# Tkinter main window
root = tk.Tk()
root.title("Be My Senses - Made for VESIT")
root.geometry("800x600")

# Load and set background image
background_image_path = r"E:\Miniproject\www.bemysenses.com\images\vesit.png"
bg_image = Image.open(background_image_path)
bg_image = bg_image.resize((800, 600), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

background_label = tk.Label(root, image=bg_photo)
background_label.place(relwidth=1, relheight=1)

# Variables for controlling prediction
is_paused = False
sentence = ""  # String to store the full sentence
char_frequency = {}  # Dictionary to store the frequency of each character

# Function to toggle prediction pause/resume
def toggle_pause():
    global is_paused
    is_paused = not is_paused
    pause_button.config(text="Resume" if is_paused else "Pause")

# Function for real-time sign language prediction with frequency analysis
def predict_sign_language():
    global is_paused, sentence, char_frequency
    cap = cv2.VideoCapture(0)
    previous_character = ""  # Track the last predicted character to avoid repeats

    def update_frame():
        global sentence  # Make sure to reference the global 'sentence' variable
        nonlocal previous_character  # Reference 'previous_character' from the outer function

        if not is_paused:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )

                        data_aux = []
                        x_, y_ = [], []

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                        # Predict the character
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]

                        # Append only if the character is different from the last one
                        if predicted_character != previous_character:
                            sentence += predicted_character
                            previous_character = predicted_character
                            # Update character frequency
                            char_frequency[predicted_character] = char_frequency.get(predicted_character, 0) + 1

                        # Display prediction on GUI
                        output_text.delete(1.0, tk.END)
                        output_text.insert(tk.END, f"Predicted Character: {predicted_character}\n")
                        output_text.insert(tk.END, f"Current Sentence: {sentence}\n")

                img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                display_label.configure(image=img)
                display_label.image = img

        display_label.after(10, update_frame)

    update_frame()

    # When done, analyze the frequency and print the final refined sentence
    def end_prediction():
        cap.release()
        cv2.destroyAllWindows()

        # Form the refined sentence using the most frequent characters
        sorted_characters = sorted(char_frequency, key=char_frequency.get, reverse=True)
        refined_sentence = ''.join(sorted_characters[:5])  # Use top 5 most frequent characters

        # Generate a meaningful sentence using AI
        meaningful_sentence = create_meaningful_sentence_with_ai(sentence)

        output_text.insert(tk.END, f"\nFinal Sentence: {sentence}\n")
        output_text.insert(tk.END, f"Refined Sentence (Top Characters): {refined_sentence}\n")
        output_text.insert(tk.END, f"Meaningful Sentence: {meaningful_sentence}\n")

    # Button to end prediction and show final sentence
    end_button = tk.Button(root, text="End Prediction", command=end_prediction)
    end_button.pack(pady=10)

# Function to create a more meaningful sentence using GPT-4 AI
def create_meaningful_sentence_with_ai(current_sentence):
    try:
        response = openai.Completion.create(
            engine="text-davinci-004",
            prompt=f"Form a meaningful and coherent sentence based on the following sequence of letters or words: {current_sentence}",
            max_tokens=50,
            temperature=0.7
        )
        # Extract the generated sentence
        refined_sentence = response.choices[0].text.strip()
        return refined_sentence
    except Exception as e:
        print(f"Error while generating sentence with AI: {e}")
        return "Could not generate a refined sentence. Please try again."

# Function to translate text to sign language images
def translate_text_to_sign():
    text_to_translate = text_entry.get()

    # Generate sign language images
    translated_images = []
    for letter in text_to_translate.upper():
        if letter in sign_language_dict:
            image_filename = sign_language_dict[letter]
            image_path = os.path.join(images_folder, image_filename)
            try:
                img = Image.open(image_path)
                translated_images.append(img)
            except FileNotFoundError:
                print(f"Image not found for letter: {letter}")

    # Concatenate and display the translated images in GUI
    if translated_images:
        resized_images = [img.resize((50, 50), Image.LANCZOS) for img in translated_images]
        total_width = sum(img.width for img in resized_images)
        max_height = max(img.height for img in translated_images)
        combined_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in resized_images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width
        img = ImageTk.PhotoImage(combined_image)
        display_label.configure(image=img)
        display_label.image = img

# Function to speak English input text
def speak_english():
    user_input = text_entry.get()
    try:
        engine.say(user_input)
        engine.runAndWait()
    except Exception as e:
        output_text.insert(tk.END, f"Error: {e}\n")

# Function to translate text to Hindi and display
def translate_to_hindi():
    user_input = text_entry.get()
    translator = Translator(to_lang="hi")
    try:
        translated_text = translator.translate(user_input)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Translated Text (Hindi): {translated_text}\n")
    except Exception as e:
        output_text.insert(tk.END, f"Error: {e}\n")

# Function to translate text to Marathi and display
def translate_to_marathi():
    user_input = text_entry.get()
    translator = Translator(to_lang="mr")
    try:
        translated_text = translator.translate(user_input)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Translated Text (Marathi): {translated_text}\n")
    except Exception as e:
        output_text.insert(tk.END, f"Error: {e}\n")

# GUI Layout
title_label = tk.Label(root, text="Be My Senses", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

subtitle_label = tk.Label(root, text="Made for VESIT", font=("Helvetica", 12))
subtitle_label.pack(pady=5)

text_entry = tk.Entry(root, width=50)
text_entry.pack(pady=10)

# Buttons for prediction, translation, and speaking
predict_button = tk.Button(root, text="Real-Time Prediction", command=predict_sign_language)
predict_button.pack(pady=10)

pause_button = tk.Button(root, text="Pause", command=toggle_pause)
pause_button.pack(pady=10)

translate_button = tk.Button(root, text="Translate Text to Sign", command=translate_text_to_sign)
translate_button.pack(pady=10)

speak_english_button = tk.Button(root, text="Speak English", command=speak_english)
speak_english_button.pack(pady=10)

translate_hindi_button = tk.Button(root, text="Translate to Hindi", command=translate_to_hindi)
translate_hindi_button.pack(pady=10)

translate_marathi_button = tk.Button(root, text="Translate to Marathi", command=translate_to_marathi)
translate_marathi_button.pack(pady=10)

# Output text area
output_text = Text(root, height=5, width=50)
output_text.pack(pady=10)

# Display label for images and video
display_label = tk.Label(root)
display_label.pack(pady=10)

# Run the GUI
root.mainloop()
