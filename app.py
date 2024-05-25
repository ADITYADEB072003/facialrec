from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import face_recognition
import numpy as np
import base64
import os
import pickle
import csv
from datetime import datetime
import logging

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the directory containing student images
images_dir = 'Images/BCA 4'
# Path to store and load encodings
encoding_file = 'Images/BCA 4/known_encodings.pkl'
# Path to the CSV file to record recognized faces
recognized_faces_csv = 'Images/BCA 4/recognized_faces.csv'
# Initialize known_students dictionary
known_students = {}
recognized_faces = set()  # Set to store recognized faces
logged_student_ids = set()  # Set to track logged student IDs

def load_known_students():
    global known_students

    if os.path.exists(encoding_file):
        # Load known_students dictionary from file if it exists
        with open(encoding_file, 'rb') as f:
            known_students = pickle.load(f)
        logging.info("Loaded known students from encoding file.")
    else:
        # Initialize empty known_students dictionary
        known_students = {}

        # Iterate through each subdirectory (student folder) in the images directory
        for subdir in os.listdir(images_dir):
            subdir_path = os.path.join(images_dir, subdir)
            if os.path.isdir(subdir_path):
                # Use folder name directly as student ID
                student_id = subdir
                student_name = subdir  # Use folder name as student name

                student_encodings = []

                # Iterate through each image file in the student folder
                for filename in os.listdir(subdir_path):
                    image_path = os.path.join(subdir_path, filename)
                    # Load image and compute encodings
                    face_encodings = compute_face_encodings(image_path)
                    if face_encodings:
                        student_encodings.extend(face_encodings)

                # If student encodings are computed, update known_students dictionary
                if student_encodings:
                    known_students[student_id] = {
                        'name': student_name,
                        'encodings': student_encodings
                    }

        # Save known_students to file using pickle for future use
        with open(encoding_file, 'wb') as f:
            pickle.dump(known_students, f)
        logging.info("Saved known students to encoding file.")

def compute_face_encodings(image_path):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    return face_encodings

def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations using CNN model for improved accuracy
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not known_students:
        logging.warning("No known encodings available.")
        return frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Initialize match list to store matches for each known encoding
        matches = []

        # Compare face encoding with each known encoding
        for student_id, student_data in known_students.items():
            for known_encoding in student_data['encodings']:
                match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.5)
                matches.append(match[0])  # Append the boolean match result

        # Check if any match is found among known encodings
        if np.any(matches):
            matched_index = np.argmax(matches)  # Get index of first True value
            student_id = list(known_students.keys())[matched_index]
            student_name = known_students[student_id]['name']

            # Draw rectangle around the face and display student ID and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{student_id} - {student_name}"
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Log recognized face in CSV if not already logged
            log_recognized_face(student_id, student_name)
        else:
            # Draw rectangle around the face and label as "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

def log_recognized_face(student_id, student_name):
    global recognized_faces, logged_student_ids
    timestamp = datetime.now()

    # Check if the student ID has already been logged
    if student_id in logged_student_ids:
        logging.info(f"Student ID {student_id} already logged. Skipping.")
        return

    # Generate a unique key for each recognized face
    face_key = f"{student_id}_{student_name}_{timestamp}"

    # Add the face to the set of recognized faces
    recognized_faces.add(face_key)

    # Append the recognized face to the CSV file
    with open(recognized_faces_csv, 'a', newline='') as csvfile:
        fieldnames = ['Student ID', 'Student Name', 'Timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Student ID': student_id, 'Student Name': student_name, 'Timestamp': timestamp})

    # Add the student ID to the set of logged student IDs
    logged_student_ids.add(student_id)
    logging.info(f"Logged recognized face: {student_id}, {student_name} at {timestamp}")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(base64_data):
    img_data = base64.b64decode(base64_data)
    img_array = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    processed_frame = recognize_faces(frame)
    
    _, buffer = cv2.imencode('.jpg', processed_frame)
    frame_data = base64.b64encode(buffer).decode('utf-8')
    emit('message', frame_data)

if __name__ == '__main__':
    load_known_students()
    # Replace 'YOUR_LAPTOP_IP_ADDRESS' with your actual IP address
    socketio.run(app, host='0.0.0.0', debug=True, port=5002)
