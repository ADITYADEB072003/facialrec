import cv2
from flask import Flask, render_template, Response
import face_recognition
import os
import numpy as np
import pickle
import hashlib
import csv
from datetime import datetime

app = Flask(__name__)

# Path to the directory containing student images
images_dir = '/Users/adityadebchowdhury/Desktop/Desktop - Aditya’s MacBook Air/opencv2/flask/testcode/Images/BCA 4'
# Path to store and load encodings
encoding_file = '/Users/adityadebchowdhury/Desktop/Desktop - Aditya’s MacBook Air/opencv2/flask/testcode/Images/BCA 4/known_encodings.pkl'
# Path to the CSV file to record recognized faces
recognized_faces_csv = '/Users/adityadebchowdhury/Desktop/Desktop - Aditya’s MacBook Air/opencv2/flask/testcode/Images/BCA 4/recognized_faces.csv'
# Initialize known_students dictionary
known_students = {}
recognized_faces = set()  # Set to store recognized faces

def compute_image_hash(image_path):
    """Compute and return the hash of an image file."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_hash = hashlib.md5(image_data).hexdigest()
    return image_hash

def load_known_students():
    global known_students

    if os.path.exists(encoding_file):
        # Load known_students dictionary from file if it exists
        with open(encoding_file, 'rb') as f:
            known_students = pickle.load(f)
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

                    # Check if image hash has changed or not previously stored
                    image_hash = compute_image_hash(image_path)
                    if image_hash not in known_students.get(student_id, {}).get('image_hashes', []):
                        # Load image and compute encodings
                        face_encodings = compute_face_encodings(image_path)
                        if face_encodings:
                            student_encodings.extend(face_encodings)
                            known_students.setdefault(student_id, {}).setdefault('image_hashes', []).append(image_hash)

                # If student encodings are computed, update known_students dictionary
                if student_encodings:
                    known_students[student_id] = {
                        'name': student_name,
                        'encodings': student_encodings,
                        'image_hashes': known_students[student_id].get('image_hashes', [])
                    }

        # Save known_students to file using pickle for future use
        with open(encoding_file, 'wb') as f:
            pickle.dump(known_students, f)

def compute_face_encodings(image_path):
    img = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    return face_encodings

def recognize_faces(frame, known_encodings):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations using CNN model for improved accuracy
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not known_encodings or len(known_encodings) == 0:
        print("No known encodings provided.")
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
from datetime import timedelta

# Define a time window within which similar faces won't be recorded again (e.g., 5 minutes)
TIME_WINDOW = timedelta(minutes=5)

def log_recognized_face(student_id, student_name):
    global recognized_faces
    timestamp = datetime.now()

    # Generate a unique key for each recognized face
    face_key = f"{student_id}"

    # Check if the face has already been recognized within the time window
    for recognized_face in recognized_faces:
        face_id, _, face_timestamp = recognized_face.split('_')
        face_timestamp = datetime.fromisoformat(face_timestamp)

        if face_id == student_id and timestamp - face_timestamp < TIME_WINDOW:
            # If the same student ID has been recognized recently, skip recording
            return

    # Add the recognized face to the set of recognized faces
    recognized_faces.add(f"{face_key}_{student_name}_{timestamp}")

    # Append the recognized face to the CSV file
    with open(recognized_faces_csv, 'a', newline='') as csvfile:
        fieldnames = ['Student ID', 'Student Name', 'Timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writerow({'Student ID': student_id, 'Student Name': student_name, 'Timestamp': timestamp})


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get known encodings from global scope
        known_encodings = [student_data['encodings'] for student_data in known_students.values()]

        processed_frame = recognize_faces(frame, known_encodings)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_known_students()
    app.run(debug=True, port=5002)
