import cv2
from flask import Flask, render_template, Response, request, redirect, session
import face_recognition
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Define a predefined username and password (for demonstration purposes)
USERNAME = 'admin'
PASSWORD = 'password'
# Path to the directory containing student images
images_dir = '/Users/adityadebchowdhury/Desktop/Desktop - Aditya’s MacBook Air/opencv2/flask/Student_Images'  # Update with your image directory path

# Initialize known_students dictionary
known_students = {}


def load_known_students():
    global known_students
    for subdir in os.listdir(images_dir):
        subdir_path = os.path.join(images_dir, subdir)
        if os.path.isdir(subdir_path):
            student_id, student_name = subdir.split('_')
            student_images = []
            student_encodings = []

            for filename in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, filename)
                img = cv2.imread(image_path)
                if img is not None:
                    # Convert image to RGB (face_recognition expects RGB)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    rgb_img = cv2.resize(rgb_img, (0, 0), fx=0.5, fy=0.5)

                    # Detect faces and compute encodings
                    face_locations = face_recognition.face_locations(rgb_img)
                    if face_locations:
                        face_encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
                        student_images.append(rgb_img)  # Store RGB image
                        student_encodings.append(face_encoding)
                    else:
                        print(f"No face detected in: {image_path}")
                else:
                    print(f"Error loading image: {image_path}")

            if student_encodings:
                known_students[student_id] = {
                    'name': student_name,
                    'images': student_images,
                    'encodings': student_encodings
                }
            else:
                print(f"No encodings found for student: {student_id}")


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
        for known_id, student_data in known_students.items():
            for known_encoding in student_data['encodings']:
                match = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=0.6)
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
        else:
            # Draw rectangle around the face and label as "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, "Unknown", (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


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
def home():
    return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect('/video_feed')
        else:
            return render_template('login.html', message='Invalid credentials. Please try again.')

    return render_template('login.html')


@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return redirect('/login')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect('/login')


if __name__ == '__main__':
    load_known_students()
    app.run(debug=True, port=5002)
