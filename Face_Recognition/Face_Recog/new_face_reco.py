import cv2 as cv
import face_recognition
import pyttsx3
import numpy as np

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load and encode known images
known_faces = []
known_names = []

# Add Amol Sharma
image_amol = face_recognition.load_image_file("img.jpeg")
encoding_amol = face_recognition.face_encodings(image_amol, num_jitters=50, model='large')[0]
known_faces.append(encoding_amol)
known_names.append("Amol Sharma")

# Add Aman Beniwal
image_aman = face_recognition.load_image_file("img1.jpeg")
encoding_aman = face_recognition.face_encodings(image_aman, num_jitters=50, model='large')[0]
known_faces.append(encoding_aman)
known_names.append("Aman Beniwal")

# Add Akshant Nain
image_akshant = face_recognition.load_image_file("img2.jpeg")
encoding_akshant = face_recognition.face_encodings(image_akshant, num_jitters=50, model='large')[0]
known_faces.append(encoding_akshant)
known_names.append("Akshant Nain")

# Launch the live camera
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Camera not working")
    exit()

# Confidence threshold
confidence_threshold = 0.5  # Lowered to improve accuracy

# Dictionary to track recognized faces and prevent duplicate announcements
recognized_faces = {}

# Start processing frames
while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive the frame")
        break

    # Convert the frame to RGB for better face recognition processing
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_faces, face_encoding)
        min_distance = min(distances)
        best_match_index = np.argmin(distances)

        top, right, bottom, left = face_location

        if min_distance < confidence_threshold:
            name = known_names[best_match_index]

            # Prevent repeated announcements
            if name not in recognized_faces:
                engine.say(name)
                engine.runAndWait()
                recognized_faces.clear()  # Clear the dictionary to allow re-announcement after a while
                recognized_faces[name] = True

            # Draw a green rectangle and label
            cv.rectangle(frame, (left, top), (right, bottom), color=(0, 255, 0), thickness=2)
            cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv.LINE_AA)

        else:
            # If no match, display "Not Recognized" and announce once
            if "Unknown" not in recognized_faces:
                engine.say("Not Recognized")
                engine.runAndWait()
                recognized_faces.clear()
                recognized_faces["Unknown"] = True

            # Draw a red rectangle and label
            cv.rectangle(frame, (left, top), (right, bottom), color=(0, 0, 255), thickness=2)
            cv.putText(frame, "Not Recognized", (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

    # Display the frame
    cv.imshow('Video Stream', frame)

    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty('Video Stream', cv.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cam.release()
cv.destroyAllWindows()
