import cv2 as cv
import face_recognition
import pyttsx3
import pandas as pd
from datetime import datetime
import os

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

# Add Vanshul Rana
image_aman = face_recognition.load_image_file("img3.jpg")
encoding_aman = face_recognition.face_encodings(image_aman, num_jitters=50, model='large')[0]
known_faces.append(encoding_aman)
known_names.append("Vanshul Rana")

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
confidence_threshold = 0.6

# Flags to manage speech and attendance
already_spoken = {}
attendance_set = set()
screenshot_dir = "screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# Prepare or load the Excel file
excel_file = "attendance.xlsx"
if os.path.exists(excel_file):
    df_attendance = pd.read_excel(excel_file)
else:
    df_attendance = pd.DataFrame(columns=["Name", "Time"])

# Start processing frames
while True:
    ret, frame = cam.read()
    if not ret:
        print("Can't receive the frame")
        break

    # Face detection in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    recognized = False

    for face_encoding, face_location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_faces, face_encoding)
        min_distance = min(distances)
        best_match_index = distances.tolist().index(min_distance)

        top, right, bottom, left = face_location

        if min_distance < confidence_threshold:
            recognized = True
            name = known_names[best_match_index]

            # Draw rectangle and label
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if name not in already_spoken:
                engine.say(name)
                engine.runAndWait()
                already_spoken.clear()
                already_spoken[name] = True

            # Save attendance and screenshot only once
            if name not in attendance_set:
                attendance_set.add(name)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df_attendance.loc[len(df_attendance)] = [name, now]
                screenshot_path = os.path.join(screenshot_dir, f"{name.replace(' ', '_')}.png")
                cv.imwrite(screenshot_path, frame)
                print(f"Attendance recorded for {name}")
        else:
            # Not recognized
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.putText(frame, "Not Recognized", (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if not recognized and "Unknown" not in already_spoken:
        engine.say("Match not found")
        engine.runAndWait()
        already_spoken.clear()
        already_spoken["Unknown"] = True

    # Show the video stream
    cv.imshow('Video Stream', frame)

    # Exit on 'q' key press or window close
    if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty('Video Stream', cv.WND_PROP_VISIBLE) < 1:
        break

# Save final attendance to Excel
df_attendance.to_excel(excel_file, index=False)
cam.release()
cv.destroyAllWindows()
