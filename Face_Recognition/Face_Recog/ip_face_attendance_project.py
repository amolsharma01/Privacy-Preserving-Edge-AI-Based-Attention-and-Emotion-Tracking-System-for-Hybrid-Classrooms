import cv2 as cv
import face_recognition
import pyttsx3

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

# Add Face2
image_aman = face_recognition.load_image_file("img1.jpeg")
encoding_aman = face_recognition.face_encodings(image_aman, num_jitters=50, model='large')[0]
known_faces.append(encoding_aman)
known_names.append("Aman Beniwal")

# Add Face3
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

# Flags to manage speech
already_spoken = {}

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

        if min_distance < confidence_threshold:
            recognized = True
            name = known_names[best_match_index]

            # Draw rectangle and label
            top, right, bottom, left = face_location
            cv.rectangle(frame, (left, top), (right, bottom), color=(0, 255, 0), thickness=2)
            cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 0, 0), 2, cv.LINE_AA)

            if name not in already_spoken:
                engine.say(name)
                engine.runAndWait()
                already_spoken.clear()
                already_spoken[name] = True
        else:
            # Not recognized, draw rectangle and label
            top, right, bottom, left = face_location
            cv.rectangle(frame, (left, top), (right, bottom), color=(0, 0, 255), thickness=2)
            cv.putText(frame, "Not Recognized", (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 255), 2, cv.LINE_AA)

    if not recognized and "Unknown" not in already_spoken:
        engine.say("Match not found")
        engine.runAndWait()
        already_spoken.clear()
        already_spoken["Unknown"] = True

    # Display the resulting frame
    cv.imshow('Video Stream', frame)

    # Check if the 'q' key is pressed or the window is closed
    if cv.waitKey(1) & 0xFF == ord('q') or cv.getWindowProperty('Video Stream', cv.WND_PROP_VISIBLE) < 1:
        break

# Release the capture and destroy windows
cam.release()
cv.destroyAllWindows()
