import cv2

# Load pre-trained cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

min_eye_distance = 45  # Minimum distance between eyes for accurate detection
min_neighbors_face = 5  # Minimum neighbors for face detection
min_neighbors_eye = 5  # Minimum neighbors for eye detection
threshold = 0.6 # Threshold for considering if facing the screen
screen_ratio = 0.5 # Ratio of face width to consider as facing the screen

# Start video capture from webcam
capture = cv2.VideoCapture(0)

total_frames = 0
facing_screen_frames = 0
not_facing_screen_frames = 0

while True:
    # Read a frame from the video capture
    ret, frame = capture.read()

    if not ret:
        break

    # Increment total frames count
    total_frames += 1

    # Convert frame to grayscale for better performance in face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=min_neighbors_face)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Define region of interest (ROI) for eye detection within the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=min_neighbors_eye)

        # Check if both eyes are detected within the face region
        if len(eyes) == 2:
            # Calculate the distance between the eyes
            eye_distances = [abs(ex1 - ex2) for (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) in zip(eyes, eyes[1:])]
            min_eye_distance_detected = min(eye_distances)

            # Check if the minimum eye distance meets the threshold
            if min_eye_distance_detected >= min_eye_distance:
                # Calculate the ratio of face width
                face_width_ratio = (eyes[-1][0] + eyes[-1][2] - eyes[0][0]) / w

                # Check if the face width ratio meets the threshold for facing the screen
                if face_width_ratio >= screen_ratio:
                    facing_screen_frames += 1
                else:
                    not_facing_screen_frames += 1

    # Display the frame with face and eye detection
    # cv2.imshow('Gaze Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

# Calculate the percentage of frames where the candidate is facing the screen
percentage_facing_screen = round((facing_screen_frames / total_frames) * 100, 2)
print("Percentage of Time Facing the Screen:", percentage_facing_screen)
