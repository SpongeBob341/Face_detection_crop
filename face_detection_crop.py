import cv2

# Load the pre-trained Haar Cascade model for face detection
facemodel = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Create a VideoCapture object to connect to the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default webcam on your system

# Read and display frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is read correctly, ret will be True
    if not ret:
        print("Failed to capture the frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using the Haar Cascade model
    face = facemodel.detectMultiScale(gray)

    # Draw rectangles around the detected faces and crop the face
    for (x, y, w, h) in face:
        # crop the face
        face_roi = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame with detected faces
        cv2.imshow('Detected face', face_roi)
    # Display the resulting frame with rectangles
    cv2.imshow('Webcam', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()
