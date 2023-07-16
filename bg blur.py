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

    # Apply Gaussian blur to the frame
    blurred = cv2.GaussianBlur(frame, (99, 99), 0)

    # Process each detected face
    for (x, y, w, h) in face:
        # Crop the detected face region
        face_cr = frame[y:y+h, x:x+w]

        # Apply Gaussian blur to the face region (to keep it clear)
        blurred_face = cv2.GaussianBlur(face_cr, (99, 99), 0)

        # Replace the blurred face region with the clear face region
        blurred[y:y+h, x:x+w] = face_cr

    # Display the resulting frame with the blurred background and clear faces
    cv2.imshow('Webcam', blurred)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()
