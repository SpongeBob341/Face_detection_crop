import cv2

# variables
# distance from camera to object(face) measured
KNOWN_DISTANCE = 40.8 #cm
# width of face in the real world or Object Plane
KNOWN_WIDTH = 14.7 #cm
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
cap = cv2.VideoCapture(0)

# face detector object
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


# focal length finder function
def focal_length(measured_distance, real_width, width_in_rf_image):

    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


# distance estimation function
def distance_finder(focal_length, real_face_width, face_width_in_frame):

    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


# face detector function
def face_data(image):

    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_image)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), WHITE, 1)
        face_width = w

    return face_width


# reading reference image from directory
ref_image = cv2.imread("Ref_image.jpg")

ref_image_face_width = face_data(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
print(focal_length_found)


while True:
    ret, frame = cap.read()

    # calling face_data function
    face_width_in_frame = face_data(frame)
    # finding the distance by calling function Distance
    if face_width_in_frame != 0:
        Distance = distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)
        # Drawing Text on the screen
        cv2.putText(
            frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, WHITE, 2
        )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()