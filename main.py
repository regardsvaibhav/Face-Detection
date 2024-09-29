import cv2

# Load the pre-trained face detector model
face_cap = cv2.CascadeClassifier("C:/Users/hp/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Start video capture
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    
    # Convert the video frame to grayscale
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the video feed with rectangles drawn around faces
    cv2.imshow("video_live", video_data)
    
    # Exit the loop when the 'a' key is pressed
    if cv2.waitKey(10) == ord("a"):
        break

# Release the video capture and close any OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
