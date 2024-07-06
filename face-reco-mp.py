import cv2
import mediapipe as mp

cam=cv2.VideoCapture(0)
face_detect = mp.solutions.face_detection.FaceDetection(min_detection_confidence = 0.2)
draw = mp.solutions.drawing_utils

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_detect.process(rgb_frame)
    
    if output.detections:
        for dtc in output.detections:
            draw.draw_detection(frame, dtc)
            winsound.Beep(5000, 500)
            
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
