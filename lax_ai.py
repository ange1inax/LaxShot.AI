import cv2
import mediapipe as mp
import numpy as np

# setup mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# math function to calculate angle between 3 points (e.g., shoulder, elbow, wrist)
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid (the vertex)
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# load your video (replace with a video of someone shooting)
cap = cv2.VideoCapture('lax_shot.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # convert color for mediapipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # if it finds a body, do the math
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # get coordinates for right arm
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # calculate it
        angle = calculate_angle(shoulder, elbow, wrist)
        
        # draw the angle on the video at the elbow
        cv2.putText(image, str(int(angle)), 
                    tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
        # draw the actual skeleton lines
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    cv2.imshow('Lax Form AI', image)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
