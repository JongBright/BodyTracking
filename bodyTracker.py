import cv2
import mediapipe as mp




mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, plm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            center_x, center_y = int(plm.x * width), int(plm.y * height)
            cv2.circle(img, (center_x,center_y), 5, (255,0,0), -1)


    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
