import cv2
import mediapipe as mp
import math



class PoseDetector:

    def __init__(self, mode=False, complexity=1, upperBody=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.modelComplexity = complexity
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComplexity, self.upperBody, self.smooth, self.detectionCon, self.trackingCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, plm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                center_x, center_y = int(plm.x * width), int(plm.y * height)
                self.lmList.append([id, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x,center_y), 5, (255,0,0), -1)
        return self.lmList


    def findAngle(self, img, p1, p2, p3, draw=True):
        x1,y1 = self.lmList[p1][1:]
        x2,y2 = self.lmList[p2][1:]
        x3,y3 = self.lmList[p3][1:]
        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))
        if angle < 0:
            angle += 360
        text = str(int(angle))
        #print(text)
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)
            cv2.line(img, (x3,y3), (x2,y2), (255,255,255), 3)
            cv2.circle(img, (x1,y1), 10, (0,0,255), -1)
            cv2.circle(img, (x1,y1), 15, (0,0,255), 2)
            cv2.circle(img, (x2,y2), 10, (0,0,255), -1)
            cv2.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3,y3), 10, (0,0,255), -1)
            cv2.circle(img, (x3,y3), 15, (0,0,255), 2)
            cv2.putText(img, text, (x2-50,y2+50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

        return angle
