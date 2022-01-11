import cv2
import mediapipe as mp
import time

class hand_tracking():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]


    def find_hands(self, image, draw = True):

        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(image_RGB)

    #check hand landmark in camera
    #print(result.multi_hand_landmarks)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                #drawing hand landmark points and connections
                    self.mp_draw.draw_landmarks(image, handLms, self.mp_hands.HAND_CONNECTIONS)
        return image

    def find_hand_positions(self, image, handNo = 0, draw = True):

        self.landmark_list = []

        if self.result.multi_hand_landmarks:
            # for one hand

            myHand = self.result.multi_hand_landmarks[handNo]

            # creating id for each landmark
            for id, landmark in enumerate(myHand.landmark):
                #convert from axis into pixel
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # print(id, cx, cy)
                self.landmark_list.append([id, cx, cy])

                # draw a point with circle on the landmark id
                if draw and id == 8:
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return self.landmark_list


    def fingersUp(self):  # checking which finger is open
        fingers = []  # storing final result
        # Thumb < sign only when  we use flip function to avoid mirror inversion else > sign
        if self.landmark_list[self.tipIds[0]][1] > self.landmark_list[self.tipIds[0] - 1][
            1]:  # checking x position of 4 is in right to x position of 3
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):  # checking tip point is below tippoint-2 (only in Y direction)
            if self.landmark_list[self.tipIds[id]][2] < self.landmark_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():

    #setting camera and resolutions

    camera = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0
    hand_detection = hand_tracking()

    #Initialised camera

    while True:
        success, image = camera.read()
        image = hand_detection.find_hands(image, draw=True)
        landmark_list = hand_detection.find_hand_positions(image, draw= True)
        if len(landmark_list) !=0:
            print(landmark_list[8]) # --> can be set into respective point on finger
                                   
        #printing FPS on the images view
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        #showing camera live images
        cv2.imshow("images", image)
        cv2.waitKey(1)


#can be use as a module later
if __name__ == "__main__":
    main()