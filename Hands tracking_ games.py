import handtracking as htm
import cv2
import mediapipe as mp
import time
import os
import numpy as np

folder_path = "Header"
myList = os.listdir(folder_path)

# ____________________________________________________________________________________
brush_thickness = 15
drawColor = (63, 255, 179)
#-------------------------------------------------------------------------------------

print(myList)
overlay_list = []
for import_path in myList:
    image = cv2.imread(f'{folder_path}/{import_path}')
    overlay_list.append(image)
print(len(overlay_list))
header = overlay_list[0]


camera = cv2.VideoCapture(0)
hand_detector = htm.hand_tracking(detectionCon=0.85, maxHands=1)

image_canvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0

while True:

    # 1.import images
    success, img = camera.read()
    img = cv2.flip(img, 1)

    # 2. find hand landmark
    img = hand_detector.find_hands(img)
    landmark_list = hand_detector.find_hand_positions(img, draw=False)

    if len(landmark_list) != 0:
        # print(landmark_list)

        # tip of index and middle finger
        x1, y1 = landmark_list[8][1:]
        x2, y2 = landmark_list[12][1:]

        # 3. check which finger is used
        fingers = hand_detector.fingersUp()
        # print(fingers)

        # 4. if selection mode is chosen
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)
            print("selection mode")

            # selecting color option
            if y1 < 125:
                if 150 < x1 < 250:
                    header = overlay_list[0]
                    drawColor = (0, 0, 255)
                elif 400 < x1 < 500:
                    header = overlay_list[1]
                    drawColor = (255, 0, 0)
                elif 600 < x1 < 700:
                    header = overlay_list[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlay_list[3]
                    drawColor = (0, 0, 0)

        # 5. if drawing mode is chosen
        if fingers[1] and fingers[2] == False:

            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                #eraser brush thickness
                brush_thickness = 50
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brush_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), drawColor, brush_thickness)
            else:
                brush_thickness = 15
                cv2.line(img,(xp, yp),(x1, y1),drawColor,brush_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), drawColor, brush_thickness)

            xp, yp = x1, y1


    # slice image into matrix for header
    img[0:125, 0:1280] = header

    #showing drawing canvas and camera
    img = cv2.addWeighted(img,0.5, image_canvas,0.5,0)
    cv2.imshow("camera", img)
    cv2.imshow("drawing canvas", image_canvas)

    cv2.waitKey(1)
