import cv2
import mediapipe as mp
import pyautogui
import numpy as np

x1 = y1 = x2 = y2 = 0
mute_counter = 0
mute_threshold = 30
mute_hold_duration = 30

webcam = cv2.VideoCapture(0)

my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger tip
                    cv2.circle(img=image, center=(x, y), radius=8, color=(220, 220, 220), thickness=2)
                    x1 = x
                    y1 = y

                if id == 4:  # Thumb tip
                    cv2.circle(img=image, center=(x, y), radius=8, color=(220, 220, 220), thickness=2)
                    x2 = x
                    y2 = y

        dist = np.hypot(x2 - x1, y2 - y1)
        cv2.line(image, (x1, y1), (x2, y2), (50, 50, 50), 5)

        if dist < mute_threshold:
            mute_counter += 1
            if mute_counter >= mute_hold_duration:
                pyautogui.press("volumemute")
                mute_counter = 0
        if dist > 50:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")

    cv2.imshow("Hand Volume Control using Python", image)

    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
