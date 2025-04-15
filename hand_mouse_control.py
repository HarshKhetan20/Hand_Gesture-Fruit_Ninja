import cv2
import mediapipe as mp
import pyautogui
import time
import math
from collections import deque

# Init MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Capture webcam
cap = cv2.VideoCapture(0)

# Variables
click_cooldown = 0.5
last_click_time = 0
open_palm_start_time = None
prev_x, prev_y = 0, 0
prev_time = time.time()

# Store recent finger points
trail = deque(maxlen=10)  # Only keep last 10 points

# Function to get which fingers are up
def get_finger_states(hand_landmarks):
    fingers = []
    tips_ids = [4, 8, 12, 16, 20]

    # Thumb
    fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)

    # Other fingers
    for i in range(1, 5):
        tip = hand_landmarks.landmark[tips_ids[i]]
        pip = hand_landmarks.landmark[tips_ids[i] - 2]
        fingers.append(tip.y < pip.y)

    return fingers

# Function to get movement speed
def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark[8]

        x = int(lm.x * w)
        y = int(lm.y * h)
        screen_x = int(lm.x * screen_w)
        screen_y = int(lm.y * screen_h)

        # Update swipe trail
        trail.append((screen_x, screen_y))

        # Move mouse to index finger
        pyautogui.moveTo(screen_x, screen_y)

        # Debug dot
        cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

        # Speed calculation
        curr_time = time.time()
        time_diff = curr_time - prev_time
        dist = get_distance(x, y, prev_x, prev_y)
        speed = dist / time_diff if time_diff > 0 else 0
        prev_x, prev_y = x, y
        prev_time = curr_time

        # Get fingers
        fingers = get_finger_states(hand_landmarks)

        print("Fingers:", fingers, "| Speed:", int(speed), "| Trail:", len(trail))

        # TAP - index finger only
        if fingers == [False, True, False, False, False] and (curr_time - last_click_time > click_cooldown):
            pyautogui.click()
            print("👆 TAP")
            last_click_time = curr_time

        # SLICE - V-sign + fast motion
        elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            if speed > 1000 and (curr_time - last_click_time > click_cooldown):
                print("💥 SLICE triggered")
                pyautogui.mouseDown()
                for point in trail:
                    pyautogui.moveTo(point[0], point[1], duration=0.01)
                pyautogui.mouseUp()
                last_click_time = curr_time

        # STOP - open palm
        elif all(fingers[1:]):
            if open_palm_start_time is None:
                open_palm_start_time = time.time()
            elif time.time() - open_palm_start_time > 2:
                print("🛑 Emergency Stop")
                break
        else:
            open_palm_start_time = None

        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Fruit Ninja - Hybrid Swipe", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
