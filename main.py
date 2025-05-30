import cv2
import mediapipe as mp
import time
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

prev_x = None
cooldown = 0.3  
last_action_time = time.time()

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        gesture = None
        current_time = time.time()

        if results.multi_hand_landmarks:
            
            hand_landmarks = results.multi_hand_landmarks[0]
            

            lm = hand_landmarks.landmark
            tip_coords = [(int(p.x * w), int(p.y * h)) for p in [lm[4], lm[8], lm[12], lm[16], lm[20]]]
            hand_center = (int(lm[9].x * w), int(lm[9].y * h))

            tip_ids = [8, 12, 16, 20]
            base_ids = [5, 9, 13, 17]

            if current_time - last_action_time > cooldown:
                
                if all(lm[tip].y > lm[base].y for tip, base in zip(tip_ids, base_ids)):
                    pyautogui.press('up')
                    gesture = "Jump"
                    print("Gesture detected: Jump")
                    last_action_time = current_time

                
                elif (lm[8].y < lm[6].y and  
                      lm[12].y > lm[10].y and  
                      lm[16].y > lm[14].y and 
                      lm[20].y > lm[18].y):    
                    pyautogui.press('down')
                    gesture = "Scroll"
                    print("Gesture detected: Scroll")
                    last_action_time = current_time

                elif prev_x is not None and (hand_center[0] - prev_x) > 30:
                    pyautogui.press('right')
                    gesture = "Move Right"
                    print("Gesture detected: Move Right")
                    last_action_time = current_time

                
                elif prev_x is not None and (hand_center[0] - prev_x) < -30:
                    pyautogui.press('left')
                    gesture = "Move Left"
                    print("Gesture detected: Move Left")
                    last_action_time = current_time

            prev_x = hand_center[0]

        
        if gesture:
            cv2.putText(image, f'Gesture: {gesture}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Controller", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
