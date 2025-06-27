import cv2
import pyautogui
import time
import util
from pynput.mouse import Button, Controller
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# Mouse control and screen info
mouse = Controller()
screen_width, screen_height = pyautogui.size()
prev_x, prev_y = 0, 0
smoothening = 7
last_action_time = 0
cooldown = 1  # in seconds

def get_pixel_landmarks(hand_landmarks, frame):
    h, w, _ = frame.shape
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

def move_mouse(index_finger_tip):
    global prev_x, prev_y
    if (
        index_finger_tip and
        isinstance(index_finger_tip, tuple) and
        len(index_finger_tip) == 2 and
        0 <= index_finger_tip[0] <= 1 and
        0 <= index_finger_tip[1] <= 1
    ):
        x = int(index_finger_tip[0] * screen_width)
        y = int(index_finger_tip[1] * screen_height / 2)
        x = prev_x + (x - prev_x) // smoothening
        y = prev_y + (y - prev_y) // smoothening
        pyautogui.moveTo(x, y)
        prev_x, prev_y = x, y

def is_left_click(landmarks, dist):
    return (
        util.get_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        util.get_angle(landmarks[9], landmarks[10], landmarks[12]) > 90 and
        dist > 50
    )

def is_right_click(landmarks, dist):
    return (
        util.get_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        util.get_angle(landmarks[5], landmarks[6], landmarks[8]) > 90 and
        dist > 50
    )

def is_double_click(landmarks, dist):
    return (
        util.get_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        util.get_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        dist > 50
    )

def is_screenshot(landmarks, dist):
    return (
        util.get_angle(landmarks[5], landmarks[6], landmarks[8]) < 50 and
        util.get_angle(landmarks[9], landmarks[10], landmarks[12]) < 50 and
        dist < 50
    )

def detect_gesture(frame, landmarks, index_finger_tip):
    global last_action_time

    if len(landmarks) < 12:
        return

    thumb_index_dist = util.get_distance([landmarks[4], landmarks[5]])
    if thumb_index_dist is None:
        return

    # Movement
    if (
        index_finger_tip and
        isinstance(index_finger_tip, tuple) and
        len(index_finger_tip) == 2 and
        0 <= index_finger_tip[0] <= 1 and
        0 <= index_finger_tip[1] <= 1
    ):
        if thumb_index_dist < 50 and util.get_angle(landmarks[5], landmarks[6], landmarks[8]) > 90:
            move_mouse(index_finger_tip)

    # Gesture actions
    if time.time() - last_action_time > cooldown:
        if is_left_click(landmarks, thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            last_action_time = time.time()

        elif is_right_click(landmarks, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            last_action_time = time.time()

        elif is_double_click(landmarks, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            last_action_time = time.time()

        elif is_screenshot(landmarks, thumb_index_dist):
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot = pyautogui.screenshot()
            screenshot.save(f'my_screenshot_{timestamp}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            last_action_time = time.time()

def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = get_pixel_landmarks(hand_landmarks, frame)
            index_finger_tip = hand_landmarks.landmark[8]
            detect_gesture(frame, landmarks, (index_finger_tip.x, index_finger_tip.y))

        cv2.imshow("Virtual Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
