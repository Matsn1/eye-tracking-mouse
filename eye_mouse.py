import cv2
import mediapipe as mp
import pyautogui
import time
import winsound
import numpy as np
from collections import deque

# -------------------- INITIAL SETUP --------------------

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
screen_w, screen_h = pyautogui.size()

# State variables
holding = False
last_right_blink = 0
left_eye_closed_time = None
right_eye_closed_time = None
mouth_open_time = None
mouth_triggered = False
hold_enabled = True  # disables hold when keyboard is open

# Calibration points
calibration_points = {
    "top_left": None,
    "top_right": None,
    "bottom_left": None,
    "bottom_right": None
}

# -------------------- HELPER FUNCTIONS --------------------

def get_pupil(landmarks):
    """Return (x, y) position of right iris center"""
    return landmarks[474].x, landmarks[474].y

# -------------------- CALIBRATION --------------------

for point in calibration_points:
    print(f"Look at {point.upper()} corner")
    winsound.Beep(1000, 500)
    collected = []
    start_time = time.time()

    while time.time() - start_time < 2:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks

        cv2.putText(frame, f"Look at {point}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

        if landmark_points:
            landmarks = landmark_points[0].landmark
            collected.append(get_pupil(landmarks))

    if collected:
        avg_x = sum(p[0] for p in collected) / len(collected)
        avg_y = sum(p[1] for p in collected) / len(collected)
        calibration_points[point] = (avg_x, avg_y)

print("Calibration complete ")
cv2.destroyWindow("Calibration")

# -------------------- MAIN LOOP --------------------

smooth_x, smooth_y = 0.0, 0.0
alpha = 0.05
pupil_buffer = deque(maxlen=5)

MOUTH_THRESHOLD = 0.04
MOUTH_HOLD_TIME = 0.2

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # ---------------- MOUSE CONTROL ----------------
        pupil_x, pupil_y = get_pupil(landmarks)

        min_x = min(calibration_points["top_left"][0], calibration_points["bottom_left"][0])
        max_x = max(calibration_points["top_right"][0], calibration_points["bottom_right"][0])
        min_y = min(calibration_points["top_left"][1], calibration_points["top_right"][1])
        max_y = max(calibration_points["bottom_left"][1], calibration_points["bottom_right"][1])

        range_x = max_x - min_x if (max_x - min_x) != 0 else 1e-6
        range_y = max_y - min_y if (max_y - min_y) != 0 else 1e-6

        norm_x = min(max((pupil_x - min_x) / range_x, 0.0), 1.0)
        norm_y = min(max((pupil_y - min_y) / range_y, 0.0), 1.0)

        pupil_buffer.append((norm_x, norm_y))
        avg_x = sum(p[0] for p in pupil_buffer) / len(pupil_buffer)
        avg_y = sum(p[1] for p in pupil_buffer) / len(pupil_buffer)

        smooth_x = alpha * avg_x + (1 - alpha) * smooth_x
        smooth_y = alpha * avg_y + (1 - alpha) * smooth_y

        screen_x = int(smooth_x * screen_w)
        screen_y = int(smooth_y * screen_h)
        pyautogui.moveTo(screen_x, screen_y)

        # Draw cursor dot
        frame_h, frame_w, _ = frame.shape
        cv2.circle(frame, (int(pupil_x * frame_w), int(pupil_y * frame_h)), 8, (0, 255, 255), -1)

        # ---------------- BLINK DETECTION ----------------
        left_eye = [landmarks[145], landmarks[159]]
        right_eye = [landmarks[374], landmarks[386]]

        left_dist = abs(left_eye[0].y - left_eye[1].y)
        right_dist = abs(right_eye[0].y - right_eye[1].y)

        left_closed = left_dist < 0.015
        right_closed = right_dist < 0.015
        both_closed = left_closed and right_closed

        if both_closed:
            if holding:
                pyautogui.mouseUp()
                holding = False
            left_eye_closed_time = None
            right_eye_closed_time = None

        else:
            # LEFT EYE ACTIONS
            if left_closed:
                if left_eye_closed_time is None:
                    left_eye_closed_time = time.time()
            else:
                if left_eye_closed_time is not None:
                    blink_duration = time.time() - left_eye_closed_time
                    if blink_duration < 0.4:
                        pyautogui.click()
                    elif 0.4 <= blink_duration < 1.2 and hold_enabled:
                        if not holding:
                            pyautogui.mouseDown()
                            holding = True
                        else:
                            pyautogui.mouseUp()
                            holding = False
                    elif blink_duration >= 1.5:
                        pyautogui.doubleClick()
                    left_eye_closed_time = None

            # RIGHT EYE ACTIONS
            if right_closed:
                if right_eye_closed_time is None:
                    right_eye_closed_time = time.time()
            else:
                if right_eye_closed_time is not None:
                    blink_duration = time.time() - right_eye_closed_time
                    if blink_duration < 0.6 and time.time() - last_right_blink > 0.6:
                        pyautogui.rightClick()
                        last_right_blink = time.time()
                    right_eye_closed_time = None

        # ---------------- MOUTH DETECTION ----------------
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        mouth_distance = np.hypot(
            upper_lip.x - lower_lip.x,
            upper_lip.y - lower_lip.y
        )

        if mouth_distance > MOUTH_THRESHOLD:
            if mouth_open_time is None:
                mouth_open_time = time.time()
            else:
                if (time.time() - mouth_open_time) > MOUTH_HOLD_TIME and not mouth_triggered:
                    try:
                        pyautogui.hotkey('winleft', 'ctrl', 'o')
                    except Exception:
                        pyautogui.hotkey('win', 'ctrl', 'o')

                    mouth_triggered = True
                    hold_enabled = False
                    if holding:
                        pyautogui.mouseUp()
                        holding = False

                    cv2.putText(
                        frame,
                        "Keyboard Opened (Hold Disabled)",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
        else:
            mouth_open_time = None
            mouth_triggered = False
            hold_enabled = True

        # ---------------- VISUAL ENHANCEMENTS ----------------
        # Draw face mesh landmarks
        for idx, lm in enumerate(landmarks):
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            if idx in [13, 14, 145, 159, 374, 386, 474, 475, 476, 477]:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Eye tracking mode banner
        cv2.putText(frame, "👁 Eye Tracking Mode ON", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # Debug info
        cv2.putText(frame, f"LeftDist:{left_dist:.4f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"RightDist:{right_dist:.4f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f"Mouth:{mouth_distance:.3f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # ---------------- DISPLAY ----------------
    cv2.imshow("Eye Controlled Mouse", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# -------------------- CLEANUP --------------------
cam.release()
cv2.destroyAllWindows()