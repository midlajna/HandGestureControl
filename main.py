
import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
from pycaw.pycaw import AudioUtilities

# -----------------------------
# 1. SYSTEM CONTROL SETUP
# -----------------------------

devices = AudioUtilities.GetSpeakers()
volume_control = devices.EndpointVolume

# -----------------------------
# 2. MEDIAPIPE SETUP
# -----------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# 3. WEBCAM SETUP
# -----------------------------

cap = cv2.VideoCapture(0)
width, height = 800, 600

# Distance range for pinch gesture
min_distance = 30
max_distance = 200

# Smoothing factor: smaller = smoother
smoothing = 0.2

# Start with estimated current levels
try:
    current_brightness = sbc.get_brightness()[0]
except Exception:
    current_brightness = 50

current_volume = int(
    volume_control.GetMasterVolumeLevelScalar() * 100
)

smooth_level = None

# Pause state and open-palm detection
paused = False
was_open_palm = False

# -----------------------------
# 4. DRAW PERCENTAGE BAR
# -----------------------------

def draw_bar(frame, x, y, label, value, color):
    bar_width = 300
    bar_height = 22

    value = max(0, min(100, int(value)))

    cv2.putText(
        frame, f"{label}: {value}%",
        (x, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + bar_width, y + bar_height),
        (100, 100, 100), -1
    )

    fill_width = int(bar_width * value / 100)

    cv2.rectangle(
        frame,
        (x, y),
        (x + fill_width, y + bar_height),
        color, -1
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + bar_width, y + bar_height),
        (230, 230, 230), 2
    )


# -----------------------------
# 5. MAIN LOOP
# -----------------------------

while True:
    success, frame = cap.read()

    if not success:
        print("Could not access webcam")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (width, height))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Header
    cv2.rectangle(frame, (0, 0), (width, 75), (35, 35, 35), -1)

    cv2.putText(
        frame, "HAND GESTURE CONTROLLER",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

    cv2.putText(
        frame, "Left: Brightness | Right: Volume",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (200, 200, 200), 1
    )

    # Divide control zones
    cv2.line(
        frame, (width // 2, 85),
        (width // 2, height - 100),
        (180, 180, 180), 2
    )

    cv2.putText(
        frame, "BRIGHTNESS",
        (100, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 220, 255), 2
    )

    cv2.putText(
        frame, "VOLUME",
        (width // 2 + 100, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 220, 255), 2
    )

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        mp_draw.draw_landmarks(
            frame, hand, mp_hands.HAND_CONNECTIONS
        )

        # Check whether four fingers are extended
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]

        extended_count = sum(
            lm[tip].y < lm[pip].y
            for tip, pip in zip(finger_tips, finger_pips)
        )

        open_palm = extended_count == 4

        # Toggle pause once per new open-palm gesture
        if open_palm and not was_open_palm:
            paused = not paused
            smooth_level = None

        was_open_palm = open_palm

        # Thumb and index fingertips
        thumb = (
            int(lm[4].x * width),
            int(lm[4].y * height)
        )

        index = (
            int(lm[8].x * width),
            int(lm[8].y * height)
        )

        center_x = int(lm[9].x * width)

        distance = np.hypot(
            index[0] - thumb[0],
            index[1] - thumb[1]
        )

        # Convert distance to a target percentage
        target_level = int(np.interp(
            distance,
            [min_distance, max_distance],
            [0, 100]
        ))

        target_level = max(0, min(100, target_level))

        # Draw fingertip connection
        cv2.circle(frame, thumb, 10, (255, 0, 255), -1)
        cv2.circle(frame, index, 10, (255, 0, 255), -1)
        cv2.line(frame, thumb, index, (0, 255, 0), 3)

        if not paused and not open_palm:
            # Smooth out abrupt changes
            if smooth_level is None:
                smooth_level = float(target_level)
            else:
                smooth_level = (
                    smoothing * target_level
                    + (1 - smoothing) * smooth_level
                )

            level = int(round(smooth_level))

            if center_x < width // 2:
                # Brightness mode
                if abs(level - current_brightness) >= 2:
                    try:
                        sbc.set_brightness(level)
                        current_brightness = level
                    except Exception as e:
                        print("Brightness error:", e)

            else:
                # Volume mode
                if abs(level - current_volume) >= 2:
                    try:
                        volume_control.SetMasterVolumeLevelScalar(
                            level / 100.0, None
                        )
                        current_volume = level
                    except Exception as e:
                        print("Volume error:", e)

    else:
        was_open_palm = False
        smooth_level = None

    # -----------------------------
    # 6. DISPLAY BARS AND STATUS
    # -----------------------------

    draw_bar(
        frame, 50, height - 75,
        "Brightness", current_brightness,
        (0, 190, 255)
    )

    draw_bar(
        frame, width // 2 + 50, height - 75,
        "Volume", current_volume,
        (60, 220, 100)
    )

    if paused:
        status = "PAUSED - Open palm to resume"
        status_color = (0, 100, 255)
    else:
        status = "ACTIVE - Show open palm to pause"
        status_color = (0, 220, 100)

    cv2.putText(
        frame, status,
        (width // 2 - 220, height - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, status_color, 2
    )

    cv2.putText(
        frame, "Press Q to quit | Press SPACE to pause/resume",
        (20, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (230, 230, 230), 1
    )

    cv2.imshow("Hand Gesture Controller", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord(" "):
        paused = not paused
        smooth_level = None

# -----------------------------
# 7. CLEANUP
# -----------------------------

cap.release()
hands.close()
cv2.destroyAllWindows()
