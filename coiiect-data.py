import cv2
import os
import mediapipe as mp
import numpy as np

# غيري هذا الاسم في كل مرة حسب الإشارة
gesture_name = "thumbs-up"  # 👈 غيّريه إلى: open_hand / fist / thumbs_up / peace ...

# المسار لحفظ البيانات
save_dir = f"hand_gesture_dataset/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)

# إعداد mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # نحول الـ landmarks إلى مصفوفة
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # نحفظ البيانات كـ ملف .npy
            np.save(os.path.join(save_dir, f'{counter}.npy'), np.array(landmarks))
            counter += 1

    cv2.putText(image, f'Data collected: {counter}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Collecting Data', image)

    if cv2.waitKey(1) & 0xFF == ord('q') or counter > 200:
        break

cap.release()
cv2.destroyAllWindows()