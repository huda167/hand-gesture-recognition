import cv2
import mediapipe as mp
import numpy as np
import pickle

# تحميل النموذج
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

# تحميل خريطة الإشارات
with open('label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# تحضير mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# تشغيل الكاميرا
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "???"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # تحويل الإحداثيات إلى مصفوفة
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # توقع التصنيف
            prediction = model.predict([landmarks])[0]
            gesture = label_map[prediction]

    # عرض اسم الإشارة
    cv2.putText(image, f'Gesture: {gesture}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Gesture Detection', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()