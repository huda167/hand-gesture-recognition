import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# مجلد البيانات
data_dir = "hand_gesture_dataset"

# تحميل البيانات والتصنيفات
X = []
y = []
label_map = {}  # عشان نحول الأسماء إلى أرقام
label_num = 0

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    label_map[label_num] = folder
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            data = np.load(file_path)
            X.append(data)
            y.append(label_num)
    label_num += 1

# تحويل إلى numpy arrays
X = np.array(X)
y = np.array(y)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# تدريب النموذج
model = RandomForestClassifier()
model.fit(X_train, y_train)

# دقة الاختبار
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# حفظ النموذج و الخرائط
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)