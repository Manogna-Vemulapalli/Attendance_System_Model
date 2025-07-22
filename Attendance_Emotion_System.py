import os
import cv2
import pandas as pd
from datetime import datetime, time
from deepface import DeepFace
from keras.models import load_model

# -------------------- Configuration --------------------
DATASET_PATH = r"C:\Users\manog\Downloads\Attendance_Emotion_System_DeepFace\dataset\train"
EMOTION_MODEL_PATH = r"C:\Users\manog\Downloads\Attendance_Emotion_System_DeepFace\converted_model.keras"
ATTENDANCE_FILE = "attendance_log.csv"

START_TIME = time(9, 30)
END_TIME = time(10, 0)

# -------------------- Load Emotion Model --------------------
if not os.path.exists(EMOTION_MODEL_PATH):
    raise FileNotFoundError(f"Emotion model not found at {EMOTION_MODEL_PATH}")
emotion_model = load_model(EMOTION_MODEL_PATH, compile=False)
print("Custom emotion model loaded successfully!")

# -------------------- Load Students & Cache Embeddings --------------------
students = [s for s in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, s))]
attendance = {s: {"status": "Absent", "emotion": "", "timestamp": ""} for s in students}

print("Caching student face embeddings...")
representations = DeepFace.find(
    img_path=os.path.join(DATASET_PATH, students[0], os.listdir(os.path.join(DATASET_PATH, students[0]))[0]),
    db_path=DATASET_PATH,
    model_name="VGG-Face",
    enforce_detection=False,
    silent=True
)
print("Face recognition model ready!")

# -------------------- Start Video Capture --------------------
cap = cv2.VideoCapture(0)
print("System is running... Press 'q' to exit.")

while True:
    now = datetime.now().time()

    if not (START_TIME <= now <= END_TIME):
        print("Outside attendance marking time (9:30 AM - 10:00 AM). Waiting...")
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
        continue

    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Identify faces
        results = DeepFace.find(
            img_path=frame,
            db_path=DATASET_PATH,
            model_name="VGG-Face",
            enforce_detection=False,
            silent=True
        )

        if not results.empty:
            for _, row in results.iterrows():
                student_name = os.path.basename(os.path.dirname(row['identity']))
                distance = row['VGG-Face_cosine']

                if distance < 0.3 and attendance[student_name]["status"] == "Absent":  # High-accuracy threshold
                    emotion_results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                    emotion = emotion_results[0]['dominant_emotion'] if isinstance(emotion_results, list) else emotion_results['dominant_emotion']

                    attendance[student_name] = {
                        "status": "Present",
                        "emotion": emotion,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    print(f"{student_name} marked Present at {attendance[student_name]['timestamp']} with emotion: {emotion}")
                    cv2.putText(frame, f"{student_name} - {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            print("No known students detected in this frame.")

    except Exception as e:
        print(f"Error processing frame: {e}")

    cv2.imshow("Attendance & Emotion System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------- Save Attendance Log --------------------
df = pd.DataFrame([
    {"Student Name": name,
     "Status": data["status"],
     "Emotion": data["emotion"],
     "Timestamp": data["timestamp"]}
    for name, data in attendance.items()
])

df.to_csv(ATTENDANCE_FILE, index=False)
print(f"Attendance saved to {ATTENDANCE_FILE}")


