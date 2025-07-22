import os
import cv2
import pandas as pd
from datetime import datetime
from deepface import DeepFace

# Paths
DATASET_PATH = r"C:\Users\manog\Downloads\Attendance_Emotion_System_DeepFace\dataset\train"
ATTENDANCE_FILE = "attendance_log.csv"

# Attendance time window (9:30 AM - 10:00 AM)
START_TIME = datetime.strptime("09:30:00", "%H:%M:%S").time()
END_TIME = datetime.strptime("10:00:00", "%H:%M:%S").time()

# Get all students from dataset folder
all_students = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]

print("Training face recognition model on dataset...")
DeepFace.find(
    img_path=os.path.join(DATASET_PATH,
                          all_students[0],
                          os.listdir(os.path.join(DATASET_PATH, all_students[0]))[0]),
    db_path=DATASET_PATH,
    enforce_detection=False
)
print("Face recognition model training completed!")

attendance = {}

cap = cv2.VideoCapture(0)
print("System is starting... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = datetime.now().time()

    if START_TIME <= current_time <= END_TIME:
        try:
            # Identify students in frame
            results = DeepFace.find(img_path=frame, db_path=DATASET_PATH, enforce_detection=False, silent=True)

            if not results.empty:
                identity = results.iloc[0]['identity']
                student_name = os.path.basename(os.path.dirname(identity))

                if student_name not in attendance:
                    attendance[student_name] = {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Status": "Present",
                        "Emotion": "Unknown"
                    }

                # Emotion detection
                emotion_results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(emotion_results, list):
                    dominant_emotion = emotion_results[0]['dominant_emotion']
                else:
                    dominant_emotion = emotion_results['dominant_emotion']

                attendance[student_name]["Emotion"] = dominant_emotion

                cv2.putText(frame, f"{student_name} - {dominant_emotion}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error analyzing frame: {e}")

    cv2.imshow("Attendance Emotion System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Mark absent students who were not detected
for student in all_students:
    if student not in attendance:
        attendance[student] = {
            "Time": "-",
            "Status": "Absent",
            "Emotion": "-"
        }

# Save to CSV
df = pd.DataFrame.from_dict(attendance, orient='index')
df.reset_index(inplace=True)
df.rename(columns={'index': 'Student'}, inplace=True)
df.to_csv(ATTENDANCE_FILE, index=False)
print(f"Attendance saved to {ATTENDANCE_FILE}")
