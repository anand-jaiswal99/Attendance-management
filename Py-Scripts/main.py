import cv2
import numpy as np
import os
import pandas as pd
import time
import json
from deepface import DeepFace
from mongo import store_db
import sys
import beepy
import urllib.request

if sys.argv[3] == "empty":
    video_capture = cv2.VideoCapture(0)
else:
    url = 'http://' + sys.argv[3] + '/shot.jpg'

# Load stored embeddings
face_db = {}
if os.path.exists("face_embeddings.json"):
    with open("face_embeddings.json", "r") as f:
        face_db = json.load(f)

attendance_record = set()
name_col, roll_no_col, time_col, classid_col = [], [], [], []

df = pd.read_excel("students/students_db.xlsx")
image_directory = "../public/assets/uploads/"
roll_record = {}

for _, row in df.iterrows():
    roll_no = str(row["roll_no"])
    name = row["name"]
    image_filename = row["image"]
    classid = row["classid"]
    roll_record[roll_no] = name
    
while True:
    try:
        if sys.argv[3] == "empty":
            ret, frame = video_capture.read()
        else:
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgNp, cv2.IMREAD_COLOR)
        
        frame = cv2.flip(frame, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        detected_faces = faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            face_roi = frame[y:y + h, x:x + w]
            face_embedding = DeepFace.represent(face_roi, model_name="Facenet")
            if not face_embedding:
                continue
            face_vector = np.array(face_embedding[0]["embedding"])
            
            name = "Unknown"
            best_match_roll_no = None
            min_distance = float("inf")
            
            for roll_no, data in face_db.items():
                stored_embedding = np.array(data["embedding"])
                distance = np.linalg.norm(stored_embedding - face_vector)
                if distance < 8 and distance < min_distance:
                    best_match_roll_no = roll_no
                    name = data["name"]
                    min_distance = distance
            
            if best_match_roll_no and best_match_roll_no not in attendance_record:
                attendance_record.add(best_match_roll_no)
                beepy.beep(sound=1)
                print(name, best_match_roll_no)
                name_col.append(name)
                roll_no_col.append(best_match_roll_no)
                curr_time = time.strftime("%H:%M:%S", time.localtime())
                time_col.append(curr_time)
                classid_col.append(classid)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.imshow("Video", frame)
    
    except Exception as e:
        print("Error processing frame:", e)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Save attendance data in the same format as the previous script
classid_col.append(classid)
data = {"Name": name_col, "RollNo": roll_no_col, "Time": time_col, "Class": classid_col}
print(data)

curr_time = time.strftime("%c", time.localtime())
file_name = "".join([s if not (s == " " and curr_time[i + 1] == " ") else "" for i, s in enumerate(curr_time)])
log_file_name = file_name

store_db(log_file_name, sys.argv[1], data)

if sys.argv[3] == "empty":
    video_capture.release()
cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os
# import pandas as pd
# import time
# import json
# from deepface import DeepFace
# from mongo import store_db
# import sys
# import beepy
# import urllib.request

# # Ensure attendance storage folder exists
# os.makedirs("attendance_logs", exist_ok=True)

# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Define CSV file name (unique for each run)
# csv_filename = f"attendance_logs/attendance_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

# def save_attendance_to_csv(data):
#     """Save attendance data to a CSV file."""
#     df = pd.DataFrame(data)
#     df.to_csv(csv_filename, index=False)
#     print(f"✅ Attendance saved to {csv_filename}")

# def get_face_embedding(image):
#     """Extracts face embedding using FaceNet from DeepFace."""
#     try:
#         embedding = DeepFace.represent(image, model_name="Facenet")
#         return np.array(embedding[0]["embedding"]) if embedding else None
#     except Exception as e:
#         print(f"Error extracting embedding: {e}")
#         return None

# def load_embeddings():
#     """Loads stored embeddings from a JSON file."""
#     if os.path.exists("face_embeddings.json"):
#         with open("face_embeddings.json", "r") as f:
#             return json.load(f)
#     return {}

# def save_embeddings(embeddings):
#     """Saves embeddings to a JSON file."""
#     with open("face_embeddings.json", "w") as f:
#         json.dump(embeddings, f, indent=4)

# # Determine video source
# if sys.argv[3] == "empty":
#     print("Video capturing from webcam")
#     video_capture = cv2.VideoCapture(0)
# else:
#     url = 'http://' + sys.argv[3] + '/shot.jpg'

# # Load stored embeddings
# face_db = load_embeddings()




# attendance_record = set()
# name_col, roll_no_col, time_col, classid_col = [], [], [], []

# df = pd.read_excel("students/students_db.xlsx")
# image_directory = "../public/assets/uploads/"

# for _, row in df.iterrows():
#     roll_no = str(row["roll_no"])
#     name = row["name"]
#     image_filename = row["image"]
#     classid = row["classid"]

#     image_path = os.path.join(image_directory, image_filename)
#     print("Checking:", image_path, os.path.exists(image_path))

#     if os.path.exists(image_path):
#         try:
#             student_image = cv2.imread(image_path)
#             if student_image is None:
#                 print(f"Error: Unable to load image {image_path}")
#                 continue
            
#             embedding = get_face_embedding(student_image)
#             if embedding is not None:
#                 face_db[roll_no] = {"name": name, "embedding": embedding.tolist()}
#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")

# # Save updated embeddings
# save_embeddings(face_db)


# while True:
#     try:
#         if sys.argv[3] == "empty":
#             ret, frame = video_capture.read()
#         else:
#             imgResp = urllib.request.urlopen(url)
#             imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
#             frame = cv2.imdecode(imgNp, cv2.IMREAD_COLOR)
#             if frame is None:
#                 print("Error: Frame capture failed")
#                 continue

#         frame = cv2.flip(frame, 2)  
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         print(f"Detected {len(faces)} faces in frame")

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#             face_roi = frame[y:y + h, x:x + w]
#             face_embedding = get_face_embedding(face_roi)  
            
            

#             if face_embedding is not None:
#                 name = "Unknown"
#                 best_match_roll_no = None
#                 min_distance = float("inf")

#                 for roll_no, data in face_db.items():
#                     stored_embedding = np.array(data["embedding"])
#                     distance = np.linalg.norm(stored_embedding - face_embedding)
#                     print("distance ",distance)

#                     if distance < 9 and distance < min_distance:  
#                         best_match_roll_no = roll_no
#                         name = data["name"]
#                         min_distance = distance  

#                 if best_match_roll_no and best_match_roll_no not in attendance_record:
#                     attendance_record.add(best_match_roll_no)
#                     beepy.beep(sound=1)
#                     print(name, best_match_roll_no)

#                     name_col.append(name)
#                     roll_no_col.append(best_match_roll_no)
#                     time_col.append(time.strftime("%H:%M:%S", time.localtime()))

#                 cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#         cv2.imshow("video_live", frame)

#     except Exception as e:
#         print("Error processing frame:", e)

#     if cv2.waitKey(10) == ord("a"):
#         break

# # Save attendance
# classid_col.append(classid)
# attendance_data = {"Name": name_col, "RollNo": roll_no_col, "Time": time_col, "Class": classid_col}
# save_attendance_to_csv(attendance_data)

# log_file_name = time.strftime("%Y-%m-%d_%H-%M-%S")  
# store_db(log_file_name, sys.argv[1], attendance_data)

# if sys.argv[3] == "empty":
#     video_capture.release()
# cv2.destroyAllWindows()

