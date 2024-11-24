import numpy as np
import os
import cv2
import face_recognition
from datetime import datetime
import pandas as pd

data_directory = r'C:\Users\manju\Downloads\PycharmProjects\pythonProject\imagebasic'
face_images = []
subject_names = []

image_files = os.listdir(data_directory)
print("Image files:", image_files)

for img_file in image_files:
    img_path = os.path.join(data_directory, img_file)
    img = cv2.imread(img_path)
    face_images.append(img)
    subject_names.append(os.path.splitext(img_file)[0])
print("Subject names:", subject_names)
def encode_faces(images):
    encoded_list = []
    for face_img in images:
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encode = face_recognition.face_encodings(rgb_img)
        if len(face_encode) > 0:
            encoded_list.append(face_encode[0])
    return encoded_list


def mark_attendance(name):
    today_date = datetime.now().strftime('%Y-%m-%d')
    attendance_file_path = r'C:\Users\manju\Downloads\PycharmProjects\pythonProject\attendance.xlsx'

    if os.path.exists(attendance_file_path):
        df = pd.read_excel(attendance_file_path, index_col=0)
    else:
        df = pd.DataFrame(columns=['Date'])

    if name not in df.columns:
        df[name] = ''

    if today_date not in df.index:
        df.loc[today_date] = ''

    df.at[today_date,name] = datetime.now().strftime('%H:%M:%S')
    df.to_excel(attendance_file_path)


known_encodings = encode_faces(face_images)
print('Encoding complete')

attendance_file_path = r'C:\Users\manju\Downloads\PycharmProjects\pythonProject\attendance.xlsx'
if not os.path.exists(attendance_file_path):
    initial_df = pd.DataFrame(columns=['Date'])
    initial_df.to_excel(attendance_file_path, index_label='Date')

cap = cv2.VideoCapture(0)
while True:
    success, webcam_img = cap.read()

    if not success:
        print("Failed to capture webcam image.")
        break

    resized_img = cv2.resize(webcam_img, (0, 0), None, 0.25, 0.25)
    rgb_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_resized_img)
    face_encodings = face_recognition.face_encodings(rgb_resized_img, face_locations)

    for face_encode, face_loc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encode)
        face_distances = face_recognition.face_distance(known_encodings, face_encode)
        print(face_distances)
        match_index = np.argmin(face_distances)
        
        if matches[match_index]:
            identified_name = subject_names[match_index].upper()
            print("Identified:", identified_name)
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(webcam_img, (x1, y1), (x2, y2), (0, 225, 0), 2)
            cv2.rectangle(webcam_img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(webcam_img, identified_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(identified_name)

    cv2.imshow('Webcam', webcam_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
