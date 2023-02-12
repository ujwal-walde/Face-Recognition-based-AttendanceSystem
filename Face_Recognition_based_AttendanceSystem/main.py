# Face Recognition based attendance system

import face_recognition as fr 
import cv2
import numpy as np 
import os 
import csv 
from datetime import datetime

video_capture = cv2.VideoCapture(0)

ms_img = fr.load_image_file("C:\\Users\\rohit\\Desktop\\Python\\TXON Intern\\AttendanceSystem\\data\\ms_dhoni.jpg")

ms_encoding = fr.face_encodings(ms_img)[0]

uv_img = fr.load_image_file("C:\\Users\\rohit\\Desktop\\Python\\TXON Intern\\AttendanceSystem\\data\\uv.jpg")

uv_encoding = fr.face_encodings(uv_img)[0]

bhuvi_img = fr.load_image_file("C:\\Users\\rohit\\Desktop\\Python\\TXON Intern\\AttendanceSystem\\data\\bhuvi.jpg")

bhuvi_encoding = fr.face_encodings(bhuvi_img)[0]

kd_img = fr.load_image_file("C:\\Users\\rohit\\Desktop\\Python\\TXON Intern\\AttendanceSystem\\data\\Kuldeep-yadav.jpg")

kd_encoding = fr.face_encodings(kd_img)[0]



known_face_encodings = [
    ms_encoding,
    uv_encoding,
    bhuvi_encoding,
    kd_encoding
]

known_face_names = [
    "MS Dhoni",
    "Yuvraj Singh",
    "Bhuvneshwar Kumar",
    "Kuldeep Yadav"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = fr.face_locations(rgb_small_frame)
        face_encodings = fr.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = fr.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = fr.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

