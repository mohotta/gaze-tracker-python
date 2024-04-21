
import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import pickle

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# commented parts are for dataset generation ( need to commnet out model loading and predicting part )

# i = 0
# csv_file = open("data.csv", 'a')
# label = 'no'

for _ in range(100000000):
    pass

while True: 

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[469:473] + landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))

        for id, landmark in enumerate([landmarks[33], landmarks[159], landmarks[133], landmarks[145], landmarks[362], landmarks[386], landmarks[263], landmarks[374]]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        for id, landmark in enumerate([landmarks[4], landmarks[152], landmarks[10], landmarks[366], landmarks[137]]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 0))

    dist_z = np.abs(landmarks[4].z)

    # eye1 - measures
    eye_1_w = np.abs(landmarks[33].x - landmarks[133].x)
    eye_1_h = np.abs(landmarks[159].y - landmarks[145].y)
    eye_1_x_left = np.abs(landmarks[33].x - landmarks[471].x)/(eye_1_w*dist_z)
    eye_1_x_right = np.abs(landmarks[133].x - landmarks[469].x)/(eye_1_w*dist_z)
    eye_1_y_top = np.abs(landmarks[159].y - landmarks[470].y)/(eye_1_h*dist_z)
    eye_1_y_bottom = np.abs(landmarks[145].y - landmarks[472].y)/(eye_1_h*dist_z)

    print("Eye 1: ", eye_1_x_left, eye_1_x_right, eye_1_y_top, eye_1_y_bottom)

    # eye2 - measures
    eye_2_w = np.abs(landmarks[362].x - landmarks[263].x)
    eye_2_h = np.abs(landmarks[386].y - landmarks[374].y)
    eye_2_x_left = np.abs(landmarks[362].x - landmarks[476].x)/(eye_2_w*dist_z)
    eye_2_x_right = np.abs(landmarks[263].x - landmarks[474].x)/(eye_2_w*dist_z)
    eye_2_y_top = np.abs(landmarks[386].y - landmarks[475].y)/(eye_2_h*dist_z)
    eye_2_y_bottom = np.abs(landmarks[374].y - landmarks[477].y)/(eye_2_h*dist_z)

    print("Eye 2: ", eye_2_x_left, eye_2_x_right, eye_2_y_top, eye_2_y_bottom)

    # face
    face_left_width = np.abs(landmarks[4].x - landmarks[137].x)/dist_z
    face_right_width = np.abs(landmarks[4].x - landmarks[366].x)/dist_z
    face_top_width = np.abs(landmarks[4].y - landmarks[10].y)/dist_z
    face_bottom_width = np.abs(landmarks[4].y - landmarks[152].y)/dist_z

    # print("face", face_left_width, face_right_width, face_top_width, face_bottom_width)

    # # write to a file
    # csv_file.write(f"{eye_1_x_left}, {eye_1_x_right}, {eye_1_y_top}, {eye_1_y_bottom}, {eye_2_x_left}, {eye_2_x_right}, {eye_2_y_top}, {eye_2_y_bottom}, {face_left_width}, {face_right_width}, {face_top_width}, {face_bottom_width}, {label}\n")

    # i += 1
    # if i >= 1000:
    #     csv_file.close()
    #     playsound("sound.wav")
    #     break

    model = pickle.load(open('model.pkl', 'rb'))
    pred = model.predict(np.array([eye_1_x_left, eye_1_x_right, eye_1_y_top, eye_1_y_bottom, eye_2_x_left, eye_2_x_right, eye_2_y_top, eye_2_y_bottom, face_left_width, face_right_width, face_top_width, face_bottom_width]).reshape(1, -1))

    print(pred)

    cv2.imshow("opencv tracker", frame)
    cv2.waitKey(1)

