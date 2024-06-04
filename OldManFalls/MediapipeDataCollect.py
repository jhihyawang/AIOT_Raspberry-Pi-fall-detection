import csv
import time
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./data/fall.mp4')

# 將座標轉換為陣列


def keypoints_to_array(results):
    keypoints_array = np.empty((0,))
    for landmark in results.pose_landmarks.landmark:
        keypoints_array = np.append(keypoints_array,
                                    [landmark.x, landmark.y, landmark.z, landmark.visibility])
    return keypoints_array


with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    time.sleep(5)  # 給你5秒擺好POSE
    # 設定CSV名稱
    csv_filename = "./OldManFalls/dataSet/1.csv"
    # 將新的關鍵點座標陣列附加到CSV文件末尾
    with open(csv_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        start_time = time.time()
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.resize(img, (640, 480))
            img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img2)
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            cv2.imshow('warmup', img)
            if time.time() - start_time >= 2:
                if results.pose_landmarks != None:
                    # 姿勢識別結果的關鍵點座標陣列
                    keypoints = keypoints_to_array(results)
                    # 關鍵點座標陣列包含了每個關鍵點的x、y、z座標、visibility
                    # 寫入每個新的關鍵點
                    csv_writer.writerow(keypoints.flatten())
                    print(keypoints)
                    #winsound.Beep(2000, 20)
                    start_time = time.time()
            if cv2.waitKey(5) == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
