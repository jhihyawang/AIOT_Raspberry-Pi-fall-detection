import cv2
import base64
import requests
from datetime import datetime
import time
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
import joblib
import pandas as pd

state = ['Normal', 'Fall down']

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 開啟鏡頭
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./OldManFalls/funny.mp4')


def keypoints_to_array(results):
    keypoints_array = np.empty((0,))
    for landmark in results.pose_landmarks.landmark:
        keypoints_array = np.append(keypoints_array,
                                    [landmark.x, landmark.y, landmark.z, landmark.visibility])
    return keypoints_array


# 加載模型
model = joblib.load('./OldManFalls/svm_model.pkl')

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while True:
        # 照一張
        ret, frame = cap.read()
        if not ret:  # 如果沒連接攝影機
            print("Error: Failed to capture image")
            # 要POST的資料
            data = {'image': "NC", 'timestamp': "0"}

            # 發送POST請求
            try:
                response = requests.post(
                    'http://172.20.10.4:5000/post_camera_frame', json=data)
                if response.status_code == 200:
                    print("err sent successfully")
                else:
                    print(
                        f"Failed to send err. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print("Failed to send err:", e)

            # 等待
            time.sleep(0.5)
            break
        # 添加文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner = (10, frame.shape[0] - 10)
        bottom_middle = (10, int(frame.shape[0]/2))
        font_scale = 1
        font_color = (255, 255, 255)  # 白色
        line_type = 2

        results = holistic.process(frame)
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
        if results.pose_landmarks != None:
            keypoints = keypoints_to_array(results)

            # 將資料轉換成DataFrame
            new_data_df = pd.DataFrame(keypoints)
            keypoints = keypoints.reshape(1, 132)
            # 進行預測
            prediction = model.predict(keypoints)
            # 添加辨識結果
            cv2.putText(frame, state[prediction[0]], bottom_middle,
                        font, font_scale, (255, 255, 0), line_type, cv2.LINE_AA)

        # 目前時間
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 添加目前時間
        cv2.putText(frame, current_time, bottom_left_corner,
                    font, font_scale, font_color, line_type, cv2.LINE_AA)

        # 轉為Base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer)

        # 要POST的資料
        data = {'image': img_base64.decode('utf-8'), 'timestamp': current_time}

        # 發送POST請求
        try:
            response = requests.post(
                'http://172.20.10.4:5000/post_camera_frame', json=data)
            if response.status_code == 200:
                print("Image sent successfully")
            else:
                print(
                    f"Failed to send image. Status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print("Failed to send image:", e)

        # 等待
        time.sleep(0.2)

#
cap.release()
cv2.destroyAllWindows()
