import cv2
import base64
import requests
from datetime import datetime
import time
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

state = ['Normal', 'Fall down']

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# 開啟鏡頭
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('./OldManFalls/higonnio.mp4')


def keypoints_to_array(results):
    keypoints_array = np.empty((0,))
    for landmark in results.pose_landmarks.landmark:
        keypoints_array = np.append(keypoints_array,
                                    [landmark.x, landmark.y, landmark.z, landmark.visibility])
    return keypoints_array

# 定義模型


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(132, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 輸出維度2，分別代表0和1的機率

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 建立模型實例
model = SimpleModel()

# 載入模型權重
model.load_state_dict(torch.load('./OldManFalls/model.pth'))

# 設置模型為評估模式
model.eval()

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
                    'http://192.168.168.38:5000/post_camera_frame', json=data)
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
            # 將資料轉換為 PyTorch 的 Tensor
            data_tensor = torch.tensor(keypoints, dtype=torch.float32)
            # 改變資料形狀以符合模型輸入的期望形狀 (1, 132)
            data_tensor = data_tensor.reshape(1, -1)
            # 使用模型進行預測
            with torch.no_grad():
                output = model(data_tensor)
                _, predicted = torch.max(output, 1)
            # 添加辨識結果
            cv2.putText(frame, state[predicted.item()], bottom_middle,
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
                'http://192.168.168.38:5000/post_camera_frame', json=data)
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
