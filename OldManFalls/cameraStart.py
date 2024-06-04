import cv2
import base64
import requests
from datetime import datetime
import time

# 開啟鏡頭
cap = cv2.VideoCapture(0)

while True:
    # 照一張
    ret, frame = cap.read()

    if not ret:  # 如果沒連接攝影機
        print("Error: Failed to capture image")
        # 要POST的資料
        data = {'image': "NC", 'timestamp': current_time}

        # 發送POST請求
        try:
            response = requests.post(
                'http://10.40.60.82:5000/post_camera_frame', json=data)
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

    # 目前時間
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 添加目前時間
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, frame.shape[0] - 10)
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    line_type = 2
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
            'http://10.40.60.82:5000/post_camera_frame', json=data)
        if response.status_code == 200:
            print("Image sent successfully")
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print("Failed to send image:", e)

    # 等待
    time.sleep(0.3)

#
cap.release()
cv2.destroyAllWindows()
