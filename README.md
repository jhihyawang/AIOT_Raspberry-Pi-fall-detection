# HW7 Raspberry Pi  CV 跌倒偵測

### 資管三 4110029009 王致雅

實驗硬體：

macOS M1 版本 14.4.1

Raspberry Pi 4 Model B 64bit

## step1 兩台電腦的環境安裝

建立虛擬環境 python -m venv pi

啟動虛擬環境 source pi/bin/activate

安裝相關套件 pip install -r requirements.txt  

- **＊mac 報錯**
    
    ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.10 (from versions: 0.10.13, 0.10.14)
    ERROR: No matching distribution found for mediapipe==0.10.10
    
    **需修改 mediapipe==0.10.14**
    

rasberry pi

![20240521_15h30m30s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/185d0a69-2edb-4eda-80b3-ba1ce642e031.png)

macOS

![截圖 2024-05-22 上午9.29.57.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258A%25E5%258D%25889.29.57.png)

![20240521_15h42m16s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/6bfc6e7f-31a1-4d78-bc78-0410389fc24b.png)

## step2 執行 [app.py](http://app.py)

- python app.py
    
    會得到兩個ip
    
    對內 ip : [http://127.0.0.1:5000](http://127.0.0.1:5000/)
    
    對外 ip：[http://172.20.10.5:5000](http://172.20.10.5:5000/)
    
    ![20240521_15h55m20s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/357e8171-aef1-4b73-95dc-dbf2f1a061ff.png)
    

- 樹莓派 打開瀏覽器
    
    輸入對內ip：[http://127.0.0.1:5000](http://127.0.0.1:5000/)
    
    ![20240522_18h39m53s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/20240522_18h39m53s_grim.png)
    

- mac 打開瀏覽器
    
    輸入對外ip：[http://172.20.10.5:5000](http://172.20.10.5:5000/)
    
    可看到樹莓派伺服器的網頁
    
    ![截圖 2024-05-22 下午6.06.49.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258B%25E5%258D%25886.06.49.png)
    

## step3 修改並執行 [cameraStart.py](http://camerastart.py/)

- 3-1 修改 ip
    
    將 [cameraStart.py](http://camerastart.py/) 內的兩個 ip 位址
    都修改成剛剛得到的對外 ip
    
    ![20240521_16h09m10s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/ae17239b-fb19-477e-991b-d87dc6af7669.png)
    
    ![20240521_16h09m10s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/cf5abbda-fa12-4c65-b96c-b627aa8ddf52.png)
    
- 3-2 執行
    
    重新開啟一個終端 並按照step1 啟動虛擬環境pi
    
    連接鏡頭
    
    python [cameraStart.py](http://camerastart.py/)
    
    開始獲取影像
    
    ![20240521_16h14m24s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/96144eb1-8def-4dc5-8d36-ec27a3565f88.png)
    
    ![20240521_16h15m20s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/20240521_16h15m20s_grim.png)
    

## step4 在macOS採集跌倒訓練資料

**＊原本的 winsound 相關程式碼要註解或刪除**

- 4-1 先以影片嘗試執行 MediapipeDataCollect.py
    
    python3 OldManFalls/MediapipeDataCollect.py
    
    ```python
    import csv
    import time
    import cv2
    import mediapipe as mp
    import numpy as np
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture('./OldManFalls/Falling.mp4')
    
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
        csv_filename = "./OldManFalls/dataSet/0.csv"
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
    
    ```
    
    ![截圖 2024-05-22 上午9.54.27.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258A%25E5%258D%25889.54.27.png)
    
- 4-2 接著自己採集
    
    修改 line32
    
    採集正常站立：csv_filename = "./OldManFalls/dataSet/0.csv”
    
    採集跌倒姿勢：csv_filename = "./OldManFalls/dataSet/1.csv”
    
    因健身影片的姿勢多變且豐富，我分別將健身影片剪輯為只有站立和倒臥作為資料採集來源
    
    站立 stand.mp4：[https://youtu.be/csUIfpDQMno?si=3NdbOvpB7vqgZmbT](https://youtu.be/csUIfpDQMno?si=3NdbOvpB7vqgZmbT)
    
    ![截圖 2024-05-22 晚上9.19.11.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E6%2599%259A%25E4%25B8%258A9.19.11.png)
    
    倒臥 fall.mp4 ：[https://youtu.be/074eiXBWpm4?si=5cVKI1g48_4y790D](https://youtu.be/074eiXBWpm4?si=5cVKI1g48_4y790D)
    
    ![截圖 2024-05-22 晚上9.23.39.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E6%2599%259A%25E4%25B8%258A9.23.39.png)
    
    ![截圖 2024-05-22 上午10.07.44.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258A%25E5%258D%258810.07.44.png)
    
- 4-3 資料集大小
    
    站立：210 筆
    
    倒臥：215 筆
    
    csv裡每列是一筆全身關節位置
    
    ![截圖 2024-05-22 上午10.08.36.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258A%25E5%258D%258810.08.36.png)
    

## step5 在macOS 開始訓練AI模型

- 5-1 SVC訓練
    
    python3 OldManFalls/trainSVC.py
    
    ![截圖 2024-05-22 下午2.35.17.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258B%25E5%258D%25882.35.17.png)
    
- 5-2 pyTorch訓練
    
    python3 OldManFalls/trainTorch.py
    
    - **＊if 報錯 ModuleNotFoundError: No module named 'torch’**
        
        執行 pip install torch torchvision torchaudio
        
        ![截圖 2024-05-22 上午10.39.49.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258A%25E5%258D%258810.39.49.png)
        
    
    ![截圖 2024-05-22 下午2.36.10.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258B%25E5%258D%25882.36.10.png)
    

訓練完成後會有兩個模型檔

[model.pth](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/model.pth)

[svm_model.pkl](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/svm_model.pkl)

![截圖 2024-05-22 上午10.42.45.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258A%25E5%258D%258810.42.45.png)

## step6 透過 FLASK 傳送辨識結果

將在mac上訓練好的模型傳到樹莓派上

在樹莓派啟動app.py 

- SVC 模型
    
    修改 [cameraStartSVM.py](http://camerastartsvm.py/)
    
    兩個 ip 位置，改為對外 ip
    
    接上鏡頭
    
    python [cameraStartSVM.py](http://camerastartsvm.py/)
    

- pyTorch 模型
    
    修改 [cameraStartTorch.py](http://camerastarttorch.py/)
    
    兩個 ip 位置，改為對外 ip
    
    接上鏡頭
    
    python [cameraStartTorch.py](http://camerastarttorch.py/)
    

![20240521_16h14m24s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/59e78755-efe3-4e33-bf7b-aaaf48d2da64.png)

![20240522_17h38m33s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/3391becc-a7cb-4dda-b9c4-acf958834668.png)

辨識結果會傳到網頁上顯示，畫面左方顯示
'Normal',或'Fall down'。

辨識結果

![截圖 2024-05-22 下午6.27.36.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258B%25E5%258D%25886.27.36.png)

![截圖 2024-05-24 下午4.23.51.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-24_%25E4%25B8%258B%25E5%258D%25884.23.51.png)

## step7 以影片進行辨識

修改 [cameraStartSVM.py](http://camerastartsvm.py/) 和 [cameraStartTorch.py](http://camerastarttorch.py/) 

```python
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./OldManFalls/funny.mp4')
```

![截圖 2024-05-22 下午6.35.55.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/%25E6%2588%25AA%25E5%259C%2596_2024-05-22_%25E4%25B8%258B%25E5%258D%25886.35.55.png)

**辨識超好笑的跌倒影片連結** 

（因為是連對外ip進行螢幕錄影 有點delay）

svm：[https://youtu.be/AyiJFFsliBs](https://youtu.be/AyiJFFsliBs)

pytorch：[https://youtu.be/1lziVNCokYM](https://youtu.be/1lziVNCokYM)

#!/bin/bash

cd ~
cd Downloads

# ls

# Download arduino-1.8.19-linuxaarch64.tar.xz

wget -O arduino-1.8.19-linuxaarch64.tar.xz [https://downloads.arduino.cc/arduino-1.8.19-linuxaarch64.tar.xz?_gl=1*1ukuqwx*_ga*MTIxMDcyNjM4OS4xNzA4NzgxNjcx*_ga_NEXN8H46L5*MTcwODkzODI5MS40LjAuMTcwODkzODI5MS4wLjAuMA](https://downloads.arduino.cc/arduino-1.8.19-linuxaarch64.tar.xz?_gl=1*1ukuqwx*_ga*MTIxMDcyNjM4OS4xNzA4NzgxNjcx*_ga_NEXN8H46L5*MTcwODkzODI5MS40LjAuMTcwODkzODI5MS4wLjAuMA)..*_fplc*aW5nTmlpNXJBT1E5Rk1wMnZGdHg5SWVIVlVZOUtLSkRIVzIlMkI4T05JOFJRYVpHSXNTTUZpT0IwTklPUGZMQ0hhQnpSUTR6azVOZ0lxNUglMkZjWUJ6QXhJR294MlZtVFdxeiUyRiUyRk4xb1lMVGx1c1d6VHBOaHE4aTIyanV1c094eUElM0QlM0Q.

# Unzip the file

tar -xf arduino-1.8.19-linuxaarch64.tar.xz

# Move the file to '/opt' folder

sudo mv arduino-1.8.19 /opt

# Install Arduino IDE

sudo /opt/arduino-1.8.19/install.sh

# Download DHT11-2.1.0.zip library

wget -O DHT11-2.1.0.zip [https://downloads.arduino.cc/libraries/github.com/dhrubasaha08/DHT11-2.1.0.zip?_gl=1*15b7zco*_ga*MTIxMDcyNjM4OS4xNzA4NzgxNjcx*_ga_NEXN8H46L5*MTcwODkxMDg4Ni4zLjEuMTcwODkxNjM5OC4wLjAuMA](https://downloads.arduino.cc/libraries/github.com/dhrubasaha08/DHT11-2.1.0.zip?_gl=1*15b7zco*_ga*MTIxMDcyNjM4OS4xNzA4NzgxNjcx*_ga_NEXN8H46L5*MTcwODkxMDg4Ni4zLjEuMTcwODkxNjM5OC4wLjAuMA)..*_fplc*NE40b1FXZjc5TnFKOUM3WWlWb1dYJTJGZzVEeW9aZzl3Y1I3NmdpZWc0d0FXUndDOHpQclM2SExocCUyQkRqWnd3UmJzWElmNzhHbG9xU2JRbkw0MENXQWNxWTFlQ3NTYzVqTTdSJTJCelVLdTE4S0RTeVZyZng2ZDUyeWo2a2xqZjNnJTNEJTNE

192.168.50.65 

wget -O HttpClient.zip [https://github.com/amcewen/HttpClient/archive/refs/heads/master.zip](https://github.com/amcewen/HttpClient/archive/refs/heads/master.zip)

```python
#include <WiFi.h>
#include <DHT.h>   //引用DHT.h程式庫
#include <ArduinoJson.h>
#include <HTTPClient.h>

const char* ssid = "Y615";
const char* password = "nchumis123";

#define DHTPIN 2    // Digital pin connected to the DHT sensor
#define DHTTYPE DHT11   // Type of the DHT sensor

DHT dht(DHTPIN, DHTTYPE);//dht(接腳,感測元件類型DHT11)
#define SERVER_URL "http://192.168.50.36:5000/post_data"

void setup() {
    Serial.begin(9600);

    Serial.println("Connecting to WiFi...");
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }

    Serial.println("");
    Serial.println("WiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
}

void loop() {
  // Read temperature and humidity from DHT11 sensor
  float humidity = dht.readHumidity();
  float temperature = dht.readTemperature();
  
  // Check if any reads failed and exit early (to try again).
  if (isnan(humidity) || isnan(temperature)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }
  
  // Print values to Serial Monitor
  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println("%");
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println("°C");
  Serial.println("=======================================");
  
  // Create JSON object
  DynamicJsonDocument jsonDoc(200);
  jsonDoc["temperature"] = temperature;
  jsonDoc["humidity"] = humidity;

  // Serialize JSON object to string
  String jsonString;
  serializeJson(jsonDoc, jsonString);

  // Send data to the server via HTTP POST
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");

    int httpResponseCode = http.POST(jsonString);

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("HTTP Response code: " + String(httpResponseCode));
      Serial.println("Response from server: " + response);
    } else {
      Serial.println("Error on sending POST: " + String(httpResponseCode));
      Serial.println(http.errorToString(httpResponseCode));
    }

    http.end();  // Free resources
  } else {
    Serial.println("Error in WiFi connection");
  }

  delay(2000); // Delay before next reading
}
```

![20240530_16h26m56s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/20240530_16h26m56s_grim.png)

![20240530_16h32m49s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/20240530_16h32m49s_grim.png)

![20240530_16h43m07s_grim.png](HW7%20Raspberry%20Pi%20CV%20%E8%B7%8C%E5%80%92%E5%81%B5%E6%B8%AC%200521388280ab46a1979092125bcc41d2/20240530_16h43m07s_grim.png)