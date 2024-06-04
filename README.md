## Raspberry Pi  CV 跌倒偵測

實驗硬體：

macOS M1 版本 14.4.1

Raspberry Pi 4 Model B 64bit

實驗結果：

<img width="1280" alt="截圖 2024-05-22 下午6 35 55" src="https://github.com/jhihyawang/fallin_detect/assets/157604262/23763c0a-3d94-4963-9324-b9f2288cf1e8">


## step1 兩台電腦的環境安裝

建立虛擬環境 python -m venv pi

啟動虛擬環境 source pi/bin/activate

安裝相關套件 pip install -r requirements.txt  

- **＊mac 報錯**
    
    ERROR: Could not find a version that satisfies the requirement mediapipe==0.10.10 (from versions: 0.10.13, 0.10.14)
    ERROR: No matching distribution found for mediapipe==0.10.10
    
    **需修改 mediapipe==0.10.14**

    
## step2 執行 [app.py](http://app.py)

- python app.py
    
    會得到兩個ip
    
    對內 ip : [http://127.0.0.1:5000](http://127.0.0.1:5000/)
    
    對外 ip：[http://172.20.10.5:5000](http://172.20.10.5:5000/)
  
- 樹莓派 打開瀏覽器
    
    輸入對內ip：[http://127.0.0.1:5000](http://127.0.0.1:5000/)
    

- mac 打開瀏覽器
    
    輸入對外ip：[http://172.20.10.5:5000](http://172.20.10.5:5000/)
    
    可看到樹莓派伺服器的網頁
    

## step3 修改並執行 [cameraStart.py](http://camerastart.py/)

- 3-1 修改 ip
    
    將 [cameraStart.py](http://camerastart.py/) 內的兩個 ip 位址
    都修改成剛剛得到的對外 ip
    

- 3-2 執行
    
    重新開啟一個終端 並按照step1 啟動虛擬環境pi
    
    連接鏡頭
    
    python [cameraStart.py](http://camerastart.py/)
    
    開始獲取影像
    

## step4 在macOS採集跌倒訓練資料

**＊原本的 winsound 相關程式碼要註解或刪除**

- 4-1 先以影片嘗試執行 MediapipeDataCollect.py

    
- 4-2 接著自己採集
    
    修改 line32
    
    採集正常站立：csv_filename = "./OldManFalls/dataSet/0.csv”
    
    採集跌倒姿勢：csv_filename = "./OldManFalls/dataSet/1.csv”
    
    因健身影片的姿勢多變且豐富，我分別將健身影片剪輯為只有站立和倒臥作為資料採集來源
    
    站立 stand.mp4：[https://youtu.be/csUIfpDQMno?si=3NdbOvpB7vqgZmbT](https://youtu.be/csUIfpDQMno?si=3NdbOvpB7vqgZmbT)
  
    
    倒臥 fall.mp4 ：[https://youtu.be/074eiXBWpm4?si=5cVKI1g48_4y790D](https://youtu.be/074eiXBWpm4?si=5cVKI1g48_4y790D)
    

    
- 4-3 資料集大小
    
    站立：210 筆
    
    倒臥：215 筆
    
    csv裡每列是一筆全身關節位置
    
 

## step5 在macOS 開始訓練AI模型

- 5-1 SVC訓練
    
    python3 OldManFalls/trainSVC.py
    
    
- 5-2 pyTorch訓練
    
    python3 OldManFalls/trainTorch.py
    
    - **＊if 報錯 ModuleNotFoundError: No module named 'torch’**
        
        執行 pip install torch torchvision torchaudio
        
訓練完成後會有兩個模型檔


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
    


辨識結果會傳到網頁上顯示，畫面左方顯示
'Normal',或'Fall down'。


## step7 以影片進行辨識

修改 [cameraStartSVM.py](http://camerastartsvm.py/) 和 [cameraStartTorch.py](http://camerastarttorch.py/) 

```python
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('./OldManFalls/funny.mp4')
```

**辨識超好笑的跌倒影片連結** 

（因為是連對外ip進行螢幕錄影 有點delay）

svm：[https://youtu.be/AyiJFFsliBs](https://youtu.be/AyiJFFsliBs)

pytorch：[https://youtu.be/1lziVNCokYM](https://youtu.be/1lziVNCokYM)
