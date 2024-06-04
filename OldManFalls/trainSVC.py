import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 讀取數據
data_0 = pd.read_csv('./OldManFalls/dataSet/0.csv', header=None)  # 假設0.csv包含類別為0的數據
data_1 = pd.read_csv('./OldManFalls/dataSet/1.csv', header=None)  # 假設1.csv包含類別為1的數據

# 建立標籤
data_0['label'] = 0
data_1['label'] = 1

# 合併數據
data = pd.concat([data_0, data_1], ignore_index=True)

# 提取特徵和標籤
x = data.drop('label', axis=1)
y = data['label']

# 劃分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# 使用SVC模型進行訓練
model = SVC()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 儲存模型
joblib.dump(model, './OldManFalls/svm_model.pkl')
