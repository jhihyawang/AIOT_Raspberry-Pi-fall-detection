import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# 讀取資料
data_0 = pd.read_csv("./OldManFalls/dataSet/0.csv")
data_1 = pd.read_csv("./OldManFalls/dataSet/1.csv")

# 將資料轉換為numpy array
data_0 = np.array(data_0)
data_1 = np.array(data_1)

# 打亂資料順序
data_0 = shuffle(data_0)
data_1 = shuffle(data_1)

# 將資料合併並正規化
data_combined = np.concatenate((data_0, data_1), axis=0)
scaler = StandardScaler()
data_combined = scaler.fit_transform(data_combined)

# 標籤資料
labels_0 = np.zeros(data_0.shape[0], dtype=np.int64)
labels_1 = np.ones(data_1.shape[0], dtype=np.int64)
labels_combined = np.concatenate((labels_0, labels_1), axis=0)

# 將資料轉換為tensor
data_combined_tensor = torch.tensor(data_combined, dtype=torch.float32)
labels_combined_tensor = torch.tensor(labels_combined, dtype=torch.long)

# 定義 DataLoader
dataset = TensorDataset(data_combined_tensor, labels_combined_tensor)
batch_size = 64
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


model = SimpleModel()

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 50
for epoch in range(num_epochs):
    correct = 0
    total = 0
    total_loss = 0
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    avg_loss = total_loss / len(data_loader)
    print(
        f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}')

# 儲存模型
torch.save(model.state_dict(), './OldManFalls/model.pth')
