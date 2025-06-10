import numpy as np
import json

# 路徑設定
YOUR_MODEL_NAME = 'fashion_mnist'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_NAME}.json'

# 載入權重與架構
weights = np.load(MODEL_WEIGHTS_PATH)
with open(MODEL_ARCH_PATH) as f:
    architecture = json.load(f)

# 激活函數
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# Flatten 與 Dense 實作
def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b

# NumPy 前向推論函數
def forward(x):
    for layer in architecture:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']

        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
        else:
            continue
    return x

# 載入 Fashion MNIST 測試集
from tensorflow.keras.datasets import fashion_mnist
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# 先針對第一筆資料做推論並印出詳細資訊
dummy_input = x_test[0:1]
output = forward(dummy_input)

print("\n🧠 模型輸出機率分布:", output)
print("✅ 預測類別:", np.argmax(output, axis=-1)[0])
print("🎯 真實標籤:", y_test[0])

# 計算整體準確率
batch_size = 128
num_samples = x_test.shape[0]
num_batches = (num_samples + batch_size - 1) // batch_size
correct = 0

for i in range(num_batches):
    start = i * batch_size
    end = min(start + batch_size, num_samples)
    batch_x = x_test[start:end]
    batch_y = y_test[start:end]

    preds = forward(batch_x)
    pred_labels = np.argmax(preds, axis=-1)
    correct += np.sum(pred_labels == batch_y)

accuracy = correct / num_samples
print(f"\n✅ NumPy 推論準確率: {accuracy * 100:.2f}%")
