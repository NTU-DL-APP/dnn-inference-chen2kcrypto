import numpy as np
import tensorflow as tf
import json
import os

# === 檔案路徑設定 ===
YOUR_MODEL_NAME = 'fashion_mnist'  # 不含副檔名
TF_MODEL_PATH = f'{YOUR_MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_NAME}.json'

# === 確認模型檔存在 ===
if not os.path.exists(TF_MODEL_PATH):
    raise FileNotFoundError(f"找不到模型檔：{TF_MODEL_PATH}")

# === 載入 Keras 模型並儲存權重與架構 ===
model = tf.keras.models.load_model(TF_MODEL_PATH)

# 儲存原始 Keras 權重（結構化）
params = {}
print("🔍 提取模型權重...\n")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        for i, w in enumerate(weights):
            param_name = f"{layer.name}_{i}"
            print(f"  {param_name}: shape={w.shape}")
            params[param_name] = w
        print()
np.savez(MODEL_WEIGHTS_PATH, **params)
print(f"✅ 權重儲存至 {MODEL_WEIGHTS_PATH}")

# 儲存架構為 JSON 格式
arch = []
for layer in model.layers:
    config = layer.get_config()
    info = {
        "name": layer.name,
        "type": layer.__class__.__name__,
        "config": config,
        "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
    }
    arch.append(info)
with open(MODEL_ARCH_PATH, "w") as f:
    json.dump(arch, f, indent=2)
print(f"✅ 架構儲存至 {MODEL_ARCH_PATH}")

# === NumPy 模型推論區 ===

# 載入架構與權重
weights = np.load(MODEL_WEIGHTS_PATH)
with open(MODEL_ARCH_PATH) as f:
    architecture = json.load(f)

# 定義激活函數
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# Dense、Flatten 等函數
def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b

# 前向推論函數
def forward(x):
    for layer in architecture:
        lname = layer['name']
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

    return x

# === 測試輸入 ===
# 模擬一筆 Fashion MNIST 圖片（28x28），已 Flatten
dummy_input = np.random.rand(1, 28*28).astype(np.float32)

# 前向傳播推論
output = forward(dummy_input)

print("\n🧠 模型輸出（機率分布）:", output)
print("✅ 預測類別:", np.argmax(output, axis=-1))
