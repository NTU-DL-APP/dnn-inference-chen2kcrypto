import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json

# 設定檔案路徑
YOUR_MODEL_NAME = 'fashion_mnist'
TF_MODEL_PATH = f'{YOUR_MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_NAME}.json'

# 載入資料
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0

# 建立模型
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 編譯並訓練
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# 儲存 Keras 模型
model.save(TF_MODEL_PATH)
print(f"✅ 模型儲存於 {TF_MODEL_PATH}")

# 提取並儲存權重
params = {}
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        for i, w in enumerate(weights):
            param_name = f"{layer.name}_{i}"
            params[param_name] = w
np.savez(MODEL_WEIGHTS_PATH, **params)
print(f"✅ 權重儲存於 {MODEL_WEIGHTS_PATH}")

# 儲存模型架構為 JSON
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
print(f"✅ 架構儲存於 {MODEL_ARCH_PATH}")
