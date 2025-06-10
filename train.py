import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers

# === 載入 Fashion MNIST 資料集 ===
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# === 資料預處理 ===
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten input: (28, 28) -> (784,)
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)

# === 建立模型 ===
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# === 編譯模型 ===
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
early_stop = callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# === 訓練模型 ===
history = model.fit(
    x_train, y_train,
    validation_split=0.15,
    epochs=30,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2
)

# === 評估模型 ===
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ 測試準確率: {acc:.4f}")

# === 儲存模型 ===
model.save("fashion_mnist.h5")
print("✅ 模型已儲存為 fashion_mnist.h5")
