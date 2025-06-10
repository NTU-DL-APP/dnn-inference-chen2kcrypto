import numpy as np
import json
import tensorflow as tf


YOUR_MODEL_NAME = 'fashion_mnist' # Default extension is h5
TF_MODEL_PATH = f'{YOUR_MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_NAME}.json'

model = tf.keras.models.load_model(TF_MODEL_PATH)

# Save weights to .npz (NumPy format)
weights = model.get_weights()
np.savez('model_weights.npz', *weights)

# Save architecture to JSON
with open('model_architecture.json', 'w') as json_file:
    json_file.write(model.to_json())
    
    import tensorflow as tf
import numpy as np

# === Step 1: Load Keras .h5 model ===
model = tf.keras.models.load_model(TF_MODEL_PATH)

# === Step 2: Print and collect weights ===
params = {}
print("🔍 Extracting weights from model...\n")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"Layer: {layer.name}")
        for i, w in enumerate(weights):
            param_name = f"{layer.name}_{i}"
            print(f"  {param_name}: shape={w.shape}")
            params[param_name] = w
        print()

# === Step 3: Save to .npz ===
np.savez(MODEL_WEIGHTS_PATH, **params)
print(f"✅ Saved all weights to {MODEL_WEIGHTS_PATH}")

# === Step 4: Reload and verify ===
print("\n🔁 Verifying loaded .npz weights...\n")
loaded = np.load(MODEL_WEIGHTS_PATH)

for key in loaded.files:
    print(f"{key}: shape={loaded[key].shape}")

# === Step 6: Extract architecture to JSON ===
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

print("✅ Architecture saved to model_architecture.json")

import numpy as np
import json

# === Load weights and architecture ===
weights = np.load(MODEL_WEIGHTS_PATH)
with open(MODEL_ARCH_PATH) as f:
    architecture = json.load(f)


# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward pass ===
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

# === Example usage ===
# Load a dummy image (batch size 1)
# Make sure it's shape: (1, 28, 28, 1)
dummy_input = np.random.rand(1, 28*28).astype(np.float32)
output = forward(dummy_input)

print("🧠 Output probabilities:", output)
print("✅ Predicted class:", np.argmax(output, axis=-1))