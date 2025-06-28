import numpy as np
from flask import Flask, request, jsonify
import threading
import os
from tensorflow.keras import layers, models # type: ignore

app = Flask(__name__)

client_updates = []  # Each item: (weights, num_samples)
lock = threading.Lock()
new_weights = None  # Store only the latest aggregated weights

# Define the model architecture (must match client)
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

INPUT_SHAPE = (5, 10, 10, 1)  # Must match client
global_model = build_model(INPUT_SHAPE)

# On startup, try to load weights
if os.path.exists("global_model.weights.h5"):
    global_model.load_weights("global_model.weights.h5")
    print("[SERVER] Loaded global model weights from disk.")
new_weights = global_model.get_weights()

# Load validation data (same as client)
val_data = np.load("processedData.npy", allow_pickle=True)
validationData = val_data[45:50]
X_val = np.array([i[0] for i in validationData])
y_val = np.array([i[1] for i in validationData])
X_val = X_val[..., np.newaxis]

def evaluate_on_val(model):
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    return acc

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json(force=True)
    weights = [np.array(w) for w in data['weights']]
    num_samples = data['num_samples']
    with lock:
        # Evaluate current global model
        current_acc = evaluate_on_val(global_model)
        # Set new weights and evaluate
        global_model.set_weights(weights)
        new_acc = evaluate_on_val(global_model)
        print(f"[SERVER] Current global acc: {current_acc:.4f}, Client acc: {new_acc:.4f}")
        if new_acc > current_acc:
            global_model.save_weights("global_model.weights.h5")
            print("[SERVER] Updated global model with improved client weights.")
            global new_weights
            new_weights = global_model.get_weights()
            status = "accepted"
        else:
            # Revert to previous weights
            if os.path.exists("global_model.weights.h5"):
                global_model.load_weights("global_model.weights.h5")
            print("[SERVER] Disregarded client weights (no improvement).")
            status = "rejected"
    return jsonify({"status": status})

@app.route('/download', methods=['GET'])
def download():
    with lock:
        if new_weights is None:
            print("[SERVER] No global weights available yet.")
            return jsonify({"weights": None})
        weights_to_send = [w.tolist() for w in new_weights]
    print("[SERVER] Sent new_weights to client.")
    return jsonify({
        "weights": weights_to_send,
    })

def fedavg_and_update():
    global new_weights, client_updates
    with lock:
        if not client_updates:
            print("[SERVER] No updates to aggregate.")
            return
        total = sum(num for _, num in client_updates)
        new_weights_local = None
        for weights, num in client_updates:
            scaled = [w * (num / total) for w in weights]
            if new_weights_local is None:
                new_weights_local = scaled
            else:
                new_weights_local = [nw + sw for nw, sw in zip(new_weights_local, scaled)]
        new_weights = new_weights_local
        global_model.set_weights(new_weights)
        global_model.save_weights("global_model.weights.h5")  # <-- Save to disk
        print(f"[SERVER] FedAvg executed. Aggregated {len(client_updates)} clients, total samples: {total}")
        print("[SERVER] Global model weights saved to global_model.weights.h5")
        client_updates = []

def command_listener():
    while True:
        cmd = input()
        if cmd.strip().lower() == "execute":
            fedavg_and_update()

if __name__ == '__main__':
    threading.Thread(target=command_listener, daemon=True).start()
    print("[SERVER] Federated server is running on port 5000...")
    app.run(host='0.0.0.0', port=5000)