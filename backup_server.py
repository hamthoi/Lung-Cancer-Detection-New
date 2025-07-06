from flask import Flask, request, jsonify
import numpy as np
import os
from tensorflow.keras import layers, models # type: ignore
import threading
from datetime import datetime

app = Flask(__name__)

lock = threading.Lock()
new_weights = None  # Store only the latest global weights
connected_clients = set()

# Model definition (must match client/main server)
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

INPUT_SHAPE = (5, 10, 10, 1)
global_model = build_model(INPUT_SHAPE)

# On startup, try to load weights
if os.path.exists("global_model.weights.h5"):
    global_model.load_weights("global_model.weights.h5")
    print("[BACKUP SERVER] Loaded global model weights from disk.")
    new_weights = global_model.get_weights()

@app.route('/receive_backup', methods=['POST'])
def receive_backup():
    global new_weights
    file = request.files['file']
    file.save("global_model.weights.h5")
    print("[BACKUP SERVER] Received backup weights.")
    with lock:
        global_model.load_weights("global_model.weights.h5")
        new_weights = global_model.get_weights()
    return "OK", 200

@app.route('/download', methods=['GET'])
def download():
    client_ip = request.remote_addr
    with lock:
        if new_weights is None:
            print(f"[BACKUP SERVER] No global weights available yet for {client_ip}")
            return jsonify({"weights": None})
        weights_to_send = [w.tolist() for w in new_weights]
    print(f"[BACKUP SERVER] Sent global weights to {client_ip}")
    return jsonify({"weights": weights_to_send})

def federated_average(global_weights, client_weights, client_samples, global_samples=1):
    total = client_samples + global_samples
    return [
        (gw * global_samples + cw * client_samples) / total
        for gw, cw in zip(global_weights, client_weights)
    ]

@app.route('/upload', methods=['POST'])
def upload():
    global new_weights
    client_ip = request.remote_addr
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = request.get_json()
    weights = [np.array(w) for w in data["weights"]]
    num_samples = data.get("num_samples", 1)
    print(f"[{timestamp}] [BACKUP SERVER] Received weights from {client_ip} with {num_samples} samples.")

    with lock:
        # Load current global weights
        global_weights = global_model.get_weights()
        # Federated average
        averaged_weights = federated_average(global_weights, weights, num_samples, global_samples=1)
        global_model.set_weights(averaged_weights)
        global_model.save_weights("global_model.weights.h5")
        new_weights = global_model.get_weights()
        print(f"[{timestamp}] [BACKUP SERVER] Updated global model by averaging with client weights from {client_ip}")

    return jsonify({"status": "accepted"})

if __name__ == "__main__":
    print("[BACKUP SERVER] Backup server is running on port 5001...")
    app.run(host="localhost", port=5001)