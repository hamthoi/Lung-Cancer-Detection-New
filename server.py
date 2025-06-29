import numpy as np
from flask import Flask, request, jsonify
import threading
import os
from tensorflow.keras import layers, models # type: ignore
from datetime import datetime

app = Flask(__name__)

lock = threading.Lock()
new_weights = None  # Store only the latest global weights
connected_clients = set()  # Track connected clients

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

def get_random_validation_set(val_size=10):
    val_data = np.load("processedData.npy", allow_pickle=True)
    np.random.shuffle(val_data)
    validationData = val_data[:val_size]
    X_val = np.array([i[0] for i in validationData])
    y_val = np.array([i[1] for i in validationData])
    X_val = X_val[..., np.newaxis]
    return X_val, y_val

def evaluate_on_val(model, val_size=10):
    X_val, y_val = get_random_validation_set(val_size)
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    return acc

def log_connection(client_ip):
    """Log client connection with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [SERVER] New client connected from: {client_ip}")
    connected_clients.add(client_ip)
    print(f"[{timestamp}] [SERVER] Active clients: {len(connected_clients)}")

def federated_average(global_weights, client_weights, client_samples, global_samples=1):
    """Weighted average of global and client weights."""
    total = client_samples + global_samples
    return [
        (gw * global_samples + cw * client_samples) / total
        for gw, cw in zip(global_weights, client_weights)
    ]

@app.route('/upload', methods=['POST'])
def upload():
    global new_weights  # <-- Move this to the top of the function
    client_ip = request.remote_addr
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log connection if new client
    if client_ip not in connected_clients:
        log_connection(client_ip)
    
    data = request.get_json(force=True)
    weights = [np.array(w) for w in data['weights']]
    num_samples = data['num_samples']
    
    print(f"[{timestamp}] [SERVER] Received update from {client_ip} with {num_samples} samples")
    
    with lock:
        first_update = not os.path.exists("global_model.weights.h5")
        if first_update:
            # First update: just take the client's model
            global_model.set_weights(weights)
            global_model.save_weights("global_model.weights.h5")
            print(f"[{timestamp}] [SERVER] First update: set global model to client weights from {client_ip}")
            new_weights = global_model.get_weights()
            status = "accepted"
        else:
            # Evaluate current global model on 10 random patients
            current_acc = evaluate_on_val(global_model, val_size=10)
            print(f"[{timestamp}] [SERVER] Accuracy of current global model: {current_acc:.4f}")
            # Set new weights and evaluate
            global_model.set_weights(weights)
            new_acc = evaluate_on_val(global_model, val_size=10)
            print(f"[{timestamp}] [SERVER] Accuracy of client model: {new_acc:.4f}")
            print(f"[{timestamp}] [SERVER] Current global acc: {current_acc:.4f}, Client acc: {new_acc:.4f}")
            if new_acc > current_acc:
                # Load previous global weights for averaging
                global_model.load_weights("global_model.weights.h5")
                global_weights = global_model.get_weights()
                # Federated average
                averaged_weights = federated_average(global_weights, weights, num_samples, global_samples=1)
                global_model.set_weights(averaged_weights)
                global_model.save_weights("global_model.weights.h5")
                print(f"[{timestamp}] [SERVER] Updated global model by averaging with client weights from {client_ip}")
                new_weights = global_model.get_weights()
                status = "accepted"
            else:
                # Revert to previous weights
                global_model.load_weights("global_model.weights.h5")
                print(f"[{timestamp}] [SERVER] Disregarded client weights from {client_ip} (no improvement)")
                status = "rejected"
    return jsonify({"status": status})

@app.route('/download', methods=['GET'])
def download():
    client_ip = request.remote_addr
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log connection if new client
    if client_ip not in connected_clients:
        log_connection(client_ip)
    
    with lock:
        if new_weights is None:
            print(f"[{timestamp}] [SERVER] No global weights available yet for {client_ip}")
            return jsonify({"weights": None})
        weights_to_send = [w.tolist() for w in new_weights]
    print(f"[{timestamp}] [SERVER] Sent global weights to {client_ip}")
    return jsonify({
        "weights": weights_to_send,
    })

if __name__ == '__main__':
    print("[SERVER] Federated server is running on port 5000...")
    print("[SERVER] Server is accessible from other machines on the network")
    print("[SERVER] Clients should connect to this machine's IP address")
    app.run(host='0.0.0.0', port=5000)