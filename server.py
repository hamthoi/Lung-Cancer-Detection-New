import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras import layers, models # type: ignore

app = Flask(__name__)

# Global model definition (must match client)
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

# Initialize global model
INPUT_SHAPE = (5, 10, 10, 1)  # Must match client
global_model = build_model(INPUT_SHAPE)
global_weights = global_model.get_weights()
total_samples = 0

@app.route('/upload', methods=['POST'])
def upload():
    global global_weights, total_samples

    # Receive weights and sample count
    data = request.get_json(force=True)
    weights = [np.array(w) for w in data['weights']]
    num_samples = data['num_samples']

    # If first update, just set
    if total_samples == 0:
        global_weights = weights
        total_samples = num_samples
    else:
        # Weighted average (FedAvg)
        for i in range(len(global_weights)):
            global_weights[i] = (global_weights[i] * total_samples + weights[i] * num_samples) / (total_samples + num_samples)
        total_samples += num_samples

    # Update model
    global_model.set_weights(global_weights)
    print(f"[SERVER] Received update: {num_samples} samples. Total samples: {total_samples}")

    return jsonify({"status": "success", "total_samples": total_samples})

@app.route('/download', methods=['GET'])
def download():
    # Send current global weights
    weights = [w.tolist() for w in global_model.get_weights()]
    print("[SERVER] Sent global weights to client.")
    return jsonify({"weights": weights})

@app.route('/test', methods=['POST'])
def test():
    # Test the global model on provided data
    data = request.get_json(force=True)
    X = np.array(data['X'])
    y = np.array(data['y'])
    global_model.set_weights([np.array(w) for w in data.get('weights', global_model.get_weights())])
    loss, acc = global_model.evaluate(X, y, verbose=0)
    print(f"[SERVER] Tested global model. Loss: {loss}, Accuracy: {acc}")
    return jsonify({"loss": float(loss), "accuracy": float(acc)})

if __name__ == '__main__':
    print("[SERVER] Federated server is running on port 5000...")
    app.run(host='0.0.0.0', port=5000)