import numpy as np
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)

client_updates = []  # Each item: (weights, num_samples)
lock = threading.Lock()
new_weights = None  # Store only the latest aggregated weights

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json(force=True)
    weights = [np.array(w) for w in data['weights']]
    num_samples = data['num_samples']
    with lock:
        client_updates.append((weights, num_samples))
        print(f"[SERVER] Queued update: {num_samples} samples. Total queued: {len(client_updates)}")
    return jsonify({"status": "queued", "queued_clients": len(client_updates)})

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
        print(f"[SERVER] FedAvg executed. Aggregated {len(client_updates)} clients, total samples: {total}")
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