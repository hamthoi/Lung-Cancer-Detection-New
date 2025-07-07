import numpy as np
from flask import Flask, request, jsonify
import threading
import os
from tensorflow.keras import layers, models # type: ignore
from datetime import datetime
import shutil
import requests
import time

app = Flask(__name__)

lock = threading.Lock()
new_weights = None  # Store only the latest global weights
connected_clients = set()  # Track connected clients
client_updates = []  # Queue to store client updates: [(weights, num_samples, client_ip), ...]
MAX_CLIENTS = 10  # Maximum number of clients before auto-executing FedAvg
MIN_CLIENTS_FOR_TIMER = 2  # Minimum clients to start countdown timer
COUNTDOWN_SECONDS = 30  # Countdown timer duration

# FedAvg tracking
fedavg_round = 0  # Current FedAvg round number
fedavg_history = []  # History of FedAvg executions: [(round, timestamp, participants, total_samples, accuracy), ...]

# Timer tracking
countdown_timer = None  # Thread for countdown timer
timer_active = False  # Whether countdown timer is active
timer_start_time = None  # When the timer started

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

def countdown_timer_function():
    """Countdown timer function that executes FedAvg after COUNTDOWN_SECONDS."""
    global timer_active, countdown_timer
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [SERVER] Starting {COUNTDOWN_SECONDS}-second countdown timer...")
    
    # Countdown loop
    for remaining in range(COUNTDOWN_SECONDS, 0, -1):
        if not timer_active:
            print(f"[{timestamp}] [SERVER] Countdown timer cancelled.")
            return
        
        # Print countdown every 10 seconds
        if remaining % 10 == 0 or remaining <= 5:
            print(f"[{timestamp}] [SERVER] Countdown: {remaining} seconds remaining...")
        
        time.sleep(1)
    
    # Timer expired, execute FedAvg
    if timer_active:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SERVER] Countdown timer expired. Executing FedAvg...")
        execute_fedavg()
    
    # Reset timer state
    timer_active = False
    countdown_timer = None

def start_countdown_timer():
    """Start the countdown timer if not already active."""
    global timer_active, countdown_timer
    
    if not timer_active and len(client_updates) >= MIN_CLIENTS_FOR_TIMER:
        timer_active = True
        countdown_timer = threading.Thread(target=countdown_timer_function, daemon=True)
        countdown_timer.start()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SERVER] Started {COUNTDOWN_SECONDS}-second countdown timer for {len(client_updates)} clients")

def stop_countdown_timer():
    """Stop the countdown timer if active."""
    global timer_active
    
    if timer_active:
        timer_active = False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SERVER] Countdown timer stopped.")

def federated_average(client_updates_list):
    """
    Execute federated averaging on collected client updates.
    
    Args:
        client_updates_list: List of tuples (weights, num_samples, client_ip)
    
    Returns:
        Averaged weights
    """
    if not client_updates_list:
        print("[SERVER] No client updates to aggregate.")
        return None
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [SERVER] Starting FedAvg with {len(client_updates_list)} clients...")
    
    # Calculate total samples across all clients
    total_samples = sum(samples for _, samples, _ in client_updates_list)
    print(f"[{timestamp}] [SERVER] Total samples across all clients: {total_samples}")
    
    # Initialize averaged weights with first client's weights
    first_weights, first_samples, first_client = client_updates_list[0]
    averaged_weights = [w * (first_samples / total_samples) for w in first_weights]
    
    # Add contributions from remaining clients
    for weights, samples, client_ip in client_updates_list[1:]:
        weight_ratio = samples / total_samples
        for i, (avg_w, client_w) in enumerate(zip(averaged_weights, weights)):
            averaged_weights[i] = avg_w + client_w * weight_ratio
    
    print(f"[{timestamp}] [SERVER] FedAvg completed successfully.")
    return averaged_weights

def execute_fedavg():
    """Execute federated averaging on queued client updates."""
    global client_updates, new_weights, fedavg_round, fedavg_history
    
    with lock:
        if not client_updates:
            print("[SERVER] No client updates to aggregate.")
            return
        
        # Stop countdown timer if active
        stop_countdown_timer()
        
        # Increment FedAvg round
        fedavg_round += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SERVER] Executing FedAvg Round {fedavg_round} with {len(client_updates)} clients...")
        
        # Log all clients participating in this round
        client_ips = [client_ip for _, _, client_ip in client_updates]
        total_samples = sum(samples for _, samples, _ in client_updates)
        print(f"[{timestamp}] [SERVER] Participating clients: {client_ips}")
        print(f"[{timestamp}] [SERVER] Total samples: {total_samples}")
        
        # Execute federated averaging
        averaged_weights = federated_average(client_updates)
        
        if averaged_weights is not None:
            # Update global model with averaged weights
            global_model.set_weights(averaged_weights)
            global_model.save_weights("global_model.weights.h5")
            new_weights = global_model.get_weights()
            
            # Send backup to backup server
            send_weights_to_backup("http://localhost:5001/receive_backup")
            
            # Evaluate new global model
            new_acc = evaluate_on_val(global_model, val_size=10)
            print(f"[{timestamp}] [SERVER] New global model accuracy: {new_acc:.4f}")
            
            # Record FedAvg history
            fedavg_info = {
                "round": fedavg_round,
                "timestamp": timestamp,
                "participants": client_ips,
                "total_samples": total_samples,
                "accuracy": new_acc,
                "num_clients": len(client_updates)
            }
            fedavg_history.append(fedavg_info)
            
            # Clear the queue
            client_updates.clear()
            
            print(f"[{timestamp}] [SERVER] FedAvg Round {fedavg_round} completed. Global model updated and saved.")
        else:
            print(f"[{timestamp}] [SERVER] FedAvg Round {fedavg_round} failed. Keeping existing global model.")

def send_weights_to_backup(backup_url="http://localhost:5001/receive_backup"):
    with open("global_model.weights.h5", "rb") as f:
        files = {"file": f}
        try:
            response = requests.post(backup_url, files=files)
            if response.status_code == 200:
                print("[SERVER] Backup weights sent to backup server.")
            else:
                print(f"[SERVER] Failed to send backup weights: {response.status_code}")
        except Exception as e:
            print(f"[SERVER] Error sending backup weights: {e}")

@app.route('/upload', methods=['POST'])
def upload():
    global client_updates
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
        # Check if this is the first update (no global model exists)
        first_update = not os.path.exists("global_model.weights.h5")
        
        if first_update:
            # First update: set global model to client weights
            global_model.set_weights(weights)
            global_model.save_weights("global_model.weights.h5")
            send_weights_to_backup("http://localhost:5001/receive_backup")
            print(f"[{timestamp}] [SERVER] First update: set global model to client weights from {client_ip}")
            global new_weights
            new_weights = global_model.get_weights()
            status = "accepted"
        else:
            # Add client update to queue
            client_updates.append((weights, num_samples, client_ip))
            print(f"[{timestamp}] [SERVER] Added {client_ip} to FedAvg queue. Queue size: {len(client_updates)}/{MAX_CLIENTS}")
            
            # Check if we should auto-execute FedAvg (10 clients reached)
            if len(client_updates) >= MAX_CLIENTS:
                print(f"[{timestamp}] [SERVER] Queue full ({MAX_CLIENTS} clients). Auto-executing FedAvg...")
                execute_fedavg()
                status = "accepted_and_fedavg_executed"
            else:
                # Check if we should start countdown timer (at least 2 clients)
                if len(client_updates) >= MIN_CLIENTS_FOR_TIMER and not timer_active:
                    start_countdown_timer()
                
                status = "queued"
    
    return jsonify({
        "status": status, 
        "queue_size": len(client_updates), 
        "max_clients": MAX_CLIENTS,
        "fedavg_round": fedavg_round,
        "timer_active": timer_active,
        "countdown_seconds": COUNTDOWN_SECONDS if timer_active else None
    })

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
        
        # Get latest FedAvg information
        latest_fedavg = fedavg_history[-1] if fedavg_history else None
        
        response_data = {
            "weights": weights_to_send,
            "fedavg_info": latest_fedavg
        }
        
        if latest_fedavg:
            print(f"[{timestamp}] [SERVER] Sent global weights (FedAvg Round {latest_fedavg['round']}) to {client_ip}")
        else:
            print(f"[{timestamp}] [SERVER] Sent global weights to {client_ip}")
            
        return jsonify(response_data)

@app.route('/status', methods=['GET'])
def status():
    """Get server status including queue information."""
    with lock:
        return jsonify({
            "queue_size": len(client_updates),
            "max_clients": MAX_CLIENTS,
            "connected_clients": len(connected_clients),
            "has_global_model": os.path.exists("global_model.weights.h5"),
            "queued_clients": [client_ip for _, _, client_ip in client_updates],
            "current_fedavg_round": fedavg_round,
            "latest_fedavg": fedavg_history[-1] if fedavg_history else None,
            "timer_active": timer_active,
            "countdown_seconds": COUNTDOWN_SECONDS,
            "min_clients_for_timer": MIN_CLIENTS_FOR_TIMER
        })

@app.route('/fedavg_history', methods=['GET'])
def get_fedavg_history():
    """Get complete FedAvg history."""
    return jsonify({
        "fedavg_history": fedavg_history,
        "total_rounds": fedavg_round
    })

def command_listener():
    """Listen for manual commands from console."""
    while True:
        try:
            cmd = input().strip().lower()
            if cmd == "execute":
                execute_fedavg()
            elif cmd == "status":
                with lock:
                    print(f"\n[SERVER] Status:")
                    print(f"  Queue size: {len(client_updates)}/{MAX_CLIENTS}")
                    print(f"  Connected clients: {len(connected_clients)}")
                    print(f"  Queued clients: {[client_ip for _, _, client_ip in client_updates]}")
                    print(f"  Has global model: {os.path.exists('global_model.weights.h5')}")
                    print(f"  Current FedAvg round: {fedavg_round}")
                    print(f"  Timer active: {timer_active}")
                    if timer_active:
                        print(f"  Countdown timer: {COUNTDOWN_SECONDS} seconds")
                    if fedavg_history:
                        latest = fedavg_history[-1]
                        print(f"  Latest FedAvg: Round {latest['round']} - {latest['num_clients']} clients, {latest['total_samples']} samples, {latest['accuracy']:.4f} accuracy")
            elif cmd == "history":
                print(f"\n[SERVER] FedAvg History:")
                for fedavg in fedavg_history:
                    print(f"  Round {fedavg['round']}: {fedavg['timestamp']} - {fedavg['num_clients']} clients, {fedavg['total_samples']} samples, {fedavg['accuracy']:.4f} accuracy")
                    print(f"    Participants: {fedavg['participants']}")
            elif cmd == "clear":
                with lock:
                    client_updates.clear()
                    stop_countdown_timer()
                    print("[SERVER] Cleared client update queue and stopped timer.")
            elif cmd == "help":
                print("\n[SERVER] Available commands:")
                print("  execute  - Run FedAvg on queued client updates")
                print("  status   - Show server status and queue information")
                print("  history  - Show complete FedAvg history")
                print("  clear    - Clear the client update queue and stop timer")
                print("  help     - Show this help message")
                print("  quit     - Exit server\n")
            elif cmd == "quit":
                print("[SERVER] Shutting down...")
                os._exit(0)
        except (EOFError, KeyboardInterrupt):
            print("\n[SERVER] Shutting down...")
            os._exit(0)
        except Exception as e:
            print(f"[SERVER] Command error: {e}")

if __name__ == '__main__':
    # Start command listener in a separate thread
    command_thread = threading.Thread(target=command_listener, daemon=True)
    command_thread.start()
    
    print("[SERVER] Federated server is running on port 5000...")
    print("[SERVER] Server is accessible from other machines on the network")
    print("[SERVER] Clients should connect to this machine's IP address")
    print(f"[SERVER] FedAvg will auto-execute when {MAX_CLIENTS} clients are queued")
    print(f"[SERVER] OR when {MIN_CLIENTS_FOR_TIMER}+ clients are queued for {COUNTDOWN_SECONDS} seconds")
    print("[SERVER] Type 'help' for available commands")
    app.run(host='0.0.0.0', port=5000)