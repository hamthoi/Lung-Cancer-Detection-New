# Lung Cancer Detection - Federated Learning System

A distributed federated learning system for lung cancer detection using 3D CNNs on DICOM medical images. The system consists of one server and multiple clients that collaborate to train a global model while keeping data local.

## 🚀 New Features

### Enhanced Federated Averaging (FedAvg)
- **True FedAvg Implementation**: The server now properly implements federated averaging by collecting updates from multiple clients before aggregating
- **Client Queue System**: Up to 10 clients can queue their updates before FedAvg execution
- **Automatic Execution**: FedAvg runs automatically when 10 clients are queued
- **Manual Execution**: Server supports manual "execute" command for on-demand FedAvg
- **Weighted Averaging**: Proper weighted averaging based on number of samples per client

### Server Enhancements
- **Command Interface**: Interactive console commands for server management
- **Status Endpoint**: Real-time server status and queue information
- **Connection Logging**: Detailed logging of client connections with timestamps
- **Network Binding**: Server binds to all interfaces for network accessibility

### Client Improvements
- **Server Status Check**: New button to check server queue and connection status
- **Better Feedback**: Enhanced status messages showing queue progress
- **Backup Server Support**: Automatic fallback to backup server if main server fails

## 📋 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client 1      │    │   Client 2      │    │   Client N      │
│   (Laptop)      │    │   (Laptop)      │    │   (Laptop)      │
│                 │    │                 │    │                 │
│ • Import Data   │    │ • Import Data   │    │ • Import Data   │
│ • Preprocess    │    │ • Preprocess    │    │ • Preprocess    │
│ • Train Local   │    │ • Train Local   │    │ • Train Local   │
│ • Send Updates  │    │ • Send Updates  │    │ • Send Updates  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │        Server             │
                    │   (Central Machine)       │
                    │                           │
                    │ • Collect Client Updates  │
                    │ • Queue Management        │
                    │ • FedAvg Execution        │
                    │ • Global Model Distribution│
                    │ • Backup Server Sync      │
                    └───────────────────────────┘
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Pydicom
- Flask
- Requests

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python numpy pandas pydicom flask requests matplotlib
   ```
3. Activate your virtual environment (if using one)

## 🚀 Usage

### Starting the Server

1. **Start the main server**:
   ```bash
   python server.py
   ```

2. **Server Commands** (available in server console):
   - `execute` - Manually run FedAvg on queued client updates
   - `status` - Show server status and queue information
   - `clear` - Clear the client update queue
   - `help` - Show available commands
   - `quit` - Exit server

### Starting Clients

1. **Start the client GUI**:
   ```bash
   python client.py
   ```

2. **Configure server connection**:
   - Enter the server IP address (e.g., `192.168.1.100`)
   - Click "Connect to Server" to test connection
   - Use "Check Server Status" to monitor queue progress

### Workflow

1. **Data Import**: Each client imports their local DICOM data
2. **Preprocessing**: Data is preprocessed and prepared for training
3. **Local Training**: Each client trains their local model
4. **Update Upload**: Clients send their trained weights to the server
5. **FedAvg Execution**: Server aggregates updates from multiple clients
6. **Global Model Distribution**: Updated global model is distributed to all clients

## 🧪 Testing the System

### Automated Test
Run the test script to simulate multiple clients:
```bash
python test_fedavg.py
```

This will:
- Simulate 12 clients sending updates
- Test automatic FedAvg execution when 10 clients are queued
- Show server status throughout the process

### Manual Testing
1. Start the server
2. Start multiple client instances
3. Import data and train on each client
4. Upload local models to server
5. Monitor queue progress using "Check Server Status"
6. Download updated global model

## 📊 FedAvg Process

### How It Works
1. **Client Update Collection**: Server collects model updates from clients in a queue
2. **Queue Management**: Updates are queued until 10 clients are ready (configurable)
3. **Weighted Averaging**: FedAvg computes weighted average based on sample counts:
   ```
   Global_Weight = Σ(Client_Weight_i × Sample_Ratio_i)
   where Sample_Ratio_i = Client_Samples_i / Total_Samples
   ```
4. **Global Model Update**: Averaged weights become the new global model
5. **Distribution**: Updated global model is available for download by clients

### Queue States
- **Empty (0/10)**: Ready for new updates
- **Partially Full (1-9/10)**: Waiting for more clients
- **Full (10/10)**: FedAvg executes automatically
- **Overflow (>10)**: Triggers additional FedAvg rounds

## 🔧 Configuration

### Server Configuration
- **MAX_CLIENTS**: Maximum clients before auto-execution (default: 10)
- **Port**: Server port (default: 5000)
- **Host**: Server binding (default: 0.0.0.0 for all interfaces)

### Client Configuration
- **Server IP**: IP address of the server machine
- **Port**: Server port (default: 5000)
- **Backup Server**: Fallback server configuration

## 📈 Monitoring

### Server Console
- Real-time connection logs with timestamps
- FedAvg execution details
- Queue status updates
- Error reporting

### Client GUI
- Connection status indicator
- Server status check button
- Progress updates during operations
- Error messages and notifications

### Status Endpoint
Access server status via HTTP:
```bash
curl http://server-ip:5000/status
```

Returns JSON with:
- Queue size and capacity
- Connected client count
- Global model availability
- Queued client IPs

## 🔒 Security Considerations

- **Data Privacy**: Raw data never leaves client machines
- **Model Weights**: Only model weights are shared (not training data)
- **Network Security**: Consider using HTTPS for production deployments
- **Access Control**: Implement authentication for production use

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**:
   - Check if server is running
   - Verify IP address and port
   - Check firewall settings

2. **FedAvg Not Executing**:
   - Check queue size with status command
   - Verify client updates are being received
   - Use manual "execute" command if needed

3. **Model Architecture Mismatch**:
   - Ensure all clients use the same model architecture
   - Check input shape consistency

### Debug Commands
- Server: Use `status` command to check queue
- Client: Use "Check Server Status" button
- Both: Check console logs for detailed information

## 📝 API Reference

### Server Endpoints

- `POST /upload` - Receive client model updates
- `GET /download` - Distribute global model weights
- `GET /status` - Get server status and queue information

### Client Methods

- `send_update_to_server()` - Upload local model to server
- `download_global_model()` - Download latest global model
- `check_server_status()` - Get server queue information
- `train_federated()` - Complete federated training cycle

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Medical imaging community for DICOM standards
- TensorFlow team for deep learning framework
- Federated learning research community 