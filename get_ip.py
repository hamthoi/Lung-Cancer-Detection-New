import socket
import requests

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def get_public_ip():
    """Get the public IP address of this machine"""
    try:
        response = requests.get("https://api.ipify.org", timeout=5)
        return response.text
    except Exception:
        return "Could not determine public IP"

if __name__ == "__main__":
    print("=" * 50)
    print("NETWORK CONFIGURATION FOR FEDERATED LEARNING")
    print("=" * 50)
    
    local_ip = get_local_ip()
    public_ip = get_public_ip()
    
    print(f"\nLocal IP Address: {local_ip}")
    print(f"Public IP Address: {public_ip}")
    
    print("\n" + "=" * 50)
    print("SETUP INSTRUCTIONS:")
    print("=" * 50)
    print("1. SERVER MACHINE:")
    print(f"   - Run: python server.py")
    print(f"   - Server will be accessible at: {local_ip}:5000")
    print("   - Other machines on the same network can connect to this IP")
    
    print("\n2. CLIENT MACHINES:")
    print("   - Run: python client.py or python client1.py")
    print(f"   - Enter server IP: {local_ip}")
    print("   - Enter server port: 5000")
    print("   - Click 'Connect' to test connection")
    
    print("\n3. NETWORK REQUIREMENTS:")
    print("   - All machines must be on the same local network")
    print("   - Firewall must allow connections on port 5000")
    print("   - Server machine's IP address must be accessible")
    
    print("\n4. TROUBLESHOOTING:")
    print("   - If connection fails, check firewall settings")
    print("   - Ensure all machines are on the same network")
    print("   - Try using the local IP address shown above")
    
    print("\n" + "=" * 50) 