import socket
import subprocess
import sys

def run_mac_speech_server():
    # TCP Server Configuration
    host_ip = "0.0.0.0"
    port_number = 5050
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host_ip, port_number))
    server_socket.listen(5)
    
    print("MacOS Speech Bridge Active on port 5050")
    print("Defaulting to 'say' command for synthesis.")
    
    try:
        while True:
            client_conn, addr = server_socket.accept()
            raw_data = client_conn.recv(8192)
            if not raw_data:
                client_conn.close()
                continue
                
            msg = raw_data.decode("utf-8").strip()
            
            # --- Protocol Handling ---
            if msg == "SPD_LIST_VOICES":
                # MacOS provides voices via 'say -v ?'
                result = subprocess.run(["say", "-v", "?"], capture_output=True, text=True)
                client_conn.sendall(result.stdout.encode("utf-8"))
                
            elif msg.startswith("SPD_SET_VOICE:"):
                # On Mac, we handle this by storing the preference or 
                # parsing the voice name. For simplicity, we stick to defaults.
                pass
                
            elif msg.startswith("SPD_RAW:"):
                text_to_say = msg[8:]
                # Call the native 'say' command
                subprocess.Popen(["say", text_to_say])
                
            else:
                # Fallback: Speak the message
                subprocess.Popen(["say", msg])
            
            client_conn.close()
            
    except KeyboardInterrupt:
        print("Shutting down Mac bridge.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    run_mac_speech_server()
