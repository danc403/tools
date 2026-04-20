
# pip install comtypes
import socket
import comtypes.client
import sys

def run_speech_server():
    # Initialize SAPI
    try:
        speak_engine = comtypes.client.CreateObject("SAPI.SpVoice")
        voices = speak_engine.GetVoices()
    except Exception as e:
        print("Error initializing SAPI.")
        return

    host_ip = "0.0.0.0"
    port_number = 5050
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host_ip, port_number))
    server_socket.listen(5)
    
    print("SAPI Protocol Controller Active on port 5050")
    
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
                response = ""
                for i in range(voices.Count):
                    response += str(i) + "|" + voices.Item(i).GetDescription() + "\n"
                client_conn.sendall(response.encode("utf-8"))
                
            elif msg.startswith("SPD_SET_VOICE:"):
                try:
                    idx = int(msg.split(":")[1])
                    if 0 <= idx < voices.Count:
                        speak_engine.Voice = voices.Item(idx)
                except: pass
                
            elif msg.startswith("SPD_SET_VOL:"):
                try:
                    vol = int(msg.split(":")[1])
                    speak_engine.Volume = max(0, min(100, vol))
                except: pass
                
            elif msg.startswith("SPD_SET_RATE:"):
                try:
                    rate = int(msg.split(":")[1])
                    speak_engine.Rate = max(-10, min(10, rate))
                except: pass
            
            elif msg.startswith("SPD_RAW:"):
                # Specifically designated text to speak
                actual_text = msg[8:]
                speak_engine.Speak(actual_text, 3)
                
            else:
                # Fallback: Speak the message as-is if no prefix is found
                # This preserves compatibility with simple pipes
                speak_engine.Speak(msg, 3)
            
            client_conn.close()
            
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    run_speech_server()
