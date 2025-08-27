import socket
import json
import pynput.keyboard

TCP_IP = "127.0.0.1"
TCP_PORT = 5000
BUFFER_SIZE = 4096

server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_sock.bind((TCP_IP, TCP_PORT))
server_sock.listen(1)

print(f"Listening for TCP JSON on {TCP_IP}:{TCP_PORT}...")

keyboard = pynput.keyboard.Controller()
states = [
    lambda: keyboard.press(pynput.keyboard.Key.left),
    lambda: keyboard.release(pynput.keyboard.Key.left),
    lambda: keyboard.press(pynput.keyboard.Key.right),
    lambda: keyboard.release(pynput.keyboard.Key.right)
]
state_index = 0
receipt_count = 0

while True:
    conn, addr = server_sock.accept()
    print(f"Connection from {addr}")
    try:
        while True:
            data = conn.recv(BUFFER_SIZE)
            if not data:
                break
            try:
                json_data = json.loads(data.decode('utf-8'))
                print(f"Received JSON from {addr}: {json_data}")
            except json.JSONDecodeError:
                print(f"Received invalid JSON from {addr}: {data}")

            receipt_count += 1
            if receipt_count % 15 == 0:
                states[state_index]()
                state_index = (state_index + 1) % 4

    finally:
        conn.close()