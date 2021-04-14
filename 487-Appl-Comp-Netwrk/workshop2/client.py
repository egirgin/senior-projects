import socket
import argparse

client_parser = argparse.ArgumentParser()


client_parser.add_argument("target_ip")
client_parser.add_argument("target_port")
client_parser.add_argument("message")

args = client_parser.parse_args()

HOST = str(args.target_ip)
PORT = int(args.target_port)

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(str(args.message + "\n").encode())
except:
    pass