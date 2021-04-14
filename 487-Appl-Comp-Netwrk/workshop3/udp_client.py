import socket
import argparse
import json

udp_client_parser = argparse.ArgumentParser()


udp_client_parser.add_argument("packet_type")
udp_client_parser.add_argument("my_ip")
udp_client_parser.add_argument("my_username")


args = udp_client_parser.parse_args()

debug = False

discoverPacket = {
    "MY_IP" : str(args.my_ip),
    "NAME" : str(args.my_username),
    "TYPE" : "DISCOVER",
    "PAYLOAD" : ""
    }

goodbyePacket = {
    "MY_IP" : str(args.my_ip),
    "NAME" : str(args.my_username),
    "TYPE" : "GOODBYE",
    "PAYLOAD" : ""
    }

discoverJSON = json.dumps(discoverPacket, ensure_ascii=False)

goodbyeJSON = json.dumps(goodbyePacket, ensure_ascii=False)

PORT = 12345

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind(('', 0))

sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

if str(args.packet_type) == "discover":
    for _ in range(3):
        if debug:
            sock.sendto((discoverJSON + "\n").encode(), ('25.255.255.255', PORT))
        else:
            sock.sendto((discoverJSON + "\n").encode(), ('<broadcast>', PORT))
elif str(args.packet_type) == "goodbye":
    for _ in range(3):
        if debug:
            sock.sendto((goodbyeJSON + "\n").encode(), ('25.255.255.255', PORT))
        else:
            sock.sendto((goodbyeJSON + "\n").encode(), ('<broadcast>', PORT))    
else:
    print("Unknown BROADCAST type!")