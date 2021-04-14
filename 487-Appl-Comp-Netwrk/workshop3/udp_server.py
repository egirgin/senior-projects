import select
import socket
import json
import subprocess
import argparse


udp_server_parser = argparse.ArgumentParser()


udp_server_parser.add_argument("username")

args = udp_server_parser.parse_args()

myUsername = str(args.username)

debug = False

with open("activeUsers.txt", "w+") as f:
        f.write("{ }")
        pass


# Find my IP.
def getIP():
    hostname = subprocess.Popen(["hostname", "-I"], stdout=subprocess.PIPE)

    ips = str(hostname.communicate()[0])

    if debug:
        my_ip = ips.split(" ")[1] # hamachi
    else:
        my_ip = ips.split(" ")[0].split("'")[1] # local

    return my_ip




PORT = 12345

bufferSize = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.bind(('', PORT))

s.setblocking(False)

nameIPs = {}


while True:

    with open("activeUsers.txt", "r+") as activeUsersFile:
        nameIPs = json.load(activeUsersFile)


    result = select.select([s], [], [])

    msg = result[0][0].recv(bufferSize)

    msg = msg.decode()

    msg = json.loads(msg)

    ip = msg["MY_IP"]

    username = msg["NAME"]

    packetType = msg["TYPE"]

    payload = msg["PAYLOAD"]



    if (packetType == "DISCOVER") and (not username in nameIPs.keys()) :

        if username == myUsername:
            continue

        nameIPs[username] = ip

        with open("activeUsers.txt", "w+") as activeUsersFile:
            json.dump(nameIPs, activeUsersFile)

        print("{} just went online!".format(username))


        # Send RESPONSE

        packet = {
            "MY_IP" : getIP(), 
            "NAME" : myUsername,
            "TYPE" : "RESPOND",
            "PAYLOAD" : ""
            }

        packetString = json.dumps(packet, ensure_ascii=False)

        messengerClient = subprocess.Popen(["python3", "tcp_client.py", ip, "12345", packetString])
        print("Responding to {}...".format(username))

        with open("messages/{}.txt".format(username), "a+") as messageFile:
            messageFile.write("First contact!\n")



    elif (packetType == "GOODBYE") and (username in nameIPs.keys()) :

        del nameIPs[username]

        with open("activeUsers.txt", "w+") as activeUsersFile:
            json.dump(nameIPs, activeUsersFile)

        print("{} just went offline!".format(username))

    else:
        pass
        
    
    



