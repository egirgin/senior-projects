import subprocess
import json
import os
import enum
import sys
from shutil import copyfile, rmtree

myUsername = "defaultUsername"
myIP = "0.0.0.0"

debug = False

class packetTypes(enum.Enum):
    DISCOVER = 1
    RESPOND = 2 
    MESSAGE = 3

def getActiveUsers():
    with open("activeUsers.txt", "r+") as activeUsersFile:
        nameIPs = json.load(activeUsersFile)
    
    return nameIPs

def listUsers():
    with open("activeUsers.txt", "r+") as activeUsersFile:
        nameIPs = json.load(activeUsersFile)
    
        print("Currently active users:")

        for i, user in enumerate(nameIPs.keys()):
            print("{}-) {}".format(i+1, user))
   

# Find my IP.
def getIP():
    hostname = subprocess.Popen(["hostname", "-I"], stdout=subprocess.PIPE)

    ips = str(hostname.communicate()[0])

    if debug:
        my_ip = ips.split(" ")[1] # hamachi
    else:
        my_ip = ips.split(" ")[0].split("'")[1] # local


    return my_ip

# Send Discover signal from 0 to 254
def discoverNetwork(myUsername):

    myIP = getIP()

    udp_client = subprocess.Popen(["python3", "udp_client.py", "discover", myIP, myUsername])

    return udp_client

# Run netcat server
def runServer():

    with open("activeUsers.txt", "w+") as f:
        f.write("{ }")
        pass

    with open("packets.txt", "w+") as f:
        pass

    tcp_server = subprocess.Popen(["python3", "tcp_server.py"])

    udp_server = subprocess.Popen(["python3", "udp_server.py", myUsername])

    return tcp_server, udp_server

def closeServer():
    closeProcess = subprocess.Popen(["python3", "tcp_client.py", getIP(), "12345", "quit"])
    closeProcess.wait()
    closeProcess.kill()


# Parse all the packets to the disk.
def parsePackets():

    with open("packets.txt", "r") as packetFile:
        packets = packetFile.readlines()

        for packet in packets:
            
            packetDict = json.loads(packet)

            if packetDict["TYPE"] == "RESPOND":
                with open("messages/{}.txt".format(packetDict["NAME"]), "a+") as messageFile:
                    messageFile.write("First contact!\n")    

                # may be a problem due to concurrent write to a single file
                userlist = getActiveUsers()

                userlist[packetDict["NAME"]] = packetDict["MY_IP"]

                with open("activeUsers.txt", "w+") as activeUsersFile:
                    json.dump(userlist, activeUsersFile)
                
            elif packetDict["TYPE"] == "MESSAGE":
                with open("messages/{}.txt".format(packetDict["NAME"]), "a+") as messagesFile:
                    messagesFile.write(packetDict["NAME"] + " : " + packetDict["PAYLOAD"] + "\n" )
            else:
                print("Packet Read Type Error!")

    # Clear packets.
    with open("packets.txt", "w+") as clearFile:
        pass

    return 

# Either send message or respond.
def sendMessage(targetName, message, packetType):

    if not targetName in getActiveUsers().keys():
        print("Unknown username!")
        return

    myIP = getIP()

    if packetType == packetTypes.MESSAGE:
        packet = {
            "MY_IP" : myIP, 
            "NAME" : myUsername,
            "TYPE" : "MESSAGE",
            "PAYLOAD" : message
            }
    else:
        print("Unknown Message Type!")
        return

    packetString = json.dumps(packet, ensure_ascii=False)

    # Get IP of that username
    targetIP = getActiveUsers()[targetName]
    
    messengerClient = subprocess.Popen(["python3", "tcp_client.py", targetIP, "12345", packetString])
    print("Sending...")
    

    if packetType == packetTypes.MESSAGE:
        with open("messages/{}.txt".format(targetName), "a+") as messageFile:
            messageFile.write("You : " + packet["PAYLOAD"] + "\n")

    return


def clearLogs():
    if os.path.exists("packets.txt"):
        os.remove("packets.txt")    
    if os.path.exists("messages/"):
        rmtree("messages")

if __name__ == "__main__":

    clearLogs()

    myUsername = input("Please type in your username: ")

    if not os.path.exists("messages"):
        os.mkdir("messages")

    tcp_server, udp_server = runServer() 
    
    print("Discovering network...")
    udp_client = discoverNetwork(myUsername)
    print("Done!")    

    try:
        while(True):
            parsePackets()

            while(True):
                parsePackets()
                listUsers()

                print("Type 'quit' to close app or 'scan' to refresh!")

                username = input("Target username: ")

                if username == "scan":
                    os.system("clear")
                    print("Scanning...")
                    continue

                if username == "quit":
                    parsePackets()
                    print("Closing...")
                    udp_client = subprocess.Popen(["python3", "udp_client.py", "goodbye", getIP(), myUsername])
                    udp_client.wait()
                    udp_client.kill()
                    print("Bye!")
                    closeServer()
                    tcp_server.kill()
                    udp_server.kill()
                    sys.exit()

                if username == myUsername:
                    os.clear()
                    print("You can not chat with yourself!")
                    continue

                try:
                    with open("messages/{}.txt".format(username), "r+") as privateMessages:
                        for line in privateMessages.readlines():
                            print(line)
                    break
                except:
                    os.system("clear")
                    print("Unknown or offline user {}! Please enter a valid username!".format(username))
                    continue

            while(True):
                print("Type 'quit' to exit conversation or 'scan' to refresh to see if there is new messages!")
                message = input("Your message to {}: ".format(username))

                if message == "quit":
                    parsePackets()
                    break
                elif message == "scan":
                    print("Scanning...")
                    parsePackets()
                    os.system("clear")
                    with open("messages/{}.txt".format(username), "r+") as privateMessages:
                        for line in privateMessages.readlines():
                            print(line)
                    continue

                else:
                    sendMessage(username, message, packetTypes.MESSAGE)
                    parsePackets()
                    os.system("clear")
                    with open("messages/{}.txt".format(username), "r+") as privateMessages:
                        for line in privateMessages.readlines():
                            print(line)


    except():
        parsePackets()
        print("Closing...")
        udp_client = subprocess.Popen(["python3", "udp_client.py", "goodbye", getIP(), myUsername])
        udp_client.wait()
        udp_client.kill()
        print("Bye!")
        closeServer()
        tcp_server.kill()
        udp_server.kill()
        sys.exit()
    