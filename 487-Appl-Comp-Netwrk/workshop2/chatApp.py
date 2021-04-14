import subprocess
import json
import os
import enum
import sys
from shutil import copyfile, rmtree

myUsername = "defaultUsername"

class packetTypes(enum.Enum):
    DISCOVER = 1
    RESPOND = 2 
    MESSAGE = 3

nameIP = {}

# Save all the IPs found to the disk.
def saveNameIP():
    with open("nameIPPairs.txt", "w+") as pairFile:
        json.dump(nameIP, pairFile)
        pairFile.write("\n")

# Print all the saved users.
def listUsers():
    for i, name in enumerate(nameIP):
        print(str(i+1) + "-) " + name)

# Restore users from disk to dict.
def restoreUsers():
    with open("nameIPPairs.txt", "r") as pairFile:
        pairs = json.load(pairFile)

    return pairs
        
# Find my IP.
def getIP():
    hostname = subprocess.Popen(["hostname", "-I"], stdout=subprocess.PIPE)

    ips = str(hostname.communicate()[0])

    #my_ip = ips.split(" ")[0].split("'")[1] # local

    my_ip = ips.split(" ")[1] # hamachi

    """

    deviceName = "ham0"

    ifconfig = subprocess.Popen(["ifconfig", deviceName], stdout=subprocess.PIPE)

    wlp3s0 = str(ifconfig.communicate()[0])

    wlp3s0_tokens = wlp3s0.split(" ")

    inetIndex = wlp3s0_tokens.index("inet")

    my_ip = wlp3s0_tokens[inetIndex+1]
    """

    return my_ip

# Send Discover signal from 0 to 254
def discoverNetwork():

    myIP = getIP()

    mask = ".".join(myIP.split(".")[:3])

    """
    allIPs = []
    
    for i in range(0, 255):
        for j in range(0,255):
            for q in range(0,255):
                allIPs.append("25." + str(i) + "." + str(j) + "." + str(q))
    """
    #allIPs = [mask + "." + str(i) for i in range(0,255)]

    allIPs = ["25.31.220.112"]

    discoverPacket = {
    "MY_IP" : myIP,
    "NAME" : myUsername,
    "TYPE" : "DISCOVER",
    "PAYLOAD" : ""
    }

    discoverJSON = json.dumps(discoverPacket, ensure_ascii=False)

    #discoverProcess = subprocess.Popen(["echo", discoverJSON], stdout=subprocess.PIPE)

    #discoverProcess.wait()

    for ip in allIPs:
        discoverClient = subprocess.Popen(["python3", "client.py", ip, "12345", discoverJSON])
    
    return

# Run netcat server
def runServer():

    #packetsFile = open("packets.txt", "a+")

    #server = subprocess.Popen(["nc", "-l", "-k", "12345"], stdout=packetsFile)

    initPackets = open("packets.txt", "w+")
    initPackets.close()

    server = subprocess.Popen(["python3", "server.py"])

    return server

def closeServer():
    closeProcess = subprocess.Popen(["python3", "client.py", getIP(), "12345", "quit"])
    closeProcess.wait()
    closeProcess.kill()


# Parse all the packets to the disk.
def parsePackets():

    with open("packets.txt", "r") as packetFile:
        packets = packetFile.readlines()

        for packet in packets:
            
            packetDict = json.loads(packet)

            if packetDict["TYPE"] == "DISCOVER":
                nameIP[packetDict["NAME"]] = packetDict["MY_IP"]
                sendMessage(packetDict["NAME"], "", packetTypes.RESPOND)
                saveNameIP()
                print("NEW USER!!! {} just went online!".format(packetDict["NAME"]))

            elif packetDict["TYPE"] == "RESPOND":
                nameIP[packetDict["NAME"]] = packetDict["MY_IP"]
                with open("messages/{}.txt".format(packetDict["NAME"]), "a+") as messageFile:
                    messageFile.write("First contact!\n")
                saveNameIP()
                print("NEW USER!!! {} just went online!".format(packetDict["NAME"]))
    
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

    if not targetName in nameIP.keys():
        print("Unknown username!")
        return

    myIP = getIP()

    if packetType == packetTypes.RESPOND:
        packet = {
            "MY_IP" : myIP, 
            "NAME" : myUsername,
            "TYPE" : "RESPOND",
            "PAYLOAD" : ""
            }
    elif packetType == packetTypes.MESSAGE:
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
    targetIP = nameIP[targetName]

    # Send packet
    #messageClient = subprocess.Popen(["echo", packetString], stdout=subprocess.PIPE)
    #messageClient.wait()

    #messengerClient = subprocess.Popen(["nc", "-N", targetIP, "12345"], stdin = messageClient.stdout)
    
    messengerClient = subprocess.Popen(["python3", "client.py", targetIP, "12345", packetString])
    print("Sending...")
    
    #messengerClient.wait()

    if packetType == packetTypes.MESSAGE:
        with open("messages/{}.txt".format(targetName), "a+") as messageFile:
            messageFile.write("You : " + packet["PAYLOAD"] + "\n")
    elif packetType == packetTypes.RESPOND:
        with open("messages/{}.txt".format(targetName), "a+") as messageFile:
            messageFile.write("First contact!\n")

    return


def clearLogs():
    if os.path.exists("nameIPPairs.txt"):
        os.remove("nameIPPairs.txt")
    if os.path.exists("packets.txt"):
        os.remove("packets.txt")    
    if os.path.exists("messages/"):
        rmtree("messages")

if __name__ == "__main__":

    clearLogs()

    myUsername = input("Please type in your username: ")

    #nameIP[myUsername] = getIP()

    if not os.path.exists("messages"):
        os.mkdir("messages")
    
    print("Let me discover Network")
    discoverNetwork()
    print("Done!")

    server = runServer()    
    
    try:
        nameIP = restoreUsers()
    except:
        saveNameIP()

    try:
        while(True):
            print("Getting ready...")
            parsePackets()

            while(True):
                parsePackets()
                listUsers()

                print("Type 'quit' to close app or 'scan' to refresh!")

                username = input("The user you want to chat with or type 'scan' or 'quit': ")

                if username == "scan":
                    os.system("clear")
                    print("Scanning...")
                    continue

                if username == "quit":
                    parsePackets()
                    print("Bye!")
                    closeServer()
                    server.kill()
                    sys.exit()

                try:
                    with open("messages/{}.txt".format(username), "r+") as privateMessages:
                        for line in privateMessages.readlines():
                            print(line)
                    break
                except:
                    os.system("clear")
                    print("Unknown username! Please enter a valid username!")
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
        closeServer()
        server.kill()
    