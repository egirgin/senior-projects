import socket
import subprocess


# Find my IP.
def getIP():
    hostname = subprocess.Popen(["hostname", "-I"], stdout=subprocess.PIPE)

    ips = str(hostname.communicate()[0])

    #my_ip = ips.split(" ")[0].split("'")[1] # local

    my_ip = ips.split(" ")[1] # hamachi

    return my_ip


HOST = getIP()  
PORT = 12345      
server_close_flag = False  

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    # keep listening
    while True:
        conn, addr = s.accept()
        
        message = ""

        # Write to packets file
        packetsFile = open("packets.txt", "a+")

        with conn:
            #print('Connected by', addr)
            while True:
                data = conn.recv(1024)

                if data == b'quit':
                    server_close_flag = True
                    print("Closing Server ... ")
                    break
                if not data:
                    break
                else:
                    data = data.decode()
                    message += str(data)

        # Close server if signal has come.
        if server_close_flag:
            packetsFile.close()
            break

        packetsFile.write(message)

        packetsFile.close()

        