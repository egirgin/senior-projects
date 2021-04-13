import os
import matplotlib.pyplot as plt
import os
import argparse
import shutil

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-n", "--name", type=str)

args = arg_parser.parse_args()


def read_logs(path):

    with open(path, "r") as logFile:
        logs = logFile.read().splitlines()

    test_accs = [] 

    for line in logs:
        if "TestAccuracy" in line:
            test_accs.append(float(line.split()[1][:5]))

    
    return test_accs

def save_list(accs_list, name):

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")

    x_axis = [i*100 for i in range(1,len(accs_list[0][1])+1)]

    for acc in accs_list:
        plt.plot(x_axis, acc[1], label=acc[0])
        
    plt.legend() 

    plt.savefig("./{}/{}.png".format(name, name))


def main():
    try:
        shutil.rmtree(args.name)
    except:
        pass

    os.mkdir(args.name)


    files = os.listdir(".")

    logs = []

    for file in files:
        if ".txt" in file:
            num_class = file.split("_")[0][1]
            num_sample = file.split("_")[1][1]
            batch_size = file.split("_")[2][1]
            learning_rate = file.split("_")[3].split(".tx")[0]
            logs.append(("N:{}/K:{}/LR:{}".format(num_class, num_sample, batch_size, learning_rate), read_logs(file)))

            os.rename(file, "./{}/{}".format(args.name, file))

   
    


    save_list(logs, args.name)

if __name__=="__main__":
    main()