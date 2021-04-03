import os 
import zipfile
import shutil
import string
import json



ints = [str(i) for i in range(0,10)]

def extract_zip(path):
    """
        Gets a path to the dataset.zip file and 
        extracts it to the path where the execution happed

    :@param path: A string to the path of zip file
    """
    
    with zipfile.ZipFile(path, "r") as zip_file:
        output_name = path.split("/")[-1].split(".")[0]
        
        # Clear output path
        try:
            shutil.rmtree("./{}".format(output_name))
        except:
            pass
        # Save folder
        zip_file.extractall("./{}".format(output_name))


    print("Files from {} extracted to {} folder!".format(path.split("/")[-1], output_name))
    
def load_dataset(data_path):
    """
        Gets the path of the data folder which contains 
        two sub-folders with names train and test. Each of 
        those two folder have to contain two separate sub-folders,
        which are named "legitimate" and "spam". Inside those
        folders there are txt files representing the mails.

    :@param data_path: path to the dataset folder
    :@return dataset: a dict consists of the train and test samples 
    for both legitimate and spam.
    """

    train_set = {
        "spam" : [],
        "legitimate" : []
    }

    test_set = {
        "spam" : [],
        "legitimate" : []
    }


    print("Processing Train-Spam...")
    for file in os.listdir("{}/training/spam".format(data_path)):
        with open("{}/training/spam/{}".format(data_path, file), "r", encoding="latin-1") as fileReader:
            train_set["spam"].append(fileReader.read())

    print("Processing Train-Legitimate...")
    for file in os.listdir("{}/training/legitimate".format(data_path)):
        with open("{}/training/legitimate/{}".format(data_path, file), "r", encoding="latin-1") as fileReader:
            train_set["legitimate"].append(fileReader.read())

    print("Processing Test-Spam...")
    for file in os.listdir("{}/test/spam".format(data_path)):
        with open("{}/test/spam/{}".format(data_path, file), "r", encoding="latin-1") as fileReader:
            test_set["spam"].append(fileReader.read())
    
    print("Processing Test-Legitimate...")
    for file in os.listdir("{}/test/legitimate".format(data_path)):
        with open("{}/test/legitimate/{}".format(data_path, file), "r", encoding="latin-1") as fileReader:
            test_set["legitimate"].append(fileReader.read())
    
    print("Done!")

    dataset = {
        "train" : train_set,
        "test" : test_set
    }

    return dataset


def process_dataset(dataset):
    """
        This function takes a dict as dataset and applies the following modifications:
            - Punctuation removal
            - Integer removal
            - Case folding
            - Tokenization
            - Remove digits
            - Remove non-alphanumerics
            - Remove letters (tokens with length one)

        :@param dataset: A dict containing both test and train datasets
        :@return dataset: Preprocesssed dataset

    """

    spam_vocab = []
    legitimate_vocab = []

    for sub_dataset in dataset.keys():
        for clss in dataset[sub_dataset].keys():

            for idx, sample in enumerate(dataset[sub_dataset][clss]):                
                # Remove puncts.
                for punc in string.punctuation:
                    sample = sample.replace(punc, " ")

                for integer in ints:
                    sample = sample.replace(integer, " ")

                # Case folding
                sample = sample.lower()

                # Tokenize
                sample = sample.split()

                # Remove numbers
                sample = [i for i in sample if not i.isdigit()]

                # Remove non alphanumerics
                sample = [i for i in sample if i.isalnum()]

                # Remove letters
                sample = [i for i in sample if len(i)>1]

                dataset[sub_dataset][clss][idx] = sample
                
                if sub_dataset == "train":
                    if clss == "spam":
                        spam_vocab += sample
                    elif clss == "legitimate":
                        legitimate_vocab += sample
                    else:
                        print("ERROR!")


    dataset["train"]["spam_vocab"] = spam_vocab
    dataset["train"]["legitimate_vocab"] = legitimate_vocab
            

    return dataset


def main():
    extract_zip("./dataset.zip")

    dataset = load_dataset("./dataset/dataset")

    dataset = process_dataset(dataset)
    
    # Save dataset to dataset.json file
    with open("dataset.json", "w+") as datasetJSON:
        json.dump(dataset, datasetJSON)

if __name__ == "__main__":
    main()
