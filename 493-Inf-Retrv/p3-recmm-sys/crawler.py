import urllib.request as request
import os
import shutil

def read_links(path):
    """
       A function that reads the books urls from a file on the given path and stores those links into a list

    :param path: path to the file containing book urls
    :return: list of urls  
    """
    with open(path, "r") as booksFile:
        book_urls = booksFile.read().splitlines()

    return book_urls

def fetch_data(urls):
    """
        A function that fetches the xml file of all the given ulrs

    :param urls: a list of urls
    :return: a list of strings which are decoded xml files
    """
    request_urls = []
    for i, url in enumerate(urls):
        os.system("clear")
        print("Fetching: " +url)
        with request.urlopen(url) as response:
            request_urls.append(response.read().decode("utf-8"))

    os.system("clear")
    return request_urls

def save_raw(raws, book_urls, folder_name):
    """
        A function that saves a list of given xml files

    :param raws: List of strings containing decoded xml files
    :param book_urls: List of strings containing book urls
    :param folder_name: A string that representing the folder name where all the xml's are going to be stored.
    :return: void
    """
    try:
        shutil.rmtree("./{}".format(folder_name))
    except Exception as e:
      print(e)
      pass

    try:
        os.mkdir("./{}".format(folder_name))
    except Exception as e:
        print(e)
        pass

    for i, book in enumerate(book_urls):
        book_name = book.split("/")[-1].replace(".", "-")
        with open(folder_name + "/"+book_name+".txt", "w+") as xml_file:
            xml_file.write(raws[i])
    