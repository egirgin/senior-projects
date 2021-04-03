import argparse
import os
import re
import math
import pickle
import time
import shutil

from book import book
from recommender import recommender

def read_raws(raw_path):
    with open(raw_path, "r") as raw_file:
        return raw_file.readlines()

#----------------------------------------------

df = {}

def calculate_idf(corpus):
    terms = list(set(corpus))

    for term in terms:
        df[term] = terms.count(term)

#------------------------------------------------

def term_freq(doc, term):
    return doc.count(term)

def log_freq_weight(term_freq):
    if term_freq > 0:
        return 1 + math.log(term_freq, 10)
    else:
        return 0

def idf_weight(term, lib_size):
    return  math.log(lib_size/df[term], 10)

def tf_idf(doc, term, lib_size):
    return log_freq_weight(term_freq(doc, term)) * idf_weight(term, lib_size)

#-----------------------------------------------------

def l2_norm(vector):

    sqr_vector = list(map(lambda x: x**2, vector))

    length = math.sqrt(sum(sqr_vector))

    return list(map(lambda x: x/length, vector)) if length != 0 else ([0] * len(vector))

#-----------------------------------------------------

def create_model():
    try:
        shutil.rmtree("./{}".format("parsedDataset"))
    except Exception as e:
      print(e)
      pass

    try:
        os.mkdir("./{}".format("parsedDataset"))
    except Exception as e:
        print(e)
        pass


    xml_files = os.listdir("./dataset")

    books = []

    corpus = []

    genres = []

    for i, xml_file in enumerate(xml_files):
        current_book = book("./dataset/" + xml_file)

        if current_book.corrupted:
            continue
        os.system("clear")
        print("Parsing: ".format(current_book.title))
        with open("./parsedDataset/{}.txt".format(xml_file), "w+") as parsedFile:
            parsedFile.write("Authors:\n")
            parsedFile.write(str(current_book.authors))
            parsedFile.write("\n")
            parsedFile.write("Description: \n")
            parsedFile.write(current_book.description)
            parsedFile.write("\n")
            parsedFile.write("Genres: \n")
            parsedFile.write(str(current_book.genres))
            parsedFile.write("\n")
            parsedFile.write("Recommendations: \n")
            parsedFile.write(str(current_book.recommendations))
            parsedFile.write("\n")


        corpus += current_book.tokens
        genres += current_book.genres

        books.append(current_book) 

    os.system("clear")
    print("All the books are parsed under 'parsedDataset' folder!")
    genres = list(set(genres))

    calculate_idf(corpus)

    total_terms = df.keys()
    
    print("Vectorizing books!")
    time.sleep(3)
    for current_book in books:
        os.system("clear")
        print(current_book.title)
        vector = []
        for term in total_terms:
            vector.append(tf_idf(current_book.tokens, term, len(total_terms)))

        current_book.vector = l2_norm(vector)

        one_hot = [0] * len(genres)

        for i, genre in enumerate(genres):
            if genre in current_book.genres:
                one_hot[i] = 1

        current_book.genres_vector = l2_norm(one_hot)


    os.system("clear")
    print("Creating Recommender!")
    final_recommender = recommender(books, df, genres)

    with open("recommender.pickle", "wb+") as recomPickle:
        pickle.dump(final_recommender, recomPickle)

    print("Done!")
    time.sleep(1)
    os.system("clear")