import math
import argparse
import pickle
import os
import time

from recommender import recommender
from crawler import * 
from evaluate import *
from modelCreator import *

parser = argparse.ArgumentParser()

parser.add_argument("path")

args = parser.parse_args()


def cos_sim(vector1, vector2):
        
    multiplication = [e1 * e2 for e1,e2 in zip(vector1, vector2)]

    return sum(multiplication)

def similarity(desc_sim, genre_sim, alpha = 0.75):
    # Default alpha rate is 0.75
    return alpha * desc_sim + (1 - alpha) * genre_sim


def rank_by_similarity(query_book, all_books, alpha=0.75):
    """
        A function that returns the a list of tuples (sim_score, book) sorted
    according to descending similarity score.
    :param query_book: the 'book' instance representing the query book
    :param all_books: all the books that are included into the recommendation system
    :param alpha: similarity calculation parameter
    :return: a list of sorted cosine_sim, book pairs for a given query
    """

    sim_vector = []

    for current_book in all_books:
        desc_sim = cos_sim(current_book.vector, query_book.vector)

        genre_sim = cos_sim(current_book.genres_vector, query_book.genres_vector)

        total_sim = similarity(desc_sim, genre_sim, alpha)

        sim_vector.append(total_sim)


    sim_pairs = zip(sim_vector, all_books)

    ranked_sims = sorted(sim_pairs, key=lambda x: x[0], reverse = True)


    return ranked_sims



def main():

    if os.path.exists(args.path):
        print("#####Model Creating Mode is Active#####")

        query_urls = read_links(args.path)

        print("Fetching XMLs")
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        os.system("clear")
        xml_books = fetch_data(query_urls)

        save_raw(xml_books, query_urls, "dataset")
        print("Fetching is completed!")
        print("You can see the raw files under 'dataset' folder!")

        time.sleep(3)
        os.system("clear")

        create_model()
        print("Recommender model is created. You can predict now!")

    else:
        print("#####Recommending Mode is Active#####")


        with open("recommender.pickle", "rb") as recomPickle:
            my_recommender = pickle.load(recomPickle)

        xml_books = fetch_data([args.path])

        try:
            shutil.rmtree("./queries")
        except Exception as e:
            print(e)
            pass

        try:
            os.mkdir("./queries")
        except Exception as e:
            print(e)
            pass

        book_name = args.path.split("/")[-1].replace(".", "-")
        with open("queries/"+book_name+".txt", "w+") as xml_file:
            xml_file.write(xml_books[0])

        current_book = my_recommender.process_query_xml("queries/"+book_name+".txt")

        ranks = rank_by_similarity(current_book, my_recommender.books, alpha=0.7)
        
        predictions = ranks[:18]

        predictions = list(map(lambda x: x[1].title, predictions))

        groundTruths = current_book.recommendations

        current_precision = precision(groundTruths, predictions)

        current_average_precision = average_precision(groundTruths, predictions)

        with open("queries/parsed_" + book_name + ".txt", "w+") as queryFile:
            queryFile.write("Authors:\n")
            queryFile.write(str(current_book.authors))
            queryFile.write("\n")
            queryFile.write("Description: \n")
            queryFile.write(current_book.description)
            queryFile.write("\n")
            queryFile.write("Genres: \n")
            queryFile.write(str(current_book.genres))
            queryFile.write("\n")
            queryFile.write("Recommendations: \n")
            queryFile.write(str(current_book.recommendations))
            queryFile.write("\n")
            queryFile.write("Predictions: \n")
            queryFile.write(str(predictions))
            queryFile.write("\n")
            queryFile.write("Precision: \n")
            queryFile.write(str(current_precision))
            queryFile.write("\n")
            queryFile.write("Average Precision: \n")
            queryFile.write(str(current_average_precision))
            queryFile.write("\n")

        print("Precision: " + str(current_precision))
        print("Average Precision: " + str(current_average_precision))
        print("More detailed info can be found under queries/parsed_{}.txt file.".format(book_name))


if __name__=="__main__": 
    main() 