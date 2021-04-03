import json
import pickle
import argparse
import re
import sys
import string

parser = argparse.ArgumentParser()

parser.add_argument("keyword")

args = parser.parse_args()


def load_trie():
    with open("trie.pickle", "rb") as pckFile:
        my_trie = pickle.load(pckFile)

    return my_trie

def load_json():
    with open("inverted_index.json", "r") as inv_index:
        inv_index = json.load(inv_index)

    return inv_index

def get_possible_words(prefix, inv_index):
    
    def get_matches(token):
        return re.match(prefix, token)

    return list(filter(get_matches, inv_index.keys()))



def process_query(keyword, inv_index, trie):
    if keyword[-1] == "*":
        possible_tokens = get_possible_words(keyword[:-1]+".*", inv_index)

        total_doc_list = []

        for token in possible_tokens:
            isExists, doc_list = trie.search(token)

            if isExists and token in inv_index.keys():

                if doc_list == inv_index[token]:
                    total_doc_list += doc_list
                else:
                    print("ERROR! There is a inconsistency between inverted index and trie.")
                    print("TOKEN: {}".format(token))
                    sys.exit()
            else:
                print("Any document does not contain token: {} or it is a stopword.".format(token))
        
        total_doc_list = list(set(total_doc_list))

        total_doc_list.sort()

        return total_doc_list



    else:
        isExists, doc_list = trie.search(keyword)

        if isExists and keyword in inv_index.keys():

            if doc_list == inv_index[keyword]:
                return doc_list
            else:
                print("ERROR! There is a inconsistency between inverted index and trie.")
                print("KEYWORD: {}".format(keyword))
                sys.exit()
        else:
            return "Any document does not contain keyword: {} or it is a stopword.".format(keyword)




if __name__ == "__main__":
    trie = load_trie()
    inv_index = load_json()

    keyword = args.keyword

    keyword = keyword.lower()
    result =  process_query(keyword, inv_index, trie)

    print(result)