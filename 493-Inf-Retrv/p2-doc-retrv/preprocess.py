import os
import json
import zipfile
from story import Story
from trie import node, trie

stories = []


def extract_reuters():

    with zipfile.ZipFile("reuters21578.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

def create_stories(reuters_path):

    sgm_list = os.listdir(reuters_path)

    sgm_list.remove("lewis.dtd")
    sgm_list.remove("README.txt")
    sgm_list.remove(".DS_Store")


    sgm_files = []
    article_ids_wo_body = []

    for sgm_file in sgm_list:
        with open(reuters_path+"/"+sgm_file, "r", encoding="latin-1") as sgm:
            current_sgm = sgm.read()
            current_sgm = current_sgm.split("</REUTERS>")
            sgm_files.append(current_sgm)

    for sgm_file in sgm_files: 
        for article in sgm_file:
            if article.isspace():
                continue

            article_date = article.split("<DATE>")

            """
                sample article_date[0] : <REUTERS TOPICS="YES" LEWISSPLIT="TRAIN" CGISPLIT="TRAINING-SET" OLDID="8914" NEWID="4001">

            """
   
            article_id = article_date[0].split(" ")[-1].split("\"")[1]
            
            is_title_exists = False
            is_body_exists = False

            if "<TITLE>" in article_date[1]:
                is_title_exists = True
            if "<BODY>" in article_date[1]:
                is_body_exists = True

            if not is_body_exists and not is_title_exists:
                article_ids_wo_body.append(article_id)
                #stories.append(Story(article_id, "", ""))
                continue

            article_title_split = article_date[1].split("</TITLE>")

            """
                sample article_title_spilt[0] : <TEXT>&#2; <TITLE>INCO SEES NO MAJOR IMPACT FROM DOW REMOVAL
            
            """

            
            article_title = article_title_split[0].split("<TITLE>")[-1]

            if is_body_exists:
                article_body = article_title_split[1].split("</BODY>")[0].split("<BODY>")[-1]
            else:
                article_ids_wo_body.append(article_id)
                article_body = ""

            stories.append(Story(int(article_id), article_title, article_body))


    return stories, article_ids_wo_body


def create_inverted_index(stories):
    
    inverted_index = {}

    for i, story in enumerate(stories):
        for token in story.normalize():
            if token in inverted_index.keys():
                inverted_index[token] += [story.id]
            else:
                inverted_index[token] = [story.id]

            inverted_index[token].sort()

        if i %1000 == 0:
            print("%{}".format(int((i*100)/len(stories))))

    
    with open("inverted_index.json", "w+") as inv_index:
        json.dump(inverted_index, inv_index)

    return inverted_index


def create_trie(stroies):

    my_trie = trie()

    for i, story in enumerate(stories):
        for token in story.normalize():
            my_trie.insert(token, story.id)

        if i %1000 == 0:
            print("%{}".format(int((i*100)/len(stories))))



    return my_trie

if __name__ == "__main__":
    if "reuters21578.zip" in os.listdir("."):
        extract_reuters()
    stories, _ = create_stories("./reuters21578")

    print("Creating inverted index...")
    inv_index = create_inverted_index(stories)
    print("Completed!")

    print("Creating trie...")
    my_trie = create_trie(stories)
    my_trie.save_trie()
    print("Completed!")
    