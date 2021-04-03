import string

class Story:

    token_freq = {}

    def __init__(self, story_id, story_title, story_body):
        self.id = int(story_id)
        self.title = story_title
        self.body = story_body


    def __str__(self):
        return "ID: {}\n{}\n{}".format(self.id, self.title, self.body)

        
    def tokenize(self):
        title_tokens = self.title.split()
        body_tokens = self.body.split()

        raw_tokens = title_tokens + body_tokens

        return raw_tokens

    def remove_punctuations(self):

        punc_free_tokens = []
        for token in self.tokenize():
            for punc in string.punctuation:
                token = token.strip().replace(punc, " ")

            punc_free_tokens += token.split()
        
        return punc_free_tokens

    
    def lowercase(self):

        def lower(string):
            return string.lower()


        return list(map(lower, self.remove_punctuations()))


    def normalize(self):

        def remove_newline(string):
            return string.strip()


        with open("stopwords.txt", "r") as stopword_file:
            stopwords = stopword_file.readlines()
            stopwords = list(map(remove_newline, stopwords))

        return list(set([token for token in self.lowercase() if token not in stopwords]))




