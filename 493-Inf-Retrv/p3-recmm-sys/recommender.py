from book import book
import math


class recommender:
    """
        A class that is the recommender itself. Vectorizes the description of the
    books using TF-IDF and one hot encodes the genres of the books under L2 Norm.
    """

    def __init__(self, books, df, genres):
        self.books = books
        self.df = df # document frequency 
        self.genres = genres # all the genres as a list


    # Four function below creates the TF-IDF vectors
    def term_freq(self, doc, term):
        return doc.count(term)

    def log_freq_weight(self, term_freq):
        if term_freq > 0:
            return 1 + math.log(term_freq, 10)
        else:
            return 0

    def idf_weight(self, term, lib_size):
        return  math.log(lib_size/self.df[term], 10)

    def tf_idf(self, doc, term, lib_size):
        return self.log_freq_weight(self.term_freq(doc, term)) * self.idf_weight(term, lib_size)

    # L2 Normalizer
    def l2_norm(self, vector):

        sqr_vector = list(map(lambda x: x**2, vector))

        length = math.sqrt(sum(sqr_vector))

        return list(map(lambda x: x/length, vector))


    def process_query_xml(self,book_xml_path):
        """
            A function that processes the query book's xml file and vectorizes it.

        :param book_xml_path: Path to the XML file of the query book.
        :return: Processed query book as class of 'book'
        """
        query_book = book(book_xml_path)

        vector = []
        for term in self.df.keys():
            vector.append(
                    self.tf_idf(
                        query_book.tokens,
                        term,
                        len(self.df.keys())
                    )
                )

        vector = self.l2_norm(vector)

        query_book.vector = vector

        one_hot = [0] * len(self.genres)

        for i, genre in enumerate(self.genres):
            if genre in query_book.genres:
                one_hot[i] = 1

        
        one_hot = self.l2_norm(one_hot)

        query_book.genres_vector = one_hot


        return query_book

    
    



    


        



