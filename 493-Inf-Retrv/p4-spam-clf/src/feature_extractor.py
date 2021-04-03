import math
import os
import sys

class MutualInformation:
    """
        Mutual Information feature extractor
    """

    def __init__(self, dataset, k=100):
        self.k = k 
        self.dataset = dataset # The training set

    
    def _create_table(self, word):
        """
            Create the frequency table from the slides : Lec14 - P30

        :@param word: a token
        :@return table: a list of table cells
        """
        
        n_11 = 0
        n_01 = 0
        n_10 = 0
        n_00 = 0

        for doc in self.dataset["spam"]:
            if word in doc:
                n_11 += 1
            else:
                n_01 += 1

        for doc in self.dataset["legitimate"]:
            if word in doc:
                n_10 += 1
            else:
                n_00 += 1
        # containing spam docs, non-containing spam docs, containing legit docs, non-containing legit docs
        return [n_11, n_01, n_10, n_00]


    def _generate_score(self, table):
        """
            Calculate the score for the given table
        
        :@param table: Table from lecture slides Lec14-P30
        :@return score: A float representing the score for the given table
        """

        #table = [i+1 for i in table ]

        total = sum(table)

        n_11 = table[0]
        n_01 = table[1]
        n_10 = table[2]
        n_00 = table[3]


        term0 = (n_11 / total) * math.log((total * n_11) / ((n_11 + n_10) * (n_11 + n_01)), 2) if n_11 != 0 else 0

        term1 = (n_01 / total) * math.log((total * n_01) / ((n_01 + n_00) * (n_01 + n_11)), 2) if n_01 != 0 else 0

        term2 = (n_10 / total) * math.log((total * n_10) / ((n_10 + n_11) * (n_10 + n_00)), 2) if n_10 != 0 else 0

        term3 = (n_00 / total) * math.log((total * n_00) / ((n_00 + n_01) * (n_00 + n_10)), 2) if n_00 != 0 else 0

        return term0 + term1 + term2 + term3


    def generate(self):
        """
            Calculate the MI score for each token in the training set

        :@return features: Top K elements based on the calculated scores
        """
        
        print("Selecting features...")

        scores = []

        for word in set(self.dataset["spam_vocab"] + self.dataset["legitimate_vocab"]):

            table = self._create_table(word)
            score = self._generate_score(table)

            scores.append((score, word))

        scores = sorted(scores, key = lambda x : x[0], reverse = True)

        features = [word[1] for word in scores[:self.k]]
        
        print("Done!")

        return features

        

