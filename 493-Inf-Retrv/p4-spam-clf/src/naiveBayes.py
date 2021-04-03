import os
import time
import math

class NaiveBayes:
    """
        Naive Bayes classifier
    """

    def __init__(self, alpha = 1):
        """
            Alpha is the laplace smoothing variable
        """
        self.alpha = alpha

    def train(self, trainset):
        """
            Calculates the log probs of the tokens in the train dataset 
            and stores them into a dict called model.

        :@param trainset: A dict containing both spam and legitimate docs
        """

        
        print("Training model...")
        start = time.time()

        # Calculate P(c)
        self.prob_spam = len(trainset["spam"]) / ( len(trainset["spam"]) + len(trainset["legitimate"]) )
        self.prob_legit = 1 - self.prob_spam

        # The vocabulary is all the train dataset 
        self.all_words = trainset["spam_vocab"] + trainset["legitimate_vocab"]

        self.model = {
            "spam" : {},
            "legitimate" : {}
        }

        self.spam_denominator = len(trainset["spam_vocab"]) + self.alpha * len(set(self.all_words))  
        self.legit_denominator = len(trainset["legitimate_vocab"]) + self.alpha * len(set(self.all_words))  

        # Calculate p(w|c)
        for idx, word in enumerate(set(self.all_words)):

            self.model["spam"][word] = ( trainset["spam_vocab"].count(word) + self.alpha ) / self.spam_denominator
            self.model["legitimate"][word] = ( trainset["legitimate_vocab"].count(word) + self.alpha ) / self.legit_denominator

        end = time.time()

        print("Training accomplished! Took {:.2f}secs!".format(end - start))
        

    def mi_train(self, trainset, mi):
        """
            The only difference is the dataset is adjusted using the
            mutual information supplied externally by the feature selector.

        :@param trainset: A dict containing both spam and legitimate docs
        :@param mi: A list of words correspond to the features selected using
                    mutual information 
        """

        print("Training Mutual Information model...")
        start = time.time()


        # Create sub dataset using selected features
        spam_features = []
        legit_features = []

        for spam_doc in trainset["spam"]:
            for feature in mi:
                if feature in spam_doc:
                    spam_features.append(feature)


        for legit_doc in trainset["legitimate"]:
            for feature in mi:
                if feature in legit_doc:
                    legit_features.append(feature)
    
        # Calculate P(c)
        self.prob_spam = len(trainset["spam"]) / ( len(trainset["spam"]) + len(trainset["legitimate"]) )
        self.prob_legit = 1 - self.prob_spam

        self.model = {
            "spam" : {},
            "legitimate" : {}
        }


        self.spam_denominator = len(spam_features) + self.alpha * len(set(mi))  
        self.legit_denominator = len(legit_features) + self.alpha * len(set(mi))  

        # Calculate p(w|c)
        for idx, word in enumerate(set(mi)):
            self.model["spam"][word] = ( spam_features.count(word) + self.alpha ) / self.spam_denominator
            self.model["legitimate"][word] = ( legit_features.count(word) + self.alpha ) / self.legit_denominator

        end = time.time()

        print("Training accomplished! Took {:.2f}secs!".format(end - start))





    def test(self, query_doc):
        """
            Takes a query or list of queries and returns the predicted
            class number or list of numbers

        :@param query_doc: Either a list of tokens representing the query doc or
                            list of list of tokens representing the list of query docs.
        :@return preds: Either an integer representing the predicted class id or
                        list of integers representing the list of predicted classes. 
        """

        if type(query_doc[0]) == list:
            multiple_query = True
        elif type(query_doc[0]) == str:
            multiple_query = False
        else:
            print("ERROR")
            return
        
        if multiple_query: # If there are multiple docs in the query

            preds = []

            for query in query_doc:
                spam_prob = math.log(self.prob_spam)
                legit_prob = math.log(self.prob_legit)
                for word in query:

                    # Calculate for spam class
                    if word in self.model["spam"].keys():
                        spam_prob += math.log(self.model["spam"][word])
                    else: # If the query token is not present in the trainingset
                        spam_prob += math.log(1/self.spam_denominator)
                    
                    # Calculate for legitimate class
                    if word in self.model["legitimate"].keys():
                        legit_prob += math.log(self.model["legitimate"][word])
                    else: # If the query token is not present in the trainingset
                        legit_prob += math.log(1/self.legit_denominator)

                preds.append(1 if spam_prob > legit_prob else 0)
            

            return preds


        else: # If only one doc is fed
            spam_prob = math.log(self.prob_spam)
            legit_prob = math.log(self.prob_legit)

            for word in query_doc:
                # Calculate for spam class
                if word in self.model["spam"].keys():
                    spam_prob += math.log(self.model["spam"][word])
                else: # If the query token is not present in the trainingset
                    spam_prob += math.log(1/self.spam_denominator)
                # Calculate for legitimate class
                if word in self.model["legitimate"].keys():
                    legit_prob += math.log(self.model["legitimate"][word])
                else: # If the query token is not present in the trainingset
                    legit_prob += math.log(1/self.legit_denominator)

            return 1 if spam_prob > legit_prob else 0


    def mi_test(self, query_doc):
        """
            The only difference from above is for the tokens that are not present
            in the training set are discarded. The reason is when the mutual information
            is applied only k features are selected. If new seen tokens are calculated
            the evaluation would be done more than k features although the training is 
            done using k of them.

        :@param query_doc: Either a list of tokens representing the query doc or
                            list of list of tokens representing the list of query docs.
        :@return preds: Either an integer representing the predicted class id or
                        list of integers representing the list of predicted classes. 
        """

        if type(query_doc[0]) == list:
            multiple_query = True
        elif type(query_doc[0]) == str:
            multiple_query = False
        else:
            print("ERROR")
            return
        
        if multiple_query:

            preds = []

            for query in query_doc:
                spam_prob = math.log(self.prob_spam)
                legit_prob = math.log(self.prob_legit)
                for word in query:
                     
                    # Calculate for spam class
                    if word in self.model["spam"].keys():
                        spam_prob += math.log(self.model["spam"][word])
                    # Calculate for legitimate class
                    if word in self.model["legitimate"].keys():
                        legit_prob += math.log(self.model["legitimate"][word])

                preds.append(1 if spam_prob > legit_prob else 0)
            

            return preds


        else:
            spam_prob = math.log(self.prob_spam)
            legit_prob = math.log(self.prob_legit)

            for word in query_doc:
                # If the token is not in the trained model then discard it.
                    
                # Calculate for spam class
                if word in self.model["spam"].keys():
                    spam_prob += math.log(self.model["spam"][word])
                # Calculate for legitimate class
                if word in self.model["legitimate"].keys():
                    legit_prob += math.log(self.model["legitimate"][word])

            return 1 if spam_prob > legit_prob else 0