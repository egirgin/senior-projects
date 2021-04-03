import pickle

class node:

    def __init__(self,  char=None):
        self.char = char
        self.children = []
        self.end_of_word = False
        self.doc_ids = []

    def __repr__(self):
        return str(self.char)

    def __str__(self):
        return str(self.char)

    def __eq__(self, other):
        return str(self.char) == other
    
class trie:

    def __init__(self):
        self.root = node()


    def insert(self, string, doc_id):
        """
            The function inserting a new token into the trie and adding doc_id to its field if the token is already exits.
            This function is called every time a token is processed. This means multiple call for a token but different document ids.
            With every call of this function for a certain token, the document ids are appended to a list, meaning that the list of document ids
            containing this token. 
        """

        current_node = self.root

        for char in string:
            # Since the eq operator is overwritten we can directly check a char int a list of nodes.
            if char in current_node.children:
                current_node = current_node.children[current_node.children.index(char)]
            else:
                current_node.children.append(node(char))
                current_node = current_node.children[current_node.children.index(char)]


        current_node.end_of_word = True

        current_node.doc_ids.append(int(doc_id))
        current_node.doc_ids.sort()

    def search(self, string):

        """
            If a token is already inserted to the trie, then this method returns True and list of document ids of the documents containing that token.
            If token is absent, this function returns False and an empty list.

        """

        current_node = self.root


        for char in string:
            # Since the eq operator is overwritten we can directly check a char int a list of nodes.
            if char in current_node.children:
                current_node = current_node.children[current_node.children.index(char)]
            else:
                # Does not exist in the trie.
                return False, []


        if current_node.end_of_word:
            return True, current_node.doc_ids
        else:
            return False, []


    def save_trie(self):
        with open("trie.pickle", "ab") as pckFile:
            pickle.dump(self, pckFile)

    

