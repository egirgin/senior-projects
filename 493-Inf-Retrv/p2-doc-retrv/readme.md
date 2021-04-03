# CmpE 493 Assingment 2 - Document Retrieval System

# Credentials : 
## Name & Surname = Emre Girgin
## ID: 2016400099

---
# Setup :
### Python Version: Python3.6

### There is no requirements. Standard Library is sufficient. 
---
# Files : 

### preprocess.py : Decompresses the reuters zip. Reads stories inside it and then creates an inverted index in the form of JSON and a pickle file containing the "trie" data structure for normalized tokens. 

### story.py : A class that helps us to process raw data into normalized tokens.

### trie.py : The node and the trie data structures aiming to conduct the retrieve operations. 

### query.py : Retrieve the list of document ids containing a given token using the inverted index JSON file and trie data structure.

---
# Running : 

### _Make sure that you have "stopwords.txt" and "reuters21578.zip" files in the directory you are running at. Alternatively, you can use the extracted version of the reuters zip as a folder like "./reuters21578/..."._

### 1-) Preprocess the raw data and produce the _inverted_index.json_ and _trie.pickle_.

``` bash 
python preprocess.py
```

### 2-) Search a query:

``` bash
python query.py <keyword>
```

Examples: 
``` bash
python query.py tur*

python query.py right
```

