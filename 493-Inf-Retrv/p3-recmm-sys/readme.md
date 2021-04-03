# CmpE 493 - Assignment 3

## Emre Girgin - 2016400099

# File Structure:

There are 6 Python files, 2 text files, one PDF report, and one ReadME file in the project folder.
---
### Python Files:
- _crawler.py_: This script contains methods for crawling the given URLs and saving their content as an XML file.
- _book.py_: This script contains a class called "book". This class takes a path to an XML file and extracts the content in it. Its fields are used while constructing the recommendation system. It has also a function to normalize its description.
- _recommender.py_: This script contains a class called "recommender". It contains all the book objects that are included in the recommendation system. It also takes the query book (the book that will be fed while evaluation) and returns its vectorized version.
- _evaluate.py_: This script contains precision and average precision methods.
- _modelCreator.py_: This script contains a set of functions that processes the books in the training set, stores them, creating the recommender object, and pickling it.
- _recommend.py_: This script takes an argument (a path to the training set URLs or a URL for the query) and either creates the recommendation system or returns the results for the query.

### Text Files:
- _stopword.txt_: This text contains the stopwords that I used while normalizing. The set is adopted from the NLTK.
- _books.txt_: A sample training set file. It is the same as that provided by the instructor. The program is tested with this set several times. It includes 1800 URLs. 
---
# Output Structure:

After calling the appropriate scripts, the following files and folders are created in the working directory.
---

- _/dataset/_: This folder contains the XML files of the training set.
- _/parsedDataset/_: This folder contains a parsed and human-readable version of the XMLs.
- _/queries/_: The XML and the parsed version of the query book.
- _recommender.pickle_: This binary file contains the recommender object created using the training set. It is created after the program is run for the training phase.

# Running

## Training Phase

Sample Running:

``` bash
python recommend.py ./books.txt

```

where books.txt file contains a list of URLs separated with a newline.

During the model creation, the output of the terminal is cleared often. This is done for increasing readability. For example, while fetching the XMLs, the only output on the terminal is the URL of the current book. Also, several timeouts are included, too. Since we are clearing the output, for some outputs, it is intended to give some time to read to the user. 

Sample books file:
``` bash
https://www.goodreads.com/book/show/168214.The_Satyricon
https://www.goodreads.com/book/show/10883.Benjamin_Franklin
https://www.goodreads.com/book/show/691272.Perpetual_Peace
https://www.goodreads.com/book/show/4700482-a-game-of-thrones
https://www.goodreads.com/book/show/7721962-a-visitor-s-guide-to-mystic-falls
https://www.goodreads.com/book/show/297593.Prometheus_Bound
https://www.goodreads.com/book/show/184373.Fascism
https://www.goodreads.com/book/show/12609433-the-power-of-habit
https://www.goodreads.com/book/show/154126.The_Discovery_of_India
https://www.goodreads.com/book/show/17377791-the-land-that-time-forgot
https://www.goodreads.com/book/show/357664.Because_of_Winn_Dixie
https://www.goodreads.com/book/show/763602.Clouds
https://www.goodreads.com/book/show/7740.Gulliver_s_Travels
https://www.goodreads.com/book/show/117251.Hamlet
https://www.goodreads.com/book/show/92307.Being_and_Time
https://www.goodreads.com/book/show/209328.Taking_Rights_Seriously
https://www.goodreads.com/book/show/44652.Fablehaven
...

```

Expected output:

```bash
Recommender model is created. You can predict now!
```


## Evaluation Phase

```bash
python recommend.py https://www.goodreads.com/book/show/4059448-a-lincoln

```
Sample output:
```bash
Precision: 0.3333333333333333
Average Precision: 0.25462633771457305
More detailed info can be found under queries/parsed_106590-Lincoln.txt file.
```


It outputs the precision and average precision scores for that book. A more detailed report for that book can be found in the **./queries/parsed_<book_name>.txt** file
