import re
import string

def get_stopwords():
    """
        A function that reads the stopwords from a file

    :return: a list of stopwords
    """
    with open("./stopwords.txt", "r") as stopwordFile:
        stopwords = stopwordFile.read().splitlines()

    return stopwords

class book:
    """
        A class that represents the books. Crawled properties of book are parsed and stored. 
    """

    def __init__(self, path):
        self.path = path # path to the xml
        # if a file does not contain a the title or recommendation, it is counted as corrupted and discarded (did not included into the recomm. system) 
        self.corrupted = False 
        self.parse_book() # Parses the xml of the book
        self.generate_tokens() # Preprocesses and generates tokens from the description 
        self.vector = [] # Vector representation of the tokens
        self.genres_vector = [] # Vector representation of the genres
        

    def __str__(self):
        return self.title

    def __eq__(self, other):
        return self.title == other.title

    def _parse_bookID(self, raw_id):
        """
            If a book has an id in its xml file, store it.
        :param raw_id: Semi-processed xml snippet
        """
        book_id = re.search('value=".*?"', raw_id, re.DOTALL).group(0)
        
        self.id = book_id[7:-1]

    def _parse_title(self, raw_title):
        """
            A function that parses the title of the book from xml given
        :param raw_title: semi-processed xml snippet
        """
        title = re.search("name\">.*</h1", raw_title, re.DOTALL).group(0)
        title = title[6:-4].strip()

        # If there is a section inside paranthesis, remove it.
        if "(" in title:
            start = title.index("(")
            try:
                end = title.index(")")
            except:
                end = len(title)-1
            title = title[:start] + title[end+1:]

        # Parse Quotation marks
        while "&quot;" in title:
            title = title.replace("&quot;", '"')
            
        title = title.strip()
        
        self.title = title

        # Parse if there is a series
        series = re.search("ries\">.*</h2", raw_title, re.DOTALL).group(0)
        series = series[6:-4].strip()

        if series != "":
            self.series = series
        else:
            self.series = False

    def _parse_authors(self, raw_authors):
        """
            A function that parses the Authors of the book
        :param raw_authors: semi-processes xml snippet
        """

        authors = raw_authors.split("<a class=\"authorName\"")

        parsed_authors= []
        for raw_author in authors[1:]:
            current_author = re.search('"name">.*</span></a>', raw_author, re.DOTALL).group(0)
            current_author = current_author[7:-11]

            # Parse Quotation mark
            while "&quot;" in current_author:
                current_author = current_author.replace("&quot;", '"')
            # Parse apostrophe
            while "&#39;" in current_author:
                current_author = current_author.replace("&#39;", "'")
            # Parse ampersand
            while "amp&;" in current_author:
                current_author = current_author.replace("amp&;", '&')

            parsed_authors.append(current_author)

        self.authors = parsed_authors

    def _parse_desc(self, raw_desc):
        """
            A function that parses the description of the book

        :param raw_desc: semi-processed xml snippet
        """


        try: # full desc
            desc = re.search('"display:none">.*</span', raw_desc, re.DOTALL).group(0)[15:-6]
            desc = desc.replace("<br />", " ")
            desc = desc.replace("<p>", " ")
            desc = desc.replace("</p>", " ")
            desc = desc.replace("<i>", " ")
            desc = desc.replace("</i>", " ")
            desc = desc.replace("<b>", " ")
            desc = desc.replace("</b>", " ")
            desc = desc.replace("<div>", " ")
            desc = desc.replace("</div>", " ")          
        except:
            try: # short desc
                desc = re.search('[0-9]">.*?</span', raw_desc, re.DOTALL).group(0)[1:-6]
                desc = desc.replace("<br />", " ")
                desc = desc.replace("<p>", " ")
                desc = desc.replace("</p>", " ")
                desc = desc.replace("<i>", " ")
                desc = desc.replace("</i>", " ")
                desc = desc.replace("<b>", " ")
                desc = desc.replace("</b>", " ")
                desc = desc.replace("<div>", " ")
                desc = desc.replace("</div>", " ")   
            except:
                desc = ""
        

        self.description = desc

    def _parse_recommendations(self, raw_recommends):
        """
            A function that parses the recommendations of the book

        :param raw_recommends: semi-processed xml snippet
        """

        recomms = re.findall('alt=".*?" src', raw_recommends, re.DOTALL)

        recommended_books = []
        for rec_book in recomms:
            book_name = rec_book[5:-5]
            # Parse apostrophe marks
            while "&#" in book_name:
                start = book_name.index('&#')
                book_name = book_name[:start] + "'" + book_name[start+5:]

            # Remove paranthesis
            if "(" in book_name:
                start = book_name.index("(")
                try:
                    end = book_name.index(")")
                except:
                    end = len(book_name)-1
                book_name = book_name[:start] + book_name[end+1:]

            # Parse Quotation marks
            while "&quot;" in book_name:
                book_name = book_name.replace("&quot;", '"')
            # Parse Ampersand
            while "amp&;" in book_name:
                book_name = book_name.replace("amp&;", '&')

            book_name = book_name.strip()
            recommended_books.append(book_name)

        
        self.recommendations = recommended_books

    def _parse_genres(self, raw_genres):
        """
            A function that parses the genres of the book

        :param raw_genres: semi-processed xml snippet
        """

        try:
            genres = re.findall('bookPageGenreLink.*?</a>', raw_genres, re.DOTALL)

            parsed_genres = []
            for genre in genres:
                if "users" in genre:
                    continue
                parsed_genres.append(re.search(">.*?<", genre, re.DOTALL).group(0)[1:-1])
        except:
            genres = []
        
        self.genres = parsed_genres


    def parse_book(self):
        """
            The general function that triggers the functions above.

        """
        
        with open(self.path, "r") as book_raw_file:
            self.raw_data = book_raw_file.read()

        try:
            raw_id = re.search('<input type="hidden" name="book_id" id="book_id" value=".*?" />', self.raw_data, re.DOTALL).group(0)
            self._parse_bookID(raw_id)
        except:
            pass

        # If a book does not have a title, then it is corrupted
        try:
            raw_title = re.search('id="bookTitle".*id="bookAuthors"', self.raw_data, re.DOTALL).group(0)
            self._parse_title(raw_title)
        except:
            self.corrupted = True
            return 
            
        raw_authors = re.search('<div id="bookAuthors" class="">.*<div id="bookMeta"', self.raw_data, re.DOTALL).group(0)
        self._parse_authors(raw_authors)

        try:
            raw_description = re.search('<div id="description" class="readable stacked" style="right:0">.*<div id=buyButtonContainer', self.raw_data, re.DOTALL).group(0)
            self._parse_desc(raw_description)
        except Exception as e:
            print(self.title + " has problems in the desc!")
            print(e)
            self.description = ""
        
        # If a book does not have the recommendations, then it is corrupted
        try:
            raw_recommendations = re.search("<div class='carouselRow' style='width: 3600px'>.*<div class=\"stacked\">", self.raw_data, re.DOTALL).group(0)
            self._parse_recommendations(raw_recommendations)
        except:
            self.corrupted = True
            return

        try:
            raw_genres = re.search("<div class=\"stacked\">.*<div id=\"aboutAuthor\"", self.raw_data, re.DOTALL).group(0)
            self._parse_genres(raw_genres) 
        except:
            print(self.title + " has problems in the genre!")
            print(e)
            self.genres = []

    def generate_tokens(self):
        """
            A function that generates tokens from raw description
        """
        
        # If a book is corrupted discard it
        if self.corrupted:
            return

        # If the book doesn't have a desc., then pass empty vectors.
        if self.description == "":
            self.tokens = []
            self.desc = []
            return


        desc = self.description


        # Remove punctuations
        for punc in string.punctuation:
            desc = desc.replace(punc, " ")

        # lowercase
        desc = desc.lower()

        desc = desc.split(" ")

        # Remove Stopwords

        stopwords = get_stopwords()

        for word in stopwords:
            if word in desc:
                desc = list(filter(lambda x: x != word, desc))

        desc = list(filter(lambda x: x != "", desc))

        self.tokens = desc

        return desc

