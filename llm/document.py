import re
import nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Document:
    def __init__(self, document: str) -> None:
        self.document = document
        self.tokens = ""
        self.vocab = {}
    
    def parse(self) -> None:
        document = re.sub(r'\s+', ' ', self.document)
        document = document.strip()
        self.document = document

    def tokenize(self) -> None:
        tokens = word_tokenize(self.document)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        self.tokens = tokens
        self.vocab.update(self.tokens)
    
    def preprocess(self) -> None:
        self.parse()
        self.tokenize()
    
