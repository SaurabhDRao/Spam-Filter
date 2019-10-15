from getStopWords import getStopWords
import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

stopWords = getStopWords()

def preprocess(sentence):
    removeChars = '([0-9]+)|(\n)|[!"#$%&\'*+,-./:;<=>?@^_`~]'
    sentence = re.sub(removeChars, "", sentence)
    sentence = re.sub('[()[]{|}]', " ", sentence)
    sentence = sentence.lower()
    words = [w for w in sentence.split() if (w not in stopWords)]
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def bagOfWords(sentenceList):
    bow = dict()

    for sentence in sentenceList:
        words = sentence.split()
        for word in words:
            if(word in bow):
                bow[word] += 1
            else:
                bow[word] = 1
    return bow