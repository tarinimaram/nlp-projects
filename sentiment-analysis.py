import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import io
import unicodedata
import numpy as np
import re
import string
from numpy import linalg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import webtext, stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

with open('kindle.txt', encoding = 'ISO-8859-2') as f:
    textRaw = f.read()

stopWords = set(stopwords.words("english"))
text = textRaw.lower()
words = word_tokenize(text)
words = nltk.pos_tag(words, tagset="universal")
#print(words)

# sent_tokenizer = PunktSentenceTokenizer(text)
# sentences = sent_tokenizer.tokenize(text)

# print(word_tokenize(text))
# print(sent_tokenize(text))

# porter_stemmer = PorterStemmer()

# nltk_tokens = nltk.word_tokenize(text)

# for w in nltk_tokens:
#     print ("Actual: %s Stem: %s" % (w, porter_stemmer.stem(w)))

wordnet_lemmatizer = WordNetLemmatizer()
lemmatizedWords = []
#nltk_tokens = nltk.word_tokenize(text)

for word in words:
    #word = word.lower()
    if word[0] in stopWords:
        continue
    elif word[0] in string.punctuation:
        continue
    else:
        if (word[1] == "VERB"):
            lemmatizedWords.append(wordnet_lemmatizer.lemmatize(word[0], 'v'))
        elif (word[1] == "NOUN"):
            lemmatizedWords.append(wordnet_lemmatizer.lemmatize(word[0], 'n'))
        elif (word[1] == "ADJ"):
            lemmatizedWords.append(wordnet_lemmatizer.lemmatize(word[0], 'a'))
        elif (word[1] == "ADV"):
            lemmatizedWords.append(wordnet_lemmatizer.lemmatize(word[0], 'r'))
        else:
            lemmatizedWords.append(wordnet_lemmatizer.lemmatize(word[0]))
#print(lemmatizedWords)

# for w in nltk_tokens:
#     print("Actual: %s Lemma: %s" % (w, wordnet_lemmatizer.lemmatize(w)))

# text = nltk.word_tokenize(text)
# print(nltk.pos_tag(text))
processedText = ""
for word in lemmatizedWords:
    processedText += word + " "
print(processedText)
sid = SentimentIntensityAnalyzer()
scores = sid.polarity_scores(processedText)
for key in sorted(scores):
    print('{0}: {1}, '.format(key, scores[key]), end ='')

print("\n")
# without preprocessing 
scoresRaw = sid.polarity_scores(textRaw)
print(textRaw)
for key in sorted(scoresRaw):
    print('{0}: {1}, '.format(key, scoresRaw[key]), end ='')

# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# with open('kindle.txt', encoding='ISO-8859-2') as f:
#     for text in f.read().split('\n'):
#         print(text)
#         scores = sid.polarity_scores(text)
#         for key in sorted(scores):
#             print('{0}: {1}, '.format(key, scores[key]), end ='')

#     print()