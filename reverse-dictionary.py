import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np
import string
from numpy import linalg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

with open('definition.txt', encoding = 'ISO-8859-2') as f:
    textRaw = f.read()

stopWords = set(stopwords.words("english"))
text = textRaw.lower()
words = word_tokenize(text)
words = nltk.pos_tag(words, tagset="universal")

wordnet_lemmatizer = WordNetLemmatizer()
lemmatizedWords = []

for word in words:
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

synonyms = lemmatizedWords

for word in lemmatizedWords: 
    for syn in wordnet.synsets(word):
        for i in syn.lemmas():
            synonyms.append(i.name())