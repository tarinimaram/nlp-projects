import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np
import string
import requests
from numpy import linalg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer

def get_definition(word, ref, key):
    uri = "https://dictionaryapi.com/api/v3/references/" + word + "/json/" + ref + "?key=" + key
    return requests.get(uri)

api_url = https://dictionaryapi.com/api/v3/references/collegiate/json/test?key=961008ea-3343-4593-a642-8a1b2975b362

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
