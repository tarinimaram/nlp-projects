from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
import string


# text = input("Enter text: ")

text = "There are many techniques available to generate extractive summarization. To keep it simple, I will be using an unsupervised learning approach to find the sentences similarity and rank them. Summarization can be defined as a task of producing a concise and fluent summary while preserving key information and overall meaning. One benefit of this will be, you don’t need to train and build a model prior start using it for your project. It’s good to understand Cosine similarity to make the best use of the code you are going to see. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them. Its measures cosine of the angle between vectors. The angle will be 0 if sentences are similar."
stopWords = set(stopwords.words("english"))
words = word_tokenize(text)

# improve to count synonyms of a word as the same word when counting frequency 
# remove punctuation 
frequencies = dict()
for word in words: 
    word = word.lower()
    if word in stopWords:
        continue
    elif word in string.punctuation:
        continue
    elif word in frequencies:
        frequencies[word] += 1
    else:
        for syn in wordnet.synsets(word):
            for i in syn.lemmas():
                if i.name() in frequencies:
                    frequencies[i.name()] += 1
                else:
                    frequencies[word] = 1


# sentences that include phrases/transitions to draw attention should score higher 
# ex: "in conclusion, ", "to summarize, ", "it's important to...", "the main point is.."
sentences = sent_tokenize(text)
sentenceScores = dict()
for sentence in sentences:
    for word, freq in frequencies.items():
        if word in sentence.lower():
            if sentence in sentenceScores:
                sentenceScores[sentence] += freq
            else:
                sentenceScores[sentence] = freq  

sumScores = 0
for sentence in sentenceScores:
    sumScores += sentenceScores[sentence]

avgScore = int(sumScores / len(sentenceScores))
#print(avgScore)

# only include the sentences that have a word frequency score 20% above average -> 
# improve to make it 1 std deviation above or a specified percentage by the user (use bell curve standard distribution)
summary = ""
for sentence in sentences:
    if (sentence in sentenceScores) and (sentenceScores[sentence] > (1.2 * avgScore)):
        summary += " " + sentence
print(summary)