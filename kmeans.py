import numpy as np
import pandas as pd
from docx import Document
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances


agri = r'books\agriculture.docx'
antro = r'books\antropology.docx'
bioch = r'books\biochemistry.docx'
cle = r'books\cleancode.docx'
cook = r'books\Cooking.docx'
crim = r'books\criminal.docx'
glos = r'books\glossary.docx'
hist = r'books\history.docx'
psy = r'books\psychology.docx'
zoo = r'books\zoology.docx'
stop_words = set(stopwords.words('english'))

def getText(filename):
    doc = Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def listToString(list):
    str1 = ' '
    return (str1.join(list))

def remove_stop_words(res):
    word_tokens = word_tokenize(res)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
  
    filtered_sentence = []
  
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return listToString(filtered_sentence)

def remove_punc(words):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~1234567890'''

    for ele in words:
        if ele in punc:
            words = words.replace(ele, "")
    return remove_stop_words(words.lower())



d1 = remove_punc(getText(agri))
d2 = remove_punc(getText(antro))
d3 = remove_punc(getText(bioch))
d4 = remove_punc(getText(cle))
d5 = remove_punc(getText(cook))
d6 = remove_punc(getText(crim))
d7 = remove_punc(getText(glos))
d8 = remove_punc(getText(hist))
d9 = remove_punc(getText(psy))
d10 = remove_punc(getText(zoo))
# print(d1)

all_docs = [d1, d2, d3, d4]
# print(all_docs)



vector = TfidfVectorizer()

result = vector.fit_transform(all_docs)


# blobs = pd.DataFrame.sparse.from_spmatrix(result)


# print('\nWord indexes:')
# print(vector.vocabulary_)
# print('\ntf-idf value:')
# print(result.toarray())

kmeans = KMeans(
    init="k-means++",
    n_clusters=4,
    n_init=5,
    max_iter=300)

kmeans.fit(result)
print(kmeans.n_iter_)
# lines_for_predicting = ["Man ancient Indian religious philosophical teachings"]
# print(kmeans.predict(vector.transform(lines_for_predicting)))
# # in matrix form
# print('\ntf-idf values in matrix form:')
# print(result.toarray())


# print(result)
for obj in result:
    print(euclidean_distances(result[0], obj))
