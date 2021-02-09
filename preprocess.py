# coding: utf-8
"""
Simple text pre-processing of Universal Declaration of Human Rights for NLP 
using NLTK library.
Created nov 19 2020 by @arena
"""

import re
import contractions
import nltk
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

# importing Universal Declaration of Human Rights from zip
file = open('English-Latin1.txt', 'rt')
document = file.read()
file.close()

print(f'Some lines in the document:\n {document[2060:2860]}...')

# Remove title of each section
doc = re.sub('Article ', '', document)
doc = ''.join(c for c in doc if not c.isdigit())

# Remove new line, and double space
doc = doc.replace('\n', ' ').replace('  ', ' ')
print(f'\nHeaders and New lines removed:\n{doc[:152]}...')

# Remove digits
doc = ''.join(c for c in doc if c not in punctuation)
print('\nRemoved punctuation.')

# Convert to lowercase
doc = doc.lower()
print(f'\nSetting lower case:\n{doc[:152]}... ')

# Expand contractions (can skip since they are removed in the next step)
doc = contractions.fix(doc).replace('  ', ' ')
print('\nExpanded contractions.')

# Tokenize
tokens = word_tokenize(doc)
print(f'\nTokenizing:\n{tokens[:40]}...')

# Remove stop words
stop_words = set(stopwords.words("english"))
tokens = [w for w in tokens if w not in stop_words]
print('\nRemoved stopwords.')

# Lemmatization
wordnet_lem = WordNetLemmatizer()
tokens_lem = [wordnet_lem.lemmatize(token) for token in tokens]

# Stemming (skip as word meaning is lost)
porter_stem = PorterStemmer()
tokens_stem = [porter_stem.stem(token) for token in tokens_lem]

print(f'\nAfter Lemmatization and removing Stop words:\n{tokens_lem[:40]}...')

# Word frequency
fdist = FreqDist(tokens_lem)
print(f'\nMost common words:\n{fdist}')
print(fdist.most_common(20))
fdist.plot(20, cumulative=False)


# END