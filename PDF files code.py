import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfminer.high_level
import nltk
import numpy as np

# Convert PDF files to plain text
text1 = pdfminer.high_level.extract_text("p.pdf")
text2 = pdfminer.high_level.extract_text("p.pdf")

# Preprocess the text
stopwords = nltk.corpus.stopwords.words("english")
text1 = [word for word in nltk.word_tokenize(text1.lower()) if word.isalpha() and word not in stopwords]
text2 = [word for word in nltk.word_tokenize(text2.lower()) if word.isalpha() and word not in stopwords]

# Load a pre-trained Word2Vec model
model = api.load("word2vec-google-news-300")

# Compute the average vector for each document
vec1 = np.mean([model[word] for word in text1 if word in model], axis=0)
vec2 = np.mean([model[word] for word in text2 if word in model], axis=0)

# Calculate the cosine similarity between Word2Vec vectors
word2vec_similarity = 1 - cosine_similarity([vec1], [vec2])[0][0]

# Convert preprocessed text back to strings for TF-IDF
text1 = ' '.join(text1)
text2 = ' '.join(text2)

# Calculate TF-IDF similarity
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
tfidf_similarity = cosine_similarity(tfidf_matrix)[0][1]

# Open a file for writing
with open("output.txt", "w") as file:
    file.write(f"TF-IDF Similarity: {tfidf_similarity:.3f}\n")
    file.write(f"Word2Vec Similarity: {word2vec_similarity:.3f}\n")

print("Output has been saved to 'output.txt'")