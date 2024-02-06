import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from sklearn.metrics.pairwise import cosine_similarity

# Read text files and store their content
def read_text_files(directory):
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
    return texts

# Sample directory containing text files
text_files_directory = 'Files'

# Read text content from files
text_documents = read_text_files(text_files_directory)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(text_documents)

# Word2Vec or FastText model training
# Choose either Word2Vec or FastText based on your preference
word_embedding_model = Word2Vec(sentences=[doc.split() for doc in text_documents], vector_size=100, window=5, min_count=1, sg=0)
#word_embedding_model = FastText(sentences=[doc.split() for doc in text_documents], vector_size=100, window=8, min_count=1, sg=0)

# Calculate average word embeddings for each document
average_embeddings = []
for doc in text_documents:
    words = doc.split()
    doc_embeddings = [word_embedding_model.wv[word] for word in words if word in word_embedding_model.wv]
    if doc_embeddings:
        average_doc_embedding = np.mean(doc_embeddings, axis=0)
        average_embeddings.append(average_doc_embedding)
    else:
        average_embeddings.append(np.zeros(word_embedding_model.vector_size))

# Calculate cosine similarity using TF-IDF
tfidf_similarity = cosine_similarity(tfidf_matrix)

# Calculate cosine similarity using Word Embeddings
embedding_similarity = cosine_similarity(average_embeddings)

print("TF-IDF Similarity:")
print(tfidf_similarity)

print("\nWord Embedding Similarity:")
print(embedding_similarity)

#1. passing result of tf idf into word2vec.
#2. Then cosine simimlarity, show result.
#3. Print both similarities and store then compare.
#4. same try with fasttext.
#5. Store result in file.
#6. How to apply berd(word embedding).