# Document Similarity Finder

This Python project aims to find similarity between various formats of documents such as .docx, .pdf, .txt, etc. It utilizes natural language processing (NLP) techniques to analyze the content of documents and compute similarity scores between them.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-format Support**: Supports various document formats including .docx, .pdf, .txt, etc.
- **Text Extraction**: Utilizes libraries like `python-docx`, `PyPDF2`, and `textract` to extract text from different document formats.
- **Preprocessing**: Cleans and preprocesses the extracted text by removing stop words, punctuation, and converting to lowercase.
- **Vectorization**: Converts the preprocessed text into numerical vectors using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings.
- **Similarity Calculation**: Computes similarity scores between pairs of documents using cosine similarity or other similarity metrics.
- **Thresholding**: Allows setting a threshold to determine whether documents are similar or not based on the similarity scores.
- **CLI Interface**: Provides a command-line interface (CLI) for easy interaction and batch processing of documents.
- **Logging**: Logs important information and errors for debugging and monitoring purposes.

## Technologies Used

- **Python Libraries**:
  - `python-docx` for handling .docx files
  - `PyPDF2` for handling .pdf files
  - `textract` for extracting text from various document formats
  - `NLTK` (Natural Language Toolkit) for text preprocessing
  - `scikit-learn` for vectorization and similarity calculation
- **CLI Framework**: `argparse` for building the command-line interface
- **Logging**: Python's built-in `logging` module for logging messages

## Installation

1. Clone the repository: `git clone https://github.com/yourusername/document-similarity-finder.git`
2. Navigate to the project directory: `cd document-similarity-finder`
3. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Run the script with appropriate command-line arguments to specify the documents to compare and the desired similarity threshold.
2. The script will extract text from the input documents, preprocess it, compute similarity scores, and output the results.
3. Adjust the threshold as needed to filter similar documents based on your requirements.

Example usage:
```bash
multiple text file.py
PDF file code.pdf 

