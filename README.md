**Overview**

This project focuses on analyzing text data by employing different chunking techniques (fixed-length, sentence-based, and paragraph-based) and evaluating the quality of these chunks using information density, lexical overlap, semantic similarity, and thematic overlap.
The primary dataset used is the Wikipedia Simple English dataset from HuggingFace. The project leverages NLP libraries such as spaCy, HuggingFace Transformers, SentenceTransformers, and Scikit-learn.

🚀 **Features**

Text Chunking:
Fixed-Length Chunking: Splits text into chunks of a specific size with overlap.
Sentence-Based Chunking: Splits text based on sentence boundaries.
Paragraph-Based Chunking: Splits text into logical paragraphs.

Information Density Calculation:
Measures the average word count per chunk across chunking techniques.
Textual Analysis:
Lexical Overlap: Measures word-level overlap between adjacent chunks.
Semantic Similarity: Measures contextual similarity using embeddings from SentenceTransformers.
Thematic Overlap: Measures entity-level similarity using spaCy's Named Entity Recognition (NER).

Evaluation Metrics:
Average Lexical Overlap
Average Semantic Similarity
Average Thematic Overlap

🛠️ **Technologies Used**

Python
LangChain
HuggingFace Transformers
spaCy
SentenceTransformers (all-MiniLM-L6-v2)
Scikit-learn
NumPy
Regular Expressions (re)


📦 Setup and Installation

Clone the Repository:
git clone https://github.com/Saurabhkokare/Text-Chunking-and-Analysis-Pipeline

pip install -r requirements.txt

python -m spacy download en_core_web_md

python main.py
