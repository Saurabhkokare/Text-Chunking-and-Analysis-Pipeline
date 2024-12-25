from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset
from langchain.schema import Document
import re
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


dataset = load_dataset(
    'wikipedia', 
    '20220301.simple', 
    split='train[:1%]', 
    trust_remote_code=True
)


extracted_data=dataset


documents = [Document(page_content=item['text']) for item in extracted_data]

    
## 1.Fixed-length chunks
def chunk_text(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(documents)
    
    return text_chunks

text_chunks=chunk_text(documents)

## 2.Senetenece-based chunking

# For sentence-based chunking We Split sentences based on '.', '!', and '?', followed by a space or end of string
    
def sentence_based_chunking(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences

sentence_chunk=sentence_based_chunking(str(documents))
sentence_chunk

## 3.Paragraph-based chunking

# For Paragraph-based chunking We Split Paragraphs based on two or more newline characters.
    
def paragraph_based_chunking(text):
    paragraphs = re.split(r'\n{2,}', text.strip())
    return [para.strip() for para in paragraphs if para.strip()]


paragraph_chunk=paragraph_based_chunking(str(documents))
paragraph_chunk


## 1.information density per chunk
import numpy as np

def information_density(chunks):
    return [len(str(chunk).split()) for chunk in chunks]

length_text_chunks_id=np.mean(information_density(text_chunks))
length_sentence_chunks_id=np.mean(information_density(sentence_chunk))
length_paragraph_chunks_id=np.mean(information_density(paragraph_chunk))

print("Calculated length of chunks using Information Density method : \n1)length of text chunks :",length_text_chunks_id,"\n2)length of sentence-based chunks :",length_sentence_chunks_id,"\n3)length of paragraph-based chunks :",length_paragraph_chunks_id)



nlp = spacy.load("en_core_web_md")
model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = text_chunks

def lexical_overlap(chunk1, chunk2):
    text1 = chunk1.text if isinstance(chunk1, spacy.tokens.Doc) else str(chunk1)
    text2 = chunk2.text if isinstance(chunk2, spacy.tokens.Doc) else str(chunk2)
    
    vectorizer = CountVectorizer().fit([text1, text2])
    X = vectorizer.transform([text1, text2]).toarray()
    overlap = np.sum(np.minimum(X[0], X[1]))
    
    return overlap / max(np.sum(X[0]), np.sum(X[1]))


def semantic_similarity(chunk1, chunk2):
    text1 = chunk1.text if isinstance(chunk1, spacy.tokens.Doc) else str(chunk1)
    text2 = chunk2.text if isinstance(chunk2, spacy.tokens.Doc) else str(chunk2)
    
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    similarity = util.cos_sim(embedding1, embedding2)
    return similarity.item()

def thematic_overlap(chunk1, chunk2):
    
    text1 = chunk1.page_content if hasattr(chunk1, 'page_content') else str(chunk1)
    text2 = chunk2.page_content if hasattr(chunk2, 'page_content') else str(chunk2)
    
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    entities1 = set(ent.text for ent in doc1.ents)
    entities2 = set(ent.text for ent in doc2.ents)
    
    overlap = len(entities1 & entities2) / max(len(entities1 | entities2), 1)
    return overlap


lexical_scores = []
semantic_scores = []
thematic_scores = []


def analyze_chunks(chunks):
    for i in range(len(chunks) - 1):
        lex_overlap = lexical_overlap(chunks[i], chunks[i+1])
        sem_similarity = semantic_similarity(chunks[i], chunks[i+1])
        them_overlap = thematic_overlap(chunks[i], chunks[i+1])
        
        lexical_scores.append(lex_overlap)
        semantic_scores.append(sem_similarity)
        thematic_scores.append(them_overlap)

analyze_chunks(chunks)
mean_lexical = np.mean(lexical_scores)
mean_semantic = np.mean(semantic_scores)
mean_thematic = np.mean(thematic_scores)
    
print("\n✅ **Overall Averages Across All Chunk Pairs:**")
print(f"  - Mean Lexical Overlap: {mean_lexical:.2f}")
print(f"  - Mean Semantic Similarity: {mean_semantic:.2f}")
print(f"  - Mean Thematic Overlap: {mean_thematic:.2f}")
