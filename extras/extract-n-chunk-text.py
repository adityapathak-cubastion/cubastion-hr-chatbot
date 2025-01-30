### 1. Importing libraries

import fitz, os, docx, nltk, re

### 2. Defining functions

# 2.1 Extracting text

def get_text_from_pdf(pdf_path): 
    doc = fitz.open(pdf_path)
    text = ""
    index_pattern = re.compile(r'Table of Contents', re.IGNORECASE)
    
    for page in doc:
        page_text = page.get_text()
        if not index_pattern.search(page_text):
            text += page.get_text()
        else: continue

    return text

def get_text_from_docx(docx_path): # alternate chunking methods below
    doc = docx.Document(docx_path)
    text = ""
    
    for para in doc.paragraphs:
        text += para.text
    
    return text

# 2.2 Chunking - major applications use RecursiveCharacterTextSplitter, Langchain - Retrieval Chain.. not using here though.

# def chunk_text(text, max_tokens = 512): # this uses sentence tokenizer. this was working well, but was creating large chunks with more than 512 tokens (which all mini lm might not be able to handle)
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        
        if current_length + len(sentence_tokens) > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = sentence_tokens
            current_length = len(sentence_tokens)
        
        else:
            current_chunk.extend(sentence_tokens)
            current_length += len(sentence_tokens)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# def chunk_text(text, max_tokens = 512): # this uses word tokenizer. this is working, fixed the > 512 tokens problem from sentence tokenizer. but might lose some context.
    words = nltk.word_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        
        if current_length + len(word_tokens) > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = word_tokens
            current_length = len(word_tokens)
        
        else:
            current_chunk.extend(word_tokens)
            current_length += len(word_tokens)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 2.2 Chunking text

# def chunk_text(text, doc_id, max_tokens = 512): # this uses sentence tokenizer + BertTokenizer. this is working, fixed the > 512 tokens issue.
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        sentence_tokens = tokenizer.tokenize(sentence)
        
        if current_length + len(sentence_tokens) > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = sentence_tokens
            current_length = len(sentence_tokens)
        
        else:
            current_chunk.extend(sentence_tokens)
            current_length += len(sentence_tokens)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    chunks_doc_ids = [doc_id] * len(chunks)
    return chunks, chunks_doc_ids

def chunk_text(text, doc_id, chunk_size = 500): # no tokenizing
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    chunks_doc_ids = [doc_id] * len(chunks)
    return chunks, chunks_doc_ids

def recombine_subwords(tokens):
    combined_tokens = []
    current_word = ""
    
    for token in tokens:
        if token.startswith("##"):  # For WordPiece tokenizer
            current_word += token[2:]
        else:
            if current_word:
                combined_tokens.append(current_word)
            current_word = token
    
    if current_word:
        combined_tokens.append(current_word)
    
    return ' '.join(combined_tokens)

def post_process_text(text):
    text = text.lower() # making text lower-case...
    text = re.sub(r'\s+', ' ', text).strip() #... and removing extra white-spaces
    return text

def post_process_chunk(chunk):
    chunk = re.sub(r'\.{2,}', '', chunk) # removing sequences of dots (2 or more dots together)...
    chunk = re.sub(r'\s*\.\s*', ' ', chunk) #... and isolated dots
    chunk = re.sub(r' \s*', ' ', chunk)
    return chunk

### 3 Main

# 3.1 Extracting text

dir_path = "C:/Users/AdityaPathak/....."
all_texts = []
all_docs_name = []

print(f"MESSAGE : Extracting text from all .pdf and .docx files in {dir_path}...\n")
for document in os.listdir(dir_path):
    if document.endswith('.pdf'):
        pdf_path = os.path.join(dir_path, document)
        pdf_text = ""
        pdf_text += get_text_from_pdf(pdf_path)
        all_texts.append(pdf_text)
        all_docs_name.append(str(os.path.basename(pdf_path)))
    
    elif document.endswith('.docx'):
        docx_path = os.path.join(dir_path, document)
        docx_text = ""
        docx_text += get_text_from_docx(docx_path)
        all_texts.append(docx_text)
        all_docs_name.append(str(os.path.basename(docx_path)))

print("-----DOCUMENT NAMES-----\n", all_docs_name)

# 3.2 Text chunking

all_texts = [post_process_text(text) for text in all_texts]

# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# nltk.download('punkt_tab')
# nltk.download('punkt')
all_chunks = []
all_chunks_doc_ids = []

# print("\nMESSAGE : Beginning the chunking process using BertTokenizer...")
print("\nMESSAGE : Beginning the chunking process...")
for doc_id, text in enumerate(all_texts):
    chunks, chunks_doc_ids = chunk_text(text, doc_id, 512) # tried 384, 300, 284 as well
    all_chunks.extend(chunks)
    all_chunks_doc_ids.extend(chunks_doc_ids)

all_chunks = [post_process_chunk(chunk) for chunk in all_chunks]    
all_chunks = [recombine_subwords(chunk.split()) for chunk in all_chunks]

for i, chunk in enumerate(all_chunks): # Post processed
    print(f"\nChunk {i + 1} | Doc {all_chunks_doc_ids[i] + 1}: ({all_docs_name[all_chunks_doc_ids[i]]})\n{chunk}\n")

print("MESSAGE : All chunks generated successfully!\n")