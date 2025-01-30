### 1. Importing libraries

from sentence_transformers import SentenceTransformer
from textExtractAndChunking import all_chunks, all_docs_name, all_chunks_doc_ids
from pinecone import Pinecone

### 2. Generating embeddings

# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print("MESSAGE : Generating embeddings from the chunks...")
embeddings = model.encode(all_chunks, convert_to_tensor = True, normalize_embeddings = True) # generating embeddings

for i, embedding in enumerate(embeddings): # printing embeddings
    print(f"\nEmbedding {i + 1}:\n{embedding}\n")

print("MESSAGE : All embeddings generated successfully!\n")
print(f"MESSAGE : Number of generated embeddings: {len(embeddings)} | Dimensions of generated embeddings: {embeddings[0].shape}\n")

### 3. Storing embeddings in 'Pinecone' Vector DB

pc = Pinecone(api_key = 'pcsk_.....')

index_name = 'hr-.....'
index = pc.Index(index_name)

print(f"MESSAGE : Storing embeddings in Pinecone index, '{index_name}'...\n")
for i, embedding in enumerate(embeddings):
    index.upsert(vectors = [{
        "id" : f"Chunk {i + 1}",
        "values" : embedding.numpy().tolist(),
        "metadata" : {"text" : all_chunks[i], "document" : all_docs_name[all_chunks_doc_ids[i]]}
        }], namespace = 'ns1')
    print(f"STORED : Chunk {i + 1}\n")

print(f"MESSAGE : The embeddings have been stored in Pinecone.")