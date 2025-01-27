from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# pc = Pinecone(api_key = 'pcsk_6kPVhJ_NRcDPMWvHpFZqbRQcGEAGnpYhRzX82i4yTBTzkBibS8sTyHj6sMcDJRGRNeaJnc')
# # index_name = 'hr-policy-docs-embeddings'
# index_name = 'hr-policy-docs-higher-embeddings'
# index = pc.Index(index_name)

# def encode_query(query):
#     query_embedding = model.encode([query], convert_to_tensor = True).numpy().tolist()
#     return query_embedding

# def search_pinecone(query_embedding, top_k = 3):
#     results = index.query(namespace = 'ns1', vector = query_embedding, top_k = top_k, include_metadata = True)
#     return results

# def retrieve_relevant_chunks(results):
#     retrieved_chunks = [match for match in results['matches']] 
#     return retrieved_chunks

# query_embed = encode_query("Leave Travel Allowance")
# res = search_pinecone(query_embed, 2)
# print(retrieve_relevant_chunks(res))


def encode_query(query):
    query_embedding = model.encode([query], convert_to_tensor = True).numpy().tolist()
    return query_embedding

print(encode_query("work from home?"))

# queries = ["What're the policies for prevention of sexual harrassment at Cubastion Consulting?", "What's the dress code for all employees of Cubastion Consulting?",\
#            "Tell me about the separation policy of Cubastion Consulting", "I have resigned from Cubastion, and my car has been stolen a few days back. How does this affect my car lease policy?", "What is the process to follow for availing the benefit of female travel reimbursement policy?"]

# for query in queries:
#     print(encode_query(query))
#     print("\n\n\n")