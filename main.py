import ollama
import numpy as np

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
KNOWLEDGE_BASE_SIZE =  3

def cosine_similarity(a, b):
    dot = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    return dot / (mag_a * mag_b)

def retrieve_similarities(query, quantity):
    query_embedding = ollama.embed(EMBEDDING_MODEL, query)['embeddings'][0]
    similarities = []
    for chunk, embedding in vector_db:
        similarity_score = cosine_similarity(embedding, query_embedding)
        similarities.append((chunk, similarity_score))
    similarities.sort(key=lambda tup: tup[1], reverse=True)
    return similarities[0:quantity]

dataset = []
with open("./facts.txt") as stream:
    dataset = stream.readlines()

vector_db = []

for chunk in dataset:
    embedding = ollama.embed(EMBEDDING_MODEL, chunk)['embeddings'][0]
    vector_db.append((chunk, embedding))

while True:
    query = input("Cat query: ")
    knowledge_base = retrieve_similarities(query, KNOWLEDGE_BASE_SIZE)

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, _ in knowledge_base])}
    '''
    stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': query},
    ],
    stream=True,
    )
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)
    print("\n")
