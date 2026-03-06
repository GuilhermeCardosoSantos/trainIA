from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Dados simulando um banco
services = [
    "pedreiro",
    "eletricista",
    "encanador",
    "pintor",
    "reforma de casa",
    "assistência técnica celular"
]

# Gerar embeddings dos serviços
service_embeddings = model.encode(services)

# Consulta do usuário
query = "consertar computador"

# Gerar embedding da consulta
query_embedding = model.encode([query])

# Calcular similaridade
similarities = cosine_similarity(query_embedding, service_embeddings)

# Encontrar o melhor resultado
best_index = similarities.argmax()

print("Consulta:", query)
print("Resultado mais próximo:", services[best_index])
print("Score:", similarities[0][best_index])