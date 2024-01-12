from flask import Flask, request
from elasticsearch import Elasticsearch
from langchain.embeddings import LlamaCppEmbeddings

# Parametri
llama_model_path = "/home/ubuntu/llama_cpp_CPUonly/models/7B/ggml-model-q4_0.gguf"
INDEX_NAME = "ieee_db"

# Model start
embeddings = LlamaCppEmbeddings(model_path=llama_model_path)

# Elasticsearch connect to db
elastic_client = Elasticsearch("http://localhost:9200")

# REST API server start
app = Flask(__name__)

@app.route("/search")   #/<knn_num>/<document>
def search_on_elastic():
    print("Richiesta ricevuta")
    
    knn_num = request.args.get('knn_num')
    document = request.args.get('document')
    
    abstract = str(document).replace("+", " ")
    
    embedded_abstract = embeddings.embed_query(abstract)

    res = elastic_client.search(index=INDEX_NAME, knn={"field": "embedding", "k":knn_num, "num_candidates": 50, "query_vector": embedded_abstract}, pretty=True)

    print("Richiesta completata")

    results = res["hits"]["hits"]
    
    return results