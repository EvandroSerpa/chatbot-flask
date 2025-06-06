from flask import Flask, render_template, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Carrega a base de conhecimento
df = pd.read_excel("base_conhecimento.xlsx")
modelo = SentenceTransformer('paraphrase-MiniLM-L6-v2')
perguntas_base = df['Pergunta'].tolist()
embeddings_base = modelo.encode(perguntas_base, convert_to_tensor=True)

# Função que encontra a melhor resposta
def buscar_resposta_semantica(pergunta_usuario):
    embedding_usuario = modelo.encode(pergunta_usuario, convert_to_tensor=True)
    similaridades = util.cos_sim(embedding_usuario, embeddings_base)[0]
    indice_mais_proximo = similaridades.argmax().item()
    score = similaridades[indice_mais_proximo].item()

    if score < 0.5:
        return "Desculpe, não entendi sua pergunta."
    return df.iloc[indice_mais_proximo]['Resposta']

@app.route("/", methods=["GET", "POST"])
def index():
    resposta = ""
    if request.method == "POST":
        pergunta = request.form["pergunta"]
        resposta = buscar_resposta_semantica(pergunta)
    return render_template("index.html", resposta=resposta)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)