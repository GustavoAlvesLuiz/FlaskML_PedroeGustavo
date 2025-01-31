import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Carregar o modelo e as classes
model = pickle.load(open("model.pkl", "rb"))
names = pickle.load(open("names.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receber os valores do formulário
        pelos = int(request.form["pelos"])
        penas = int(request.form["penas"])
        oviparo = int(request.form["oviparo"])
        voador = int(request.form["voador"])
        aquatico = int(request.form["aquatico"])
        predador = int(request.form["predador"])
        com_dentes = int(request.form["com_dentes"])
        coluna_vertebral = int(request.form["coluna_vertebral"])
        respira = int(request.form["respira"])
        venenoso = int(request.form["venenoso"])
        barbatanas = int(request.form["barbatanas"])
        pernas = int(request.form["pernas"])
        cauda = int(request.form["cauda"])
        domestico = int(request.form["domestico"])

        # Preparar a entrada para o modelo
        entrada = [[pelos, penas, oviparo, voador, aquatico, predador, com_dentes, coluna_vertebral, respira, venenoso, barbatanas, pernas, cauda, domestico]]
        
        # Realizar a previsão
        pred = model.predict(entrada)
        
        # Obter a categoria do animal
        output = [k for k, v in names.items() if v == pred[0]][0]
        prediction_text = f"A categoria prevista do animal é: {output}"

    except Exception as e:
        prediction_text = f"Erro ao processar a entrada: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text)

@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])
    output = names[pred[0]]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
