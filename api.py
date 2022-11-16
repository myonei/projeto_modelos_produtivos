from fastapi import FastAPI
from pydantic import BaseModel
import requests 
import mlflow 

# uvicorn api:app --reload --port 8001

#'Peso', 'TipoMotor', 'Cilindros', 'HP', 'Tempo', 
class Carro(BaseModel):
    Peso:int
    TipoMotor:int
    Cilindros:int
    HP:int
    Tempo:int
    
app = FastAPI()

@app.get("/experiments")
def get_experiments():
    url = 'http://localhost:5000/api/2.0/preview/mlflow/experiments/list'
    response = requests.request('GET', url=url)
    dados = response.json()
    return dados

@app.post("/model")
def predict(carros: Carro):
    mlflow.set_tracking_uri(uri='http://localhost:5000/')
    PATH = 'models:/carros_logit/Production'
    classes = ['baixo', 'médio', 'alto']
    loaded_model = mlflow.sklearn.load_model(PATH)

    dados = [[car[1] for car in carros]]
    label = loaded_model.predict(dados) #list array [[],[],[]]

    resultado = classes[int(label[0])]
    return {'class': resultado}
