from fastapi import FastAPI
import joblib

app = FastAPI()

model, vectorizer = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(input: str):
    vec = vectorizer.transform([input])
    pred = model.predict(vec)[0]
    return {"prediction": int(pred)}
