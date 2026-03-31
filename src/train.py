import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import mlflow
import joblib

df = pd.read_csv("data/data.csv")

X=df["text"]
y=df["label"]

vectorizer=CountVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

acc = model.score(X_vec, y)

mlflow.set_experiment("mlops-sandbox")

with mlflow.start_run():
    mlflow.log_metric("accurary", acc)
    joblib.dump((model, vectorizer), "models/model.pkl")
    mlflow.log_artifact("models/model.pkl")

print("Model trained. Accuracy:", acc)
