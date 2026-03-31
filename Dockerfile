FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn scikit-learn pandas joblib

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
