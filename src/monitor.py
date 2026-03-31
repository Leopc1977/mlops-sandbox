import pandas as pd


def detect_drift(train_file, prod_file):
    train = pd.read_csv(train_file)
    prod = pd.read_csv(prod_file)
    drift = abs(train['label'].mean() - prod['label'].mean())
    return drift


drift_score = detect_drift("data/data.csv", "data/new_data.csv")
print("Drift score:", drift_score)
