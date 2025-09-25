import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

def train_sentiment(input_path, model_path):
    df = pd.read_csv(input_path)

    X = df['cleaned_feedback']
    y = df['sentiment']

    vectorizer = TfidfVectorizer(max_features=500)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    with open(model_path, "wb") as f:
        pickle.dump((vectorizer, model), f)
    print(f"Sentiment model saved to {model_path}")

if __name__ == "__main__":
    train_sentiment("data/processed/student_feedback_clean.csv", "models/sentiment_model.pkl")
