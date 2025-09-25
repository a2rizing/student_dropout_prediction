import pickle

def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_sentiment(text, model_path="models/sentiment_model.pkl"):
    vectorizer, model = load_model(model_path)
    X = vectorizer.transform([text])
    return model.predict(X)[0]
