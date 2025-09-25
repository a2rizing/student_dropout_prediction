import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def process_feedback(input_path, output_path):
    df = pd.read_csv(input_path)
    df['cleaned_feedback'] = df['feedback'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_feedback("data/raw/student_feedback.csv", "data/processed/student_feedback_clean.csv")
    process_feedback("data/raw/teacher_feedback.csv", "data/processed/teacher_feedback_clean.csv")
