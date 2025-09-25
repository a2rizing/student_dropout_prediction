import pandas as pd

# Load dataset (student feedback first)
df = pd.read_csv("data/raw/student_feedback.csv")

# Show first 5 rows
print(df.head())

# Check data balance
print(df['sentiment'].value_counts())
