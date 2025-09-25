import pandas as pd
import pickle

def map_students_to_teachers(student_file="data/processed/student_feedback_clean.csv",
                             teacher_file="data/processed/teacher_feedback_clean.csv",
                             model_path="models/sentiment_model.pkl",
                             output_file="data/processed/student_teacher_mapping.csv"):
    # Load feedback data
    students = pd.read_csv(student_file)
    teachers = pd.read_csv(teacher_file)

    # Load trained sentiment model
    with open(model_path, "rb") as f:
        vectorizer, model = pickle.load(f)

    # Predict sentiment for teacher feedback
    teachers["sentiment_pred"] = model.predict(vectorizer.transform(teachers["cleaned_feedback"]))

    # Rule-based mapping:
    mapping = []
    for i, student in students.iterrows():
        if "low grades" in student["cleaned_feedback"] or student["sentiment"] == "negative":
            teacher = teachers[teachers["sentiment_pred"] == "positive"].sample(1).iloc[0]
        else:
            teacher = teachers.sample(1).iloc[0]

        mapping.append({
            "student_id": student["id"],
            "teacher_id": teacher["id"],
            "student_feedback": student["feedback"],
            "teacher_feedback": teacher["feedback"]
        })

    mapping_df = pd.DataFrame(mapping)
    mapping_df.to_csv(output_file, index=False)
    print(f"✅ Student–Teacher mapping saved to {output_file}")

if __name__ == "__main__":
    map_students_to_teachers()
