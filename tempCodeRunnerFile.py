import pandas as pd
import random

# Sample feedback phrases
student_pos = [
    "The teacher explains concepts clearly.", "My teacher motivates me to do better.",
    "Very patient and helpful.", "Always encourages participation.", 
    "Makes lessons engaging and fun."
]
student_neu = [
    "The lessons are fine but sometimes too long.", "The class pace is average.",
    "Sometimes good, sometimes boring.", "The explanations are okay.",
    "Nothing special about the teaching style."
]
student_neg = [
    "The teacher rushes through topics.", "Doesn't clear doubts properly.",
    "Gets irritated when asked questions.", "Ignores students' problems.",
    "Makes learning stressful."
]

teacher_pos = [
    "The student works hard and submits on time.", "Very attentive in class.",
    "Shows improvement every week.", "Helps classmates during group work.",
    "Consistently performs well in tests."
]
teacher_neu = [
    "The student performs at an average level.", "Not outstanding but steady.",
    "Completes work but without much interest.", "Neutral behavior in class.",
    "Sometimes participates, sometimes doesn’t."
]
teacher_neg = [
    "The student rarely pays attention.", "Often misses deadlines.",
    "Distracts others during class.", "Does not complete homework.",
    "Shows lack of interest in studies."
]

def generate_feedback(pos, neu, neg, count=100):
    data = []
    for i in range(count):
        if i % 3 == 0:
            text = random.choice(pos); sentiment = "Positive"
        elif i % 3 == 1:
            text = random.choice(neu); sentiment = "Neutral"
        else:
            text = random.choice(neg); sentiment = "Negative"
        data.append((i+1, text, sentiment))
    return pd.DataFrame(data, columns=["id", "feedback_text", "sentiment"])

# Generate datasets
df_students = generate_feedback(student_pos, student_neu, student_neg, 100)
df_teachers = generate_feedback(teacher_pos, teacher_neu, teacher_neg, 100)

# Save to CSV inside dataset folder
df_students.to_csv("data/raw/student_feedback.csv", index=False)
df_teachers.to_csv("data/raw/teacher_feedback.csv", index=False)
