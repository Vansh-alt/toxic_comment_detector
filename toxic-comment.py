# Import required libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Define dataset
data = {
    'comment': [
        # Toxic (1)

        "i hate you", "go to hell", "you are so stupid", "shut up",
        "get lost", "you idiot", "you're the worst", "disgusting behavior",
        "what a moron", "stop being dumb", "nobody likes you", "you're trash",
        "just die", "kill yourself", "pathetic loser", "ugly face",
        "you are useless", "worst human ever", "you make me sick", "go away",
        "you're pathetic", "ugly", "such a cheap person", "so chringe", "you're dirty",
        "dirty minded",

        # Non-Toxic (0)

        "you are amazing", "thank you", "have a great day", "you are welcome",
        "such a nice person", "stay strong", "keep going", "you can do it",
        "i appreciate you", "wonderful work", "i believe in you", "you're great",
        "nice job", "stay positive", "much love",
        "so proud of you", "how are you", "good luck",
        "all the best", "happy birthday", "congratulations", "you're beautiful",
        "what a lovely smile", "you're awesome", "i love your attitude", "well done",
       
    ],
    'label': [
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1,

        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0,
        
    ]
}

# Create DataFrame

df = pd.DataFrame(data)

# Split data into train and test

X_train, X_test, y_train, y_test = train_test_split(df['comment'], 
df['label'], test_size=0.3, random_state=42)

# Vectorize text using TF-IDF

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_vec, y_train)
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Evaluate model

y_pred = model.predict(X_test_vec)
print("Classification report:\n", classification_report(y_test, y_pred))
print("Model accuracy:", model.score(X_test_vec, y_test))

# Save model and vectorizer

joblib.dump(model, "toxic_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Predict function

def predict_comment(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    return "toxic" if result == 1 else "non-toxic"

# Interactive testing

while True:
    comment = input("Enter a comment (or 'exit' to quit): ")
    if comment.lower() == 'exit':
        break
    print("Prediction:", predict_comment(comment))



