import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
vectorizer = TfidfVectorizer 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = {
    'comment': [
        "You're awesome!", 
        "I hate you", 
        "What a great day!", 
        "You are so stupid", 
        "Thank you for your help", 
        "You're an idiot", 
        "I love this place", 
        "Get lost loser", 
        "Fantastic work", 
        "Shut up!",
        "get lost",
        "don't talk to me",
        "so annoying",
        "what wrong with you"],

        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,]   # 0 = non-toxic , 1 = toxic
}  

# create a data frame
df = pd.DataFrame(data)
print(df['label'].value_counts()) 


# split data
X_train, X_test, y_train, y_test = train_test_split(df['comment'],  df['label'], test_size=0.3, random_state=42) 

# vectorize text

vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# train model

model= LogisticRegression()
model.fit(x_train_vec, y_train) 

# predict and evaluate

y_pred = model.predict(X_test_vec)
print("classification report: \n", classification_report(y_test, y_pred))
print("model accuracy:",model.score(X_test_vec, y_test)) 

# function to predict custom input

def predict_comment(text): 
    vec = vectorizer.transform([text]) 
    result = model.predict(vec) [0]
    return "toxic" if result == 1 else "non-toxic"

# try it out

while True:
    comment = input("enter a comment(or 'exit' to quit): ")
    if comment.lower() == 'exit' : 
        break
    print("prediction:", predict_comment(comment)) 





