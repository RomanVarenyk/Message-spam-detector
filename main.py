import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def predict_message(model, vectorizer, message):
    message_transformed = vectorizer.transform([message])
    prediction = model.predict(message_transformed)
    probabilities = model.predict_proba(message_transformed)
    confidence = max(probabilities[0])
    label = 'human' if prediction[0] == 0 else 'AI'
    return label, confidence


data_path = 'spam.csv'
data = pd.read_csv(data_path, header=None, names=['label', 'message'], usecols=[0, 1], encoding='ISO-8859-1')

data = data[data['label'].isin(['ham', 'spam'])]

data['label'] = data['label'].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_transformed, y_train)