import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("data/Twitter_Data.csv")

df = df[['clean_text', 'category']]

df.dropna(inplace=True)

# Convert labels
label_map = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

df['category'] = df['category'].map(label_map)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df['clean_text'] = df['clean_text'].apply(preprocess)

X = df['clean_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
    )

model.fit(X_train_vec, y_train)

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy*100:.2f}%")

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print(df["category"].value_counts())

print("Model Saved Successfully")