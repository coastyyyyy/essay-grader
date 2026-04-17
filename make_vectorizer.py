from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

texts = [
    "education is important",
    "technology helps people",
    "hard work gives success",
    "students should study",
    "environment must be protected"
]

vectorizer = TfidfVectorizer()
vectorizer.fit(texts)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Done")
