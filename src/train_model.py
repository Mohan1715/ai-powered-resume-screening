import pandas as pd
import re
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# 1️⃣ Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


# -----------------------------
# 2️⃣ Load Dataset
# -----------------------------
df = pd.read_csv("../data/resumes.csv")

df["Cleaned"] = df["Resume_str"].apply(clean_text)

X = df["Cleaned"]
y = df["Category"]

# -----------------------------
# 3️⃣ TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)
X_tfidf = vectorizer.fit_transform(X)

# -----------------------------
# 4️⃣ Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5️⃣ Train Model
# -----------------------------
model = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    solver='lbfgs'
)
model.fit(X_train, y_train)

# -----------------------------
# 6️⃣ Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 7️⃣ Evaluation Metrics
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Accuracy:", accuracy)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 8️⃣ Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 9️⃣ Save Model
# -----------------------------
joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n🔥 Model saved successfully!")