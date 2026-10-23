
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import hashlib
import joblib
import os

# Load the dataset
dataset = load_dataset("Jinyan1/COLING_2025_MGT_en")

# Select reproducible samples
train_data = (
    dataset["train"]
    .shuffle(seed=42)
    .select(range(20000))
)

dev_data = (
    dataset["dev"]
    .shuffle(seed=42)
    .select(range(5000))
)

# Hash every training text to identify exact overlaps
print("Checking training texts for duplicates...")

train_hashes = set()

for row in dataset["train"]:
    text_hash = hashlib.sha256(
        row["text"].encode("utf-8")
    ).hexdigest()
    train_hashes.add(text_hash)

print("Unique training text hashes:", len(train_hashes))

# Remove development texts appearing anywhere in training
clean_indices = []

for index, row in enumerate(dev_data):
    text_hash = hashlib.sha256(
        row["text"].encode("utf-8")
    ).hexdigest()

    if text_hash not in train_hashes:
        clean_indices.append(index)

removed_count = len(dev_data) - len(clean_indices)
dev_data = dev_data.select(clean_indices)

print("Development examples originally:", 5000)
print("Exact overlaps removed:", removed_count)
print("Clean development examples:", len(dev_data))

if len(dev_data) == 0:
    raise ValueError("No development examples remain.")

# Extract text and labels
X_train = train_data["text"]
y_train = train_data["label"]

X_dev = dev_data["text"]
y_dev = dev_data["label"]

print("\nTraining examples:", len(X_train))
print("Clean development examples:", len(X_dev))

# Build the machine-learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
print("Evaluating on clean development data...")
predictions = model.predict(X_dev)

print("\nAccuracy:", round(
    accuracy_score(y_dev, predictions), 4
))

print("\nClassification report:")
print(classification_report(
    y_dev,
    predictions,
    labels=[0, 1],
    target_names=["Human-written", "AI-generated"],
    digits=4,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_dev, predictions, labels=[0, 1]))

# Save the model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ai_text_detector.joblib")

print("\nModel saved to model/ai_text_detector.joblib")
