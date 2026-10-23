
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset = load_dataset(
    "Jinyan1/COLING_2025_MGT_en",
)

train_sample = list(dataset["train"].take(1000))
dev_sample = list(dataset["dev"].take(100))

print("Training sample size:", len(train_sample))
print("Development sample size:", len(dev_sample))


train_texts = [row["text"] for row in train_sample]
dev_texts = [row["text"] for row in dev_sample]

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english"
)

train_vectors = vectorizer.fit_transform(train_texts)
dev_vectors = vectorizer.transform(dev_texts)

similarities = cosine_similarity(dev_vectors, train_vectors)

print("Similarity matrix shape:", similarities.shape)
print("Highest similarity:", similarities.max())


best_dev_index, best_train_index = divmod(
    similarities.argmax(),
    similarities.shape[1]
)

print("\nMost similar development text:")
print(dev_texts[best_dev_index])

print("\nMost similar training text:")
print(train_texts[best_train_index])

print("\nSimilarity score:")
print(similarities[best_dev_index, best_train_index])


import numpy as np

flat_indices = np.argsort(similarities.ravel())[-10:][::-1]

print("\nTop 10 most similar text pairs:")

for flat_index in flat_indices:
    dev_index, train_index = divmod(
        flat_index, similarities.shape[1]
    )

    score = similarities[dev_index, train_index]

    print("\nSimilarity:", round(score, 4))
    print("Development:", dev_texts[dev_index][:300])
    print("Training:", train_texts[train_index][:300])


