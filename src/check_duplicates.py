import hashlib
from datasets import load_dataset

# Stream records so the entire dataset doesn't need to fit in memory.
dataset = load_dataset('Jinyan1/COLING_2025_MGT_en', streaming=True)

print(dataset)

# Create fingerprints for all training texts so we can later check for exact duplicates in the dev set.
train_hashes = set()
for row in dataset["train"]:
    text = row["text"]
    text_hash = hashlib.sha256(text.encode("utf-8")).digest()
    train_hashes.add(text_hash)

duplicate_count = 0
duplicate_hashes = set()

# Create fingerprints for dev texts that'll be used to check for exact duplicates
for row in dataset["dev"]:
    text = row["text"]
    text_hash = hashlib.sha256(text.encode("utf-8")).digest()
    if text_hash in train_hashes:
        duplicate_count = duplicate_count + 1
        duplicate_hashes.add(text_hash)

print(duplicate_count)
print(len(duplicate_hashes))