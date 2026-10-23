
from datasets import load_dataset

# Load the dataset normally
dataset = load_dataset("Jinyan1/COLING_2025_MGT_en")

# Get the training and development splits
train_data = dataset["train"]
dev_data = dataset["dev"]

print("Full training dataset:", len(train_data))
print("Full development dataset:", len(dev_data))

# Shuffle and select manageable samples
train_data = train_data.shuffle(seed=42).select(range(20000))
dev_data = dev_data.shuffle(seed=42).select(range(5000))

print("\nSelected training examples:", len(train_data))
print("Selected development examples:", len(dev_data))

# Count the labels
for name, split in [
    ("Training", train_data),
    ("Development", dev_data)
]:
    labels = split["label"]

    print(f"\n{name} data:")
    print("Human-written:", labels.count(0))
    print("AI-generated:", labels.count(1))
