
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

data_path = Path("backend/data/processed/amrita/amrita_sequences.npz")
data = np.load(data_path, allow_pickle=True)

y = data["y"]
groups = data["groups"]
indices = np.arange(len(y))

# Recreate the same split used by training
train_val, test = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y
)

train, val = train_test_split(
    train_val,
    test_size=0.15,
    random_state=43,
    stratify=y[train_val]
)

def show_files(title, selected):
    print(f"\n{'=' * 15} {title}: {len(selected)} videos {'=' * 15}")
    for i in selected:
        print(groups[i])

show_files("TRAINING", train)
show_files("VALIDATION", val)
show_files("TESTING", test)
