import json
import random
import os

# Load the fully updated prompt file
input_path = "C:\PES\internship\campedui-codegen\data\merged_campedui_components.json"
with open(input_path, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Shuffle the entries for unbiased splitting
random.shuffle(full_data)

# Define split ratio
split_ratio = 0.8
cutoff = int(len(full_data) * split_ratio)
train_set = full_data[:cutoff]
val_set = full_data[cutoff:]

# Save train and validation splits
train_path = "data/train_prompts.json"
val_path = "data/val_prompts.json"

with open(train_path, "w", encoding="utf-8") as f:
    json.dump(train_set, f, indent=2, ensure_ascii=False)

with open(val_path, "w", encoding="utf-8") as f:
    json.dump(val_set, f, indent=2, ensure_ascii=False)

train_path, val_path
