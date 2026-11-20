import re
from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parent
fake_file = next((f for f in folder.glob("*.csv") if f.name.lower() == "fake.csv"), None)
real_file = next((f for f in folder.glob("*.csv") if f.name.lower() == "true.csv"), None)
output_file = folder / "news_dataset.csv"

if fake_file is None or real_file is None:
    print("Place Fake.csv, True.csv, and combine.py in the same folder.")
    raise SystemExit(1)

def clean(t):
    # Remove giveaways that reveal the source, not the truthfulness.
    t = re.sub(r"^.{0,120}?\(Reuters\)\s*-?\s*", "", t, count=1)
    t = re.sub(r"(?i)\breuters\b", "", t)
    t = re.sub(r"(?i)featured image.*$", "", t)
    t = re.sub(r"https?://\S+|pic\.twitter\.com/\S+|@\w+", "", t)
    return t.strip()

fake = pd.read_csv(fake_file)[["text"]].copy()
real = pd.read_csv(real_file)[["text"]].copy()
fake["label"] = "fake"
real["label"] = "real"

dataset = pd.concat([fake, real], ignore_index=True)
dataset["text"] = dataset["text"].fillna("").astype(str).map(clean)
dataset = dataset[dataset["text"].str.len() >= 50]
dataset = dataset.drop_duplicates(subset=["text"])
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
dataset.to_csv(output_file, index=False, encoding="utf-8")

print(f'Loading....')
print(f"Total: {len(dataset):,} | fake: {(dataset.label == 'fake').sum():,} | real: {(dataset.label == 'real').sum():,}")
