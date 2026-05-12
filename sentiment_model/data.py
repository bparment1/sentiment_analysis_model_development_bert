import re
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

#too many hard coded values, use config
MODEL_NAME = "bert-base-uncased"

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts #movie review tex
        self.labels = labels #0,1 negative, postive sentiment
        self.tokenizer = tokenizer #tokenizer for text

    def clean(self, text):
        text = text.strip() #removes leading/trailing white space
        text = re.sub(r"http\S+", "URL", text) #replace URLs with URL
        text = re.sub(r"@\w+", "USER", text) #replace @mentions with USER
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.clean(self.texts[idx])
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,    #should not be hard coded, use config                  # explicit max_length for BERT
            return_tensors="pt"
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),  #  bug 1 fixed + explicit dtype
        }


class DataModule:
    def __init__(self, tokenizer):
        raw = load_dataset("imdb")
        split = raw["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")

        self.train = SentimentDataset(split["train"]["text"], split["train"]["label"], tokenizer)
        self.val   = SentimentDataset(split["test"]["text"],  split["test"]["label"],  tokenizer)
        self.test  = SentimentDataset(raw["test"]["text"],    raw["test"]["label"],    tokenizer)

    def loaders(self, batch_size=8, num_workers=2):     #  configurable + num_workers
        return (
            DataLoader(self.train, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
            DataLoader(self.val,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(self.test,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
        )

    @staticmethod
    def get_tokenizer(model_name=MODEL_NAME):           # bug 2 fixed — default arg added
        return AutoTokenizer.from_pretrained(model_name)