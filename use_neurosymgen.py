#!/usr/bin/env python3
"""
استفاده عمومی از NeuroSymGen برای هر task
اینجا کدهای task-specific (XSS, Phishing, ...) رو اضافه می‌کنیم
"""

from neurosymgen import NeuroSymGenLayer, GrokReActAgent, HardwareOptimizer
from neurosymgen.kg.hetero_graph import create_sample_hetero_kg
import torch
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# ✅ اینجا کدهای task-specific رو می‌نویسیم
# ============================================================

def create_xss_kg(num_features: int = 128, num_nodes: int = 5):
    """
    XSS-specific KG - اینجا باشه نه تو core library
    """
    nodes = []
    patterns = {
        "script_tag": "script",
        "event_handler": "onerror",
        "javascript_proto": "javascript:",
        "svg_payload": "svg",
        "iframe_inject": "iframe"
    }

    for pattern in patterns.values():
        feature = torch.randn(num_features)
        # اگه pattern توی payload باشه، اولین 10 dim رو 1 کن
        nodes.append(feature)

    while len(nodes) < num_nodes:
        nodes.append(torch.randn(num_features))

    return torch.stack(nodes).unsqueeze(0)  # [1, 5, 128]


def create_phishing_kg(num_features: int = 128):
    """
    Phishing-specific KG
    """
    nodes = []
    patterns = {
        "login_form": "login",
        "urgent_language": "urgent",
        "mismatched_url": "url_mismatch",
        "suspicious_sender": "sender",
        "https_check": "https"
    }

    for _ in patterns:
        nodes.append(torch.randn(num_features))

    return torch.stack(nodes).unsqueeze(0)


def create_sentiment_kg(num_features: int = 128):
    """
    Sentiment-specific KG
    """
    nodes = []
    patterns = {
        "positive_words": ["good", "great", "excellent"],
        "negative_words": ["bad", "terrible", "awful"],
        "intensifiers": ["very", "extremely", "absolutely"],
        "negations": ["not", "never", "no"],
        "emojis": ["😊", "😢", "😡"]
    }

    for _ in patterns:
        nodes.append(torch.randn(num_features))

    return torch.stack(nodes).unsqueeze(0)


# ============================================================
# ✅ CORE: General Dataset
# ============================================================

class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, text_column, label_column, max_len=512, task="generic"):
        self.df = pd.read_csv(csv_path)
        self.max_len = max_len
        self.task = task
        self.text_column = text_column

        self.vocab = self._build_vocab(text_column)

    def _build_vocab(self, text_column):
        chars = set()
        for text in self.df[text_column]:
            chars.update(str(text))
        return {c: i + 1 for i, c in enumerate(sorted(chars))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx][self.text_column])
        label = int(self.df.iloc[idx][label_column])

        # Encode
        encoded = [self.vocab.get(c, 0) for c in text[:self.max_len]]
        encoded += [0] * (self.max_len - len(encoded))
        x = torch.tensor(encoded, dtype=torch.float32)

        # KG: بر اساس task انتخاب کن
        if self.task == "xss":
            kg_data = create_xss_kg()
        elif self.task == "phishing":
            kg_data = create_phishing_kg()
        elif self.task == "sentiment":
            kg_data = create_sentiment_kg()
        else:
            kg_data = create_sample_hetero_kg().concept.x.unsqueeze(0)

        return x, kg_data, torch.tensor(label, dtype=torch.float32)


# ============================================================
# ✅ MAIN
# ============================================================

def main():
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "xss"

    dataset = GeneralDataset(
        csv_path=f"{task}_dataset.csv",
        text_column="text",  # یا payload یا email
        label_column="label",
        task=task
    )

    model = NeuroSymGenLayer(input_size=512, num_rules=4, output_size=1)
    train_and_eval(model, dataset, task=task)


if __name__ == "__main__":
    main()