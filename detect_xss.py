#!/usr/bin/env python3
"""
XSS Detection - NeuroSymGen Simplified
Dataset: sentence, label
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from neurosymgen.reasoning.grok_client import GrokReActAgent

# NeuroSymGen
from neurosymgen import NeuroSymGenLayer, HardwareOptimizer

# ==================== CONFIG ====================
CSV_PATH = "XSS_dataset.csv"  # فایل خودت رو اینجا بگذار
MAX_LEN = 512  # طول حداکثر جمله
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001


# ================================================

def create_vocab_from_dataset(df, text_column):
    """ساخت vocabulary از کاراکترها"""
    chars = set()
    for text in df[text_column]:
        chars.update(str(text))
    return {c: i + 1 for i, c in enumerate(sorted(chars))}


def encode_text(text, vocab, max_len=MAX_LEN):
    """تبدیل text به tensor"""
    encoded = [vocab.get(c, 0) for c in str(text)[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return torch.tensor(encoded, dtype=torch.float32)


def create_kg_generic(text, num_nodes=5, num_features=128):
    """KG generic برای هر payload"""
    # 5 pattern: length, special_chars, numbers, uppercase, whitespace
    patterns = {
        "length": len(text) > 100,
        "special_chars": any(c in text for c in "<>\"'"),
        "numbers": any(c.isdigit() for c in text),
        "uppercase": sum(c.isupper() for c in text) > 5,
        "whitespace": text.count(" ") > 10
    }

    nodes = []
    for present in patterns.values():
        feature = torch.randn(num_features)
        if present:
            feature[:10] = 1.0
        nodes.append(feature)

    # Pad nodes
    while len(nodes) < num_nodes:
        nodes.append(torch.randn(num_features))

    return torch.stack(nodes)  # [1, 5, 128]


def train_model(model, train_loader, epochs=EPOCHS):
    """Training loop"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.BCELoss()

    print("🔥 Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, kg, y in train_loader:
            optimizer.zero_grad()
            output, info = model(x, kg_data=kg)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


# def evaluate_model(model, test_loader):
#     """Evaluation"""
#     model.eval()
#     preds = []
#     labels = []
#
#     print("\n📊 Evaluating...")
#     with torch.no_grad():
#         for x, kg, y in test_loader:
#             output, _ = model(x, kg_data=kg)
#             preds.extend((output.squeeze() > 0.5).float().cpu().numpy())
#             labels.extend(y.cpu().numpy())
#
#     accuracy = accuracy_score(labels, preds)
#     print(f"✅ Accuracy: {accuracy:.4f}")
#
#     if accuracy < 1.0:  # اگه خطا داشتیم
#         try:
#             reasoner = GrokReActAgent()
#             # یه نمونه خطا بگیر
#             wrong_idx = np.where(np.array(preds) != np.array(labels))[0][0]
#             wrong_payload = df.iloc[test_idx[wrong_idx]]['sentence']
#
#             exp = reasoner.reason(f"Explain XSS detection error: {wrong_payload}")
#             print(f"\n🤖 LLM Analysis:\n{exp}")
#         except:
#             print("⚠️ LLM not available")
#     return accuracy
def evaluate_model(model, test_loader, df, test_idx):
    """Evaluation"""
    model.eval()
    preds = []
    labels = []

    print("\n📊 Evaluating...")
    with torch.no_grad():
        for x, kg, y in test_loader:
            output, _ = model(x, kg_data=kg)
            preds.extend((output.squeeze() > 0.5).float().cpu().numpy())
            labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(labels, preds)
    print(f"✅ Accuracy: {accuracy:.4f}")

    if accuracy < 1.0:  # اگه خطا داشتیم
        try:
            reasoner = GrokReActAgent()
            # یه نمونه خطا بگیر
            wrong_idx = np.where(np.array(preds) != np.array(labels))[0][0]

            # ✅ اصلاح اینجا: test_idx رو از ورودی می‌گیریم
            wrong_payload = df.iloc[test_idx[wrong_idx]]['sentence']

            exp = reasoner.reason(f"Explain XSS detection error: {wrong_payload}")
            print(f"\n🤖 LLM Analysis:\n{exp}")
        except Exception as e:
            print(f"⚠️ LLM not available: {e}")

    return accuracy


def main():
    print("=" * 60)
    print("🛡️  XSS Detection - NeuroSymGen (Simple)")
    print("=" * 60)

    # 1. Load dataset
    print("\n📂 Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} samples")

    # 2. Create vocab
    vocab = create_vocab_from_dataset(df, "sentence")
    print(f"✅ Vocab size: {len(vocab)}")

    # 3. Encode all data
    X = torch.stack([encode_text(text, vocab) for text in df["sentence"]])
    y = torch.tensor(df["Label"].values, dtype=torch.float32)
    kg_data = torch.stack([create_kg_generic(text) for text in df["sentence"]])

    # 4. Split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # 5. Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[train_idx], kg_data[train_idx], y[train_idx]),
        batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X[test_idx], kg_data[test_idx], y[test_idx]),
        batch_size=BATCH_SIZE
    )

    # 6. Model
    # model = NeuroSymGenLayer(input_size=MAX_LEN, num_rules=4, output_size=1)
    # model = NeuroSymGenLayer(input_size=MAX_LEN, num_rules=4, output_size=128)
    model = NeuroSymGenLayer(
        input_size=512,
        num_rules=4,
        output_size=128,  # internal size (باید قابل تقسیم بر num_heads)
        num_heads=8,  # یا 1, 2, 4, 8, 16
        final_output_size=1  # این مهمترین پارامتره
    )
    if torch.cuda.is_available():
        model = model.cuda()

    # 7. Train
    train_model(model, train_loader)

    # 8. Evaluate
    evaluate_model(model, test_loader , df, test_idx)

    # 9. Hardware benchmark
    if torch.cuda.is_available():
        x_sample = X[0].unsqueeze(0).cuda()
        kg_sample = kg_data[0].cuda()

        optimizer = HardwareOptimizer(model, device="cuda")
        opt_model = optimizer.optimize(x_sample)
        bench = optimizer.benchmark({"x": x_sample, "kg_data": kg_sample})

        print(f"\n⚡ Throughput: {bench['throughput_qps']:.2f} QPS")
        print(f"⚡ Latency: {bench['latency_ms']:.2f} ms")

    # 10. Save
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": {"max_len": MAX_LEN}
    }, "xss_simple.pt")
    print("\n💾 Saved to xss_simple.pt")


if __name__ == "__main__":
    main()