## Baby BERT

# 🧠 Baby BERT from Scratch

This project demonstrates a **minimal BERT-style Transformer** built from scratch in PyTorch — ideal for understanding the **encoder-based** architecture that powers models like BERT.

---

## 📌 Objective

Build and run a **minimal encoder-only Transformer block** for demonstration and educational purposes:

* Input: token IDs
* Output: contextual embeddings or masked token predictions
* Includes: token & position embeddings, self-attention, feedforward, and layer normalization
* Also includes BERT's two pretraining objectives:
  - Masked Language Modeling (MLM)
  - Next Sentence Prediction (NSP)

---

## 🧾 BERT vs GPT: Core Difference

| Feature         | GPT                        | BERT                      |
|----------------|-----------------------------|---------------------------|
| Architecture    | Decoder-only                | Encoder-only              |
| Attention Mask | Causal (left to right)      | Bidirectional (full)      |
| Use Case       | Generation                  | Classification, QA, etc.  |

---

## 💾 Code Overview (Core Encoder Block)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Config ===
batch_size = 2
seq_len = 10
vocab_size = 1000
embed_dim = 64
ff_dim = 128

# === Input tokens ===
tokens = torch.randint(0, vocab_size, (batch_size, seq_len))  # [2, 10]

# === Embedding layers ===
token_embed = nn.Embedding(vocab_size, embed_dim)
pos_embed = nn.Embedding(seq_len, embed_dim)

x_token = token_embed(tokens)
positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
x_pos = pos_embed(positions)

x = x_token + x_pos  # [2, 10, 64]

# === Layer Norm + Self-Attention ===
norm1 = nn.LayerNorm(embed_dim)
Wq = nn.Linear(embed_dim, embed_dim)
Wk = nn.Linear(embed_dim, embed_dim)
Wv = nn.Linear(embed_dim, embed_dim)

x_norm = norm1(x)
Q = Wq(x_norm)
K = Wk(x_norm)
V = Wv(x_norm)

attn_scores = Q @ K.transpose(-2, -1) / (embed_dim ** 0.5)
attn_weights = F.softmax(attn_scores, dim=-1)
attn_output = attn_weights @ V

x = x + attn_output  # Residual

# === Layer Norm + Feedforward ===
norm2 = nn.LayerNorm(embed_dim)
ff1 = nn.Linear(embed_dim, ff_dim)
ff2 = nn.Linear(ff_dim, embed_dim)

x_norm = norm2(x)
ff_output = ff2(F.relu(ff1(x_norm)))

x = x + ff_output  # Final residual
```

---

## 🔐 Masked Language Modeling (MLM)

In MLM, you randomly mask some tokens (e.g., 15%) and train the model to predict the original token.

```python
# Sample input: "The cat sat on the [MASK]."
input_ids = torch.tensor([[101, 1996, 4937, 2938, 2006, 103, 102]])  # 103 = [MASK]

# Model tries to predict the missing word (e.g., "mat") at position of 103
```

You use a cross-entropy loss on the masked positions only:

```python
masked_positions = (input_ids == 103)  # Boolean mask
logits = model(input_ids)  # [batch, seq_len, vocab_size]
masked_logits = logits[masked_positions]
masked_labels = true_ids[masked_positions]
loss = F.cross_entropy(masked_logits, masked_labels)
```

---

## 🔗 Next Sentence Prediction (NSP)

In NSP, you give the model two segments: A and B. The model learns to predict if B **follows** A.

```python
# Segment A: "The cat sat on the mat."
# Segment B: "It was warm and cozy."

input_ids = torch.tensor([
  [101, 1996, 4937, 2938, 2006, 1996, 7965, 102, 2009, 2001, 4010, 1998, 10996, 102]
])
segment_ids = torch.tensor([
  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
])
```

The `[CLS]` token at the start (ID 101) is used for prediction. A simple classification head predicts `IsNext` or `NotNext`:

```python
cls_embedding = x[:, 0, :]  # Take [CLS] embedding
logits = classification_head(cls_embedding)  # [batch, 2]
loss = F.cross_entropy(logits, labels)
```

---

## 🧠 Summary of Training Tasks

| Task                  | Description                                 |
|-----------------------|---------------------------------------------|
| MLM (Masked LM)       | Predict random masked tokens                |
| NSP (Next Sentence)   | Predict whether sentence B follows sentence A |

These two objectives help BERT learn **deep bidirectional representations** of language.

---

## ✅ Output

```python
print(x.shape)  # torch.Size([2, 10, 64])
```

For NSP, the final output is:
```python
print(logits.shape)  # torch.Size([batch_size, 2])
```

---

## 📖 Next Steps

- Wrap encoder into a `nn.Module`
- Train on toy MLM + NSP dataset
- Fine-tune for classification, QA, etc.

---

🎓 About

This material is part of the "LLMs Under the Hood" masterclass by Ricardo Calix — a 90-minute session designed for engineers and data scientists who want to deeply understand how Transformers work — both decoder (GPT) and encoder (BERT) styles.
