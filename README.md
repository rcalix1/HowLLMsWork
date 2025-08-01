## How LLMs Work

* AGS
* How LLMs work under the hood

## Link

* https://github.com/rcalix1/DeepLearningAlgorithms/tree/main/SecondEdition/Chapter10_Transformers/GPTs
* 


# 🧠 LLMs Under the Hood: Understanding Attention in Transformers

This repository walks through the **Attention Mechanism** at the core of Transformer models like GPT, BERT, and LLaMA — implemented from scratch in PyTorch.

## 📌 Objective

We implement step-by-step how **Queries (Q)**, **Keys (K)**, and **Values (V)** are computed from input embeddings, how attention scores are generated, and how causal masking ensures autoregressive behavior.

---

## 🔄 Workflow Overview

### 1. Input Tensor

We simulate a batch of 32 sequences, each with 40 tokens, and an embedding dimension of 512:

```python
x = torch.randn(32, 40, 512)
