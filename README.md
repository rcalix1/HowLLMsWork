## How LLMs Work

* AGS
* How LLMs work under the hood

## Link

* https://github.com/rcalix1/DeepLearningAlgorithms/tree/main/SecondEdition/Chapter10_Transformers/GPTs
* 


# 🧠 LLMs Under the Hood: Understanding Attention in Transformers

# 🧠 LLMs Under the Hood: Understanding Attention in Transformers

This repository walks through the Attention Mechanism at the core of Transformer models like GPT, BERT, and LLaMA — implemented from scratch in PyTorch.

## 📌 Objective

We implement step-by-step how Queries (Q), Keys (K), and Values (V) are computed from input embeddings, how attention scores are generated, and how causal masking ensures autoregressive behavior.

## 🔄 Workflow Overview

1. Input Tensor

We simulate a batch of 32 sequences, each with 40 tokens, and an embedding dimension of 512:

x = torch.randn(32, 40, 512)

2. Compute Q, K, V Projections

Each token is projected into lower-dimensional Q, K, and V vectors (dimension = 64) using learned weights and biases:

Q = x @ wq + bq  
K = x @ wk + bk  
V = x @ wv + bv

3. Compute Attention Scores

attention_scores = Q @ K.transpose(-2, -1)  # shape: (32, 40, 40)

Each row tells us how much a token attends to every other token in the sequence.

4. Apply Causal Masking

To ensure no "peeking into the future" during training (crucial for language models):

tril = torch.tril(torch.ones(40, 40))  
attention_scores = attention_scores.masked_fill(tril == 0, float('-inf'))

5. Softmax Normalization

attention_probs = F.softmax(attention_scores, dim=-1)

This converts raw scores into probabilities.

6. Compute Weighted Output

out = attention_probs @ V

Each output is a context-aware representation, weighted by relevance to the current token.

7. Multi-Head Simulation

We simulate 8 attention heads by copying and concatenating:

out_cat = torch.cat([out] * 8, dim=-1)

8. Final Output Projection

The concatenated output is projected back to the original embedding size:

z = out_cat @ w0 + b0

## 🧠 Key Concepts

- Self-Attention: Every token can attend to every other token in the same sequence.
- Causal Masking: Used in decoder-only models to preserve left-to-right generation.
- Multi-Head Attention: Enables the model to attend to different representation subspaces simultaneously.
- Projection Weights: Learned parameters that help the model abstract meaningful context.

## 📦 Requirements

- Python 3.8+
- PyTorch

pip install torch

## 🧪 Run the Code

Just launch the notebook or run the script directly:

python attention_demo.py

## 📚 Suggested Readings

- Attention is All You Need: https://arxiv.org/abs/1706.03762
- The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT

## 🎓 Author

This material is part of the "LLMs Under the Hood" masterclass — a 90-minute deep dive for developers exploring how Transformers really work.

