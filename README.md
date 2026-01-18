# Tokenization Practice

This repository contains a Jupyter Notebook implementing and practicing subword tokenization techniques. The implementation explores Byte Pair Encoding (BPE) from scratch and compares it with production-level libraries.

## Features

* **Manual BPE Implementation**: Functions to count byte pairs, merge them, and encode/decode text sequences.
* **Tiktoken Integration**: Comparisons between GPT-2 and cl100k_base (GPT-4) encoding schemes.
* **SentencePiece Training**: Steps to train a custom BPE model using a toy dataset with specific vocabulary constraints.

## Core BPE Logic

The manual merge process identifies the most frequent adjacent byte pairs and replaces them with a new token ID:

```python
def Byt(token):
    count = {}
    for pair in zip(token, token[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count

# Merging logic
stats = Byt(tokens)
top_pair = max(stats, key=stats.get)
# Iterative merging follows to reduce sequence length
