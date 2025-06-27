# New_Tasty_GAT
This module builds a personalized restaurant recommender using a multimodal Graph Attention Network with community detection.

## Overview
We integrate user-restaurant interaction graphs with BERT-based review embeddings and ResNet-18 visual features. GAT propagates attention within detected communities and learns rich contextualized restaurant representations.

## Pipeline Highlights

1. **Multimodal Feature Extraction**:
   - BERT (text), ResNet-18 (image)

2. **Community Detection**:
   - Louvain algorithm to detect user/restaurant clusters

3. **Graph Attention Network**:
   - Tokenize neighbors, apply multi-head attention
   - Aggregate attention-weighted features

4. **Multimodal Attention Fusion**:
   - Cross-modal attention (Text ⟷ Image + Graph Embedding)

5. **Rating Prediction**:
   - MLP with fused user + item + community + attention vectors

6. **Evaluation**
    - RMSE
    - Recall@50
    - NDCG@50

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install datasets transformers scikit-learn networkx python-louvain tqdm seaborn matplotlib pillow pandas numpy requests faiss-cpu
```

## Original Project

[`New Tasty`](https://github.com/zhoumiaosen/New_Tasty)