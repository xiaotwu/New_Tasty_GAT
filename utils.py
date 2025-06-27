import os
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms

from transformers import BertTokenizer, BertModel
from datasets import Dataset
import networkx as nx
from community import community_louvain
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import faiss

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

class DataPreprocessor:
    """Data preprocessing class"""

    def __init__(self, json_path: str, sample_frac: float = 0.2):
        self.json_path = json_path
        self.sample_frac = sample_frac

    def load_data(self) -> Dataset:
        """Load and preprocess data"""
        try:
            with open(self.json_path) as f:
                data = [json.loads(line) for line in tqdm(f, desc="Loading JSON data")]

            # Extract required fields
            rows = [{
                'business_id': entry['business_id'],
                'user_id': entry['user_id'],
                'rating': entry['rating'],
                'review_text': entry.get('review_text', ''),
                'pic_url': pic['url'][0]
            } for entry in tqdm(data, desc="Processing entries")
              for pic in entry.get('pics', [])]

            df = pd.DataFrame(rows)
            if self.sample_frac < 1.0:
                df = df.sample(frac=self.sample_frac, random_state=SEED)
                df = df.reset_index(drop=True)

            self._print_statistics(df)
            return Dataset.from_pandas(df)

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _print_statistics(self, df: pd.DataFrame) -> None:
        """Print dataset statistics"""
        print("\nDataset Statistics:")
        print(f"Total samples: {len(df):,}")
        print(f"Unique users: {df['user_id'].nunique():,}")
        print(f"Unique businesses: {df['business_id'].nunique():,}")
        print("\nRating Distribution:")
        rating_dist = df['rating'].value_counts().sort_index()
        sns.barplot(x=rating_dist.index, y=rating_dist.values)
        plt.title("Rating Distribution")
        plt.show()

# Load data
preprocessor = DataPreprocessor('./dataset/image_review_all.json')
dataset = preprocessor.load_data()

class FeatureExtractor:
    """Feature extractor"""

    def __init__(self, image_dir: str, cache_dir: str):
        self.image_dir = image_dir
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize models
        self.init_models()

    def init_models(self):
        """Initialize feature extraction models"""
        # Image transformations
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # ResNet model
        self.resnet = nn.Sequential(
            *list(models.resnet18(weights='IMAGENET1K_V1').children())[:-1]
        ).to(DEVICE).eval()

        # BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(DEVICE).eval()

    @torch.no_grad()
    def extract_image_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract image features"""
        cache_path = os.path.join(self.cache_dir, 'image_features.npz')

        if os.path.exists(cache_path):
            print("Loading cached image features...")
            return np.load(cache_path)['features']

        print("Extracting image features...")
        features = []
        for idx in tqdm(range(len(df))):
            image_path = os.path.join(self.image_dir, f"{idx}.jpg")
            try:
                image = Image.open(image_path).convert('RGB')
                image_tensor = self.image_transform(image).unsqueeze(0).to(DEVICE)
                feat = self.resnet(image_tensor).view(-1).cpu().numpy()
            except:
                feat = np.zeros(512)
            features.append(feat)

        features = np.stack(features)
        np.savez_compressed(cache_path, features=features)
        return features

    @torch.no_grad()
    def extract_text_features(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Extract text features"""
        cache_path = os.path.join(self.cache_dir, 'text_features.npz')

        if os.path.exists(cache_path):
            print("Loading cached text features...")
            return np.load(cache_path)['features']

        print("Extracting text features...")
        features = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True,
                                   max_length=256, return_tensors='pt')
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = self.bert(**inputs).last_hidden_state[:, 0]
            features.append(outputs.cpu().numpy())

        features = np.vstack(features)
        np.savez_compressed(cache_path, features=features)
        return features

# Extract features
extractor = FeatureExtractor(image_dir, './cache')
image_features = extractor.extract_image_features(df)
text_features = extractor.extract_text_features(df['review_text'].tolist())

# Merge features
meta_features = np.hstack([text_features, image_features])
print(f"\nFinal feature dimension: {meta_features.shape}")

# # Constructe the intra-community adjacency matrix
def build_topk_neighbors_faiss(meta_array, df, topk):
    meta_array = meta_array.astype('float32')
    topk_neighbors = np.zeros((len(df), topk), dtype=int)

    for c in df['community_idx'].unique():
        idx = df[df['community_idx'] == c].index.tolist()
        if len(idx) <= topk:
            continue

        local_feats = meta_array[idx]

        # Normalize features
        faiss.normalize_L2(local_feats)

        # Build index
        d = local_feats.shape[1]
        index = faiss.IndexFlatIP(d)
        # If already used faiss-gpu, use following code to build GPU index
        # res = faiss.StandardGpuResources()
        # index = faiss.index_cpu_to_gpu(res, 0, index)

        # Add to index
        index.add(local_feats)

        # Search
        D, I = index.search(local_feats, topk + 1)  # include itself
        I = I[:, 1:]  # remove self-match (first column)

        for i, idx_i in enumerate(idx):
            topk_neighbors[idx_i, :] = np.array(idx)[I[i]]

    return topk_neighbors

topk = 10
topk_neighbors = build_topk_neighbors_faiss(meta_features, df, topk)

# Extract community tag keywords
community_keywords = {}
for comm in df['community_idx'].unique():
    texts = df[df['community_idx'] == comm]['review_text']
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf = vec.fit_transform(texts)
    top_words = np.argsort(tfidf.sum(axis=0)).tolist()[0][-10:][::-1]
    keywords = [vec.get_feature_names_out()[i] for i in top_words]
    community_keywords[comm] = keywords

print("\nTop Keywords per Community:")
for k, v in community_keywords.items():
    print(f"Community {k}: {', '.join(v)}")


# Visualize item embedding distribution
item_labels = df['community_idx'].values
ratings = df['rating'].values
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(item_emb_matrix.cpu())

plt.figure(figsize=(30, 30))
sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=item_labels, size=ratings,
                sizes=(10, 50), palette='tab10', alpha=0.8)
plt.title("Item Embedding Visualization by Community")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.legend(title='Community', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Train and validation
# Prepare Dataset (Saved on CPU)
train_idx, val_idx = train_test_split(np.arange(len(label_tensor)), test_size=0.2, random_state=42)

train_ds = TensorDataset(
    user_tensor[train_idx].cpu(),
    item_tensor[train_idx].cpu(),
    comm_tensor[train_idx].cpu(),
    meta_tensor[train_idx].cpu(),
    label_tensor[train_idx].cpu()
)
val_ds = TensorDataset(
    user_tensor[val_idx].cpu(),
    item_tensor[val_idx].cpu(),
    comm_tensor[val_idx].cpu(),
    meta_tensor[val_idx].cpu(),
    label_tensor[val_idx].cpu()
)

train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, pin_memory=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, pin_memory=True, num_workers=0)


# Plot Training Curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(1, 2, 2)
plt.plot(val_rmses, marker='o', color='orange')
plt.title('Validation RMSE Curve')
plt.xlabel('Epoch')
plt.ylabel('RMSE')

plt.tight_layout()
plt.show()


def precision_at_k(ranked_items, ground_truth_items, k):
    ranked_items_k = ranked_items[:k]
    hits = len(set(ranked_items_k) & set(ground_truth_items))
    return hits / k

def recall_at_k(ranked_items, ground_truth_items, k):
    ranked_items_k = ranked_items[:k]
    hits = len(set(ranked_items_k) & set(ground_truth_items))
    return hits / len(ground_truth_items) if len(ground_truth_items) > 0 else 0.0

def ndcg_at_k(ranked_items, ground_truth_items, k):
    ranked_items_k = ranked_items[:k]
    dcg = 0.0
    for i, item in enumerate(ranked_items_k):
        if item in ground_truth_items:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(ground_truth_items), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg

def evaluate_model_user_item_sampled(
    model,
    user_tensor,
    item_tensor,
    comm_tensor,
    meta_tensor,
    label_tensor,
    K=50,
    user_sample_size=10000,
    max_neg_per_user=1000,
    rating_threshold=3
):
    model.eval()

    if label_tensor.max() <= 1.0:
        print("[Warning] label_tensor appears normalized (max <= 1). Please ensure you are passing original ratings!")

    valid_users = user_tensor[label_tensor >= rating_threshold].unique().cpu().numpy()
    sampled_users = np.random.choice(valid_users, size=min(user_sample_size, len(valid_users)), replace=False)

    precision_list, recall_list, ndcg_list = [], [], []

    all_item_ids = item_tensor.cpu().numpy()
    print(f"Evaluating performance (user_sample={len(sampled_users)}, max_neg_per_user={max_neg_per_user}, threshold={rating_threshold})...")

    empty_user_count = 0

    for u in tqdm(sampled_users, desc="Evaluating Users"):
        ground_truth_items = item_tensor[(user_tensor == u) & (label_tensor >= rating_threshold)].cpu().numpy()

        if len(ground_truth_items) == 0:
            empty_user_count += 1
            continue

        # Dynamic negative sampling number = number of positive samples × 50, capped to max_neg_per_user
        neg_num = min(len(ground_truth_items) * 50, max_neg_per_user)

        neg_pool = list(set(all_item_ids) - set(ground_truth_items))
        if len(neg_pool) == 0:
            continue  # Prevent the negative sample pool from being empty

        sampled_negatives = np.random.choice(neg_pool, size=min(neg_num, len(neg_pool)), replace=False)

        candidates = np.concatenate([ground_truth_items, sampled_negatives])
        candidates_tensor = torch.tensor(candidates, dtype=torch.long)

        u_batch = torch.full((len(candidates_tensor),), u, dtype=torch.long).to(DEVICE)
        i_batch = candidates_tensor.to(DEVICE)
        c_batch = comm_tensor[candidates_tensor].to(DEVICE)
        meta_batch = meta_tensor[candidates_tensor].to(DEVICE)

        with torch.no_grad():
            scores = model(u_batch, i_batch, c_batch, meta_batch).cpu().numpy()

        ranked_indices = np.argsort(-scores)
        ranked_candidates = candidates[ranked_indices]

        precision = precision_at_k(ranked_candidates, ground_truth_items, K)
        recall = recall_at_k(ranked_candidates, ground_truth_items, K)
        ndcg = ndcg_at_k(ranked_candidates, ground_truth_items, K)

        precision_list.append(precision)
        recall_list.append(recall)
        ndcg_list.append(ndcg)

    if len(precision_list) == 0:
        print("[Warning] No valid users with ground truth found! Cannot compute metrics.")
        return np.nan, np.nan, np.nan

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)

    print(f"\nSampled Evaluation Results (K={K}):")
    print(f"Precision@{K}: {avg_precision:.4f}")
    print(f"Recall@{K}: {avg_recall:.4f}")
    print(f"NDCG@{K}: {avg_ndcg:.4f}")
    print(f"Skipped {empty_user_count} users with no valid ground truth.")

    return avg_precision, avg_recall, avg_ndcg, precision_list, recall_list, ndcg_list


precision, recall, ndcg, precision_list, recall_list, ndcg_list = evaluate_model_user_item_sampled(
    model,
    user_tensor,
    item_tensor,
    comm_tensor,
    meta_tensor,
    torch.tensor(df['rating'].values),  # Notice: the original rating is passed
    K=50,
    user_sample_size=10000,
    max_neg_per_user=1000,  # Dynamic negative sampling upper limit
    rating_threshold=3
)


# Plot evaluation results
plt.figure(figsize=(18,5))

plt.subplot(1,3,1)
plt.hist(precision_list, bins=50, color='skyblue')
plt.title("Precision@K Distribution")
plt.xlabel("Precision")
plt.ylabel("Number of Users")

plt.subplot(1,3,2)
plt.hist(recall_list, bins=50, color='lightgreen')
plt.title("Recall@K Distribution")
plt.xlabel("Recall")
plt.ylabel("Number of Users")

plt.subplot(1,3,3)
plt.hist(ndcg_list, bins=50, color='lightcoral')
plt.title("NDCG@K Distribution")
plt.xlabel("NDCG")
plt.ylabel("Number of Users")

plt.tight_layout()
plt.show()