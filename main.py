from data_loader import DataPreprocessor
from model import RecommenderWithGATCommunity
from utils import DEVICE, evaluate_model_user_item_sampled
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os

def main():
    # 1. Load and preprocess data
    preprocessor = DataPreprocessor('./dataset/image_review_all.json')
    dataset = preprocessor.load_data()
    df = dataset.to_pandas()

    # 2. Load preprocessed tensors
    meta_tensor = torch.tensor(np.load("./cache/meta_features.npy"), dtype=torch.float32).to(DEVICE)
    user_tensor = torch.tensor(np.load("./cache/user_tensor.npy"))
    item_tensor = torch.tensor(np.load("./cache/item_tensor.npy"))
    comm_tensor = torch.tensor(np.load("./cache/comm_tensor.npy"))
    label_tensor = torch.tensor(np.load("./cache/label_tensor.npy"), dtype=torch.float32)
    item_emb_matrix = torch.tensor(np.load("./cache/item_emb_matrix.npy"), dtype=torch.float32).to(DEVICE)

    # 3. Build DataLoader
    train_idx, val_idx = train_test_split(np.arange(len(label_tensor)), test_size=0.2, random_state=42)
    train_ds = TensorDataset(user_tensor[train_idx], item_tensor[train_idx], comm_tensor[train_idx], meta_tensor[train_idx], label_tensor[train_idx])
    val_ds = TensorDataset(user_tensor[val_idx], item_tensor[val_idx], comm_tensor[val_idx], meta_tensor[val_idx], label_tensor[val_idx])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

    # 4. Initialize the model
    model = RecommenderWithGATCommunity(
        n_users=user_tensor.max().item() + 1,
        item_emb_matrix=item_emb_matrix,
        n_comms=comm_tensor.max().item() + 1,
        meta_dim=meta_tensor.shape[1]
    ).to(DEVICE)

    # 5. Define loss function, optimizer, and scaler
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    # 6. Training loop with early stopping
    best_rmse = float('inf')
    patience = 3
    trigger_times = 0

    for epoch in range(1, 51):
        model.train()
        total_loss = 0

        for u, i, c, meta, y in train_loader:
            u, i, c, meta, y = u.to(DEVICE), i.to(DEVICE), c.to(DEVICE), meta.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda'):
                pred = model(u, i, c, meta)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate RMSE
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for u, i, c, meta, y in val_loader:
                u, i, c, meta, y = u.to(DEVICE), i.to(DEVICE), c.to(DEVICE), meta.to(DEVICE), y.to(DEVICE)
                out = model(u, i, c, meta).cpu().numpy()
                preds.append(out)
                truths.append(y.cpu().numpy())

        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        rmse = np.sqrt(np.mean((preds - truths) ** 2))

        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Val RMSE: {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            trigger_times = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    # 7. Evaluate
    print("Running evaluation on validation set...")
    evaluate_model_user_item_sampled(
        model,
        user_tensor,
        item_tensor,
        comm_tensor,
        meta_tensor,
        torch.tensor(df['rating'].values),  # Original ratings
        K=50
    )

if __name__ == "__main__":
    main()
