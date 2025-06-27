# Initialize the model
model = RecommenderWithGATCommunity(
    n_users=len(np.unique(user_idx)),
    item_emb_matrix=item_emb_matrix,
    n_comms=len(np.unique(comm_idx)),
    meta_dim=meta_tensor.shape[1]
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()


# Training Loop
best_rmse = float('inf')
patience = 3  # Early stopping patience
trigger_times = 0

train_losses = []
val_rmses = []

for epoch in range(1, 51):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)  # tqdm: Display training progress
    for u, i, c, meta, y in train_bar:
        u, i, c, meta, y = u.to(DEVICE), i.to(DEVICE), c.to(DEVICE), meta.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            pred = model(u, i, c, meta)
            loss = criterion(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for u, i, c, meta, y in val_bar:
            u, i, c, meta, y = u.to(DEVICE), i.to(DEVICE), c.to(DEVICE), meta.to(DEVICE), y.to(DEVICE)
            out = model(u, i, c, meta).cpu().numpy()
            preds.append(out)
            truths.append(y.cpu().numpy())

    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    rmse = np.sqrt(mean_squared_error(truths, preds))
    val_rmses.append(rmse)

    print(f"Epoch {epoch} Summary: Train Loss = {avg_train_loss:.4f}, Val RMSE = {rmse:.4f}")

    # Early Stopping
    if rmse < best_rmse:
        best_rmse = rmse
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved at epoch {epoch} with RMSE = {rmse:.4f}")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered after {patience} bad epochs.")
            break
