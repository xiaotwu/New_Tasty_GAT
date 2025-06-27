import torch
import torch.nn as nn

class RecommenderWithGATCommunity(nn.Module):
    def __init__(self, n_users, item_emb_matrix, n_comms, meta_dim):
        super().__init__()
        self.item_emb_matrix = item_emb_matrix
        self.u_emb = nn.Embedding(n_users, 32)
        self.c_emb = nn.Embedding(n_comms, 8)
        self.text_proj = nn.Linear(meta_dim // 2, meta_dim)
        self.img_proj = nn.Linear(meta_dim // 2, meta_dim)
        self.attn = nn.MultiheadAttention(meta_dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(meta_dim)
        self.mlp = nn.Sequential(
            nn.Linear(32 + 32 + 8 + meta_dim + meta_dim, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, u, i, c, meta):
        u_vec = self.u_emb(u)
        i_vec = self.item_emb_matrix[i]
        c_vec = self.c_emb(c)
        t, v = meta[:, :meta.shape[1]//2], meta[:, meta.shape[1]//2:]
        t_proj = self.text_proj(t)
        v_proj = self.img_proj(v)
        attn_out, _ = self.attn(t_proj.unsqueeze(1), v_proj.unsqueeze(1), v_proj.unsqueeze(1))
        attn_out = self.norm(attn_out.squeeze(1) + t_proj)
        x = torch.cat([u_vec, i_vec, c_vec, t_proj, attn_out], dim=1)
        return self.mlp(x).squeeze()
