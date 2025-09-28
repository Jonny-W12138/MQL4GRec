import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PositionEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int, device: torch.device):
        positions = torch.arange(seq_len, device=device)
        return self.pos_emb(positions)  # [L, D]


class SASRec(nn.Module):
    """
    SASRec with multimodal fusion
    - item embedding: nn.Embedding(n_items, d_model)
    - multimodal reconstructed features: text/image -> project to d_model and fuse
    Fusion options: sum or gated sum
    """
    def __init__(
        self,
        n_items: int,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        max_len: int = 200,
        dropout: float = 0.2,
        fusion: str = "gated_sum",  # "sum" or "gated_sum"
        use_text: bool = True,
        use_image: bool = True,
        text_dim: int = 256,  # reconstructed text feature dim
        image_dim: int = 256,  # reconstructed image feature dim
    ):
        super().__init__()
        self.n_items = n_items
        self.d_model = d_model
        self.max_len = max_len
        self.use_text = use_text
        self.use_image = use_image
        self.fusion = fusion

        self.item_emb = nn.Embedding(n_items, d_model, padding_idx=0)
        self.pos_emb = PositionEmbedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

        # projections for modalities
        if self.use_text:
            self.text_proj = nn.Sequential(nn.Linear(text_dim, d_model), GELU(), nn.Dropout(dropout))
        if self.use_image:
            self.image_proj = nn.Sequential(nn.Linear(image_dim, d_model), GELU(), nn.Dropout(dropout))

        if self.fusion == "gated_sum":
            # gate computed from concatenated available embeddings
            gate_in = d_model * (1 + int(self.use_text) + int(self.use_image))
            self.gate = nn.Sequential(nn.Linear(gate_in, d_model), nn.Sigmoid())
        elif self.fusion == "sum":
            self.gate = None
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

        # output layer doesn't need parameters; we use dot-product with item embeddings
        self.norm = nn.LayerNorm(d_model)

    def _fuse(self, item_vec: torch.Tensor, text_vec: torch.Tensor = None, image_vec: torch.Tensor = None):
        parts = [item_vec]
        if self.use_text and text_vec is not None:
            parts.append(text_vec)
        if self.use_image and image_vec is not None:
            parts.append(image_vec)

        if self.fusion == "sum":
            fused = torch.stack(parts, dim=0).sum(dim=0)
        else:
            cat = torch.cat(parts, dim=-1)
            g = self.gate(cat)
            fused = item_vec
            if self.use_text and text_vec is not None:
                fused = fused + g * text_vec
            if self.use_image and image_vec is not None:
                fused = fused + g * image_vec
        return fused

    def forward(self, seq_items: torch.Tensor, seq_text: torch.Tensor = None, seq_image: torch.Tensor = None, attn_mask: torch.Tensor = None):
        """
        Inputs:
          - seq_items: [B, L] item ids (0 as padding)
          - seq_text:  [B, L, mm_dim] reconstructed text feature per item (optional)
          - seq_image: [B, L, mm_dim] reconstructed image feature per item (optional)
          - attn_mask: [B, L] boolean mask for padding positions
        Returns:
          - h: [B, L, D] encoded sequence hidden states
        """
        B, L = seq_items.shape
        device = seq_items.device
        x_item = self.item_emb(seq_items)  # [B, L, D]
        x_text = None
        x_image = None
        if self.use_text and seq_text is not None:
            x_text = self.text_proj(seq_text)  # [B, L, D]
        if self.use_image and seq_image is not None:
            x_image = self.image_proj(seq_image)  # [B, L, D]

        x = self._fuse(x_item, x_text, x_image)  # [B, L, D]
        pos = self.pos_emb(L, device).unsqueeze(0)  # [1, L, D]
        x = x + pos
        x = self.dropout(x)
        # create key_padding_mask: True for padding positions
        key_padding_mask = attn_mask if attn_mask is not None else (seq_items == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [B, L, D]
        h = self.norm(h)
        return h

    def predict_next(self, h_last: torch.Tensor, all_item_emb: torch.Tensor, all_text: torch.Tensor = None, all_image: torch.Tensor = None):
        """
        h_last: [B, D] last hidden
        all_item_emb: [N, D]
        all_text: [N, D] projected
        all_image: [N, D] projected
        Returns: logits [B, N]
        """
        item_vec = all_item_emb  # [N, D]
        text_vec = all_text if (self.use_text and all_text is not None) else None
        image_vec = all_image if (self.use_image and all_image is not None) else None

        if self.fusion == "sum":
            fused_items = item_vec.clone()
            if text_vec is not None:
                fused_items = fused_items + text_vec
            if image_vec is not None:
                fused_items = fused_items + image_vec
        else:
            parts = [item_vec]
            if text_vec is not None:
                parts.append(text_vec)
            if image_vec is not None:
                parts.append(image_vec)
            cat = torch.cat(parts, dim=-1)  # [N, D*k]
            g = self.gate(cat)  # [N, D]
            fused_items = item_vec
            if text_vec is not None:
                fused_items = fused_items + g * text_vec
            if image_vec is not None:
                fused_items = fused_items + g * image_vec

        logits = h_last @ fused_items.t()  # [B, N]
        return logits

    def get_all_item_embeddings(self):
        return self.item_emb.weight  # [N, D]

    def project_modalities_all(self, text_feats: torch.Tensor = None, image_feats: torch.Tensor = None):
        """
        Project precomputed reconstructed modality features for all items.
        text_feats/image_feats: [N, mm_dim]
        Returns projected tensors or None
        """
        text_proj = self.text_proj(text_feats) if (self.use_text and text_feats is not None) else None
        image_proj = self.image_proj(image_feats) if (self.use_image and image_feats is not None) else None
        return text_proj, image_proj