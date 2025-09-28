import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import prepare_datasets, collate_fn
from model import SASRec
from utils import set_seed, hr_at_k, ndcg_at_k, compute_rank


def train_epoch(model: SASRec, loader: DataLoader, device: torch.device, optimizer: optim.Optimizer, n_items: int, mm_text_all=None, mm_img_all=None):
    model.train()
    total_loss = 0.0

    all_item_emb = model.get_all_item_embeddings()  # [N, D]
    all_text_proj = mm_text_all
    all_img_proj = mm_img_all

    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        seq_items = batch["seq_items"].to(device)       # [B, L]
        attn_mask = batch["attn_mask"].to(device)       # [B, L]
        target = batch["target"].to(device)             # [B]
        seq_text = batch["seq_text"].to(device) if batch["seq_text"] is not None else None
        seq_image = batch["seq_image"].to(device) if batch["seq_image"] is not None else None

        h = model(seq_items, seq_text, seq_image, attn_mask)  # [B, L, D]
        # last non-padding position
        last_idx = (seq_items != 0).sum(dim=1) - 1  # [B]
        h_last = h[torch.arange(h.size(0), device=device), last_idx]  # [B, D]

        logits = model.predict_next(h_last, all_item_emb, all_text_proj, all_img_proj)  # [B, N]
        loss = nn.CrossEntropyLoss()(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=float(loss.item()))

    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_epoch(model: SASRec, loader: DataLoader, device: torch.device, k: int = 10, mm_text_all=None, mm_img_all=None):
    model.eval()
    ranks_all = []

    all_item_emb = model.get_all_item_embeddings()
    all_text_proj = mm_text_all
    all_img_proj = mm_img_all

    for batch in loader:
        seq_items = batch["seq_items"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        target = batch["target"].to(device)
        seq_text = batch["seq_text"].to(device) if batch["seq_text"] is not None else None
        seq_image = batch["seq_image"].to(device) if batch["seq_image"] is not None else None

        h = model(seq_items, seq_text, seq_image, attn_mask)
        last_idx = (seq_items != 0).sum(dim=1) - 1
        h_last = h[torch.arange(h.size(0), device=device), last_idx]

        logits = model.predict_next(h_last, all_item_emb, all_text_proj, all_img_proj)
        # mask historical items in each user's sequence (exclude padding 0 and the target itself)
        for i in range(seq_items.size(0)):
            hist = seq_items[i]
            uniq = torch.unique(hist[hist != 0])
            # keep target
            uniq = uniq[uniq != target[i]]
            if uniq.numel() > 0:
                logits[i, uniq] = float("-inf")
        ranks = compute_rank(logits, target)
        ranks_all.extend(ranks)

    hr = hr_at_k(ranks_all, k)
    ndcg = ndcg_at_k(ranks_all, k)
    return hr, ndcg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default='data/Instruments/Instruments.train.inter')
    parser.add_argument("--valid", type=str, default='data/Instruments/Instruments.valid.inter')
    parser.add_argument("--test", type=str, default='data/Instruments/Instruments.test.inter')
    parser.add_argument("--lemb", type=str, default='data/Instruments_new/Instruments_new.index_lemb.json')
    parser.add_argument("--vitmb", type=str, default='data/Instruments_new/Instruments_new.index_vitmb.json')
    parser.add_argument("--text_ckpt", type=str, default='log/Instruments/llama_256/best_collision_model.pth')
    parser.add_argument("--image_ckpt", type=str, default='log/Instruments/ViT-L-14_256/best_collision_model.pth')
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--fusion", type=str, default="gated_sum", choices=["sum", "gated_sum"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mm_dim", type=int, default=256)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    train_ds, valid_ds, test_ds, n_items = prepare_datasets(
        args.train, args.valid, args.test,
        args.max_len,
        args.lemb, args.vitmb,
        args.text_ckpt, args.image_ckpt,
        device
    )

    # Precompute all modality features for ranking efficiency (optional)
    use_text = args.lemb is not None and args.text_ckpt is not None
    use_image = args.vitmb is not None and args.image_ckpt is not None

    # 动态确定文本/图像重建维度
    text_dim = None
    image_dim = None
    if use_text or use_image:
        from rqvae import RQVAEReconstructor
        text_recon_tmp = RQVAEReconstructor(args.text_ckpt, device) if use_text else None
        img_recon_tmp = RQVAEReconstructor(args.image_ckpt, device) if use_image else None
        text_dim = text_recon_tmp.emb_dim if use_text else args.mm_dim
        image_dim = img_recon_tmp.emb_dim if use_image else args.mm_dim

    model = SASRec(
        n_items=n_items,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
        dropout=args.dropout,
        fusion=args.fusion,
        use_text=use_text,
        use_image=use_image,
        text_dim=(text_dim if text_dim is not None else args.mm_dim),
        image_dim=(image_dim if image_dim is not None else args.mm_dim),
    ).to(device)

    # Build all-items projected modality matrices, if any
    mm_text_all = None
    mm_img_all = None
    if use_text or use_image:
        # reconstruct features for all item ids [0..n_items-1] without gradients
        from rqvae import RQVAEReconstructor, load_item_codes
        text_map = load_item_codes(args.lemb) if use_text else {}
        img_map = load_item_codes(args.vitmb) if use_image else {}
        text_recon = RQVAEReconstructor(args.text_ckpt, device) if use_text else None
        img_recon = RQVAEReconstructor(args.image_ckpt, device) if use_image else None

        with torch.no_grad():
            if use_text:
                text_feats = []
                for i in range(n_items):
                    codes = text_map.get(i)
                    vec = text_recon.reconstruct_from_codes(codes) if codes is not None else torch.zeros(text_recon.emb_dim, device=device)
                    text_feats.append(vec)
                text_feats = torch.stack(text_feats, dim=0)  # [N, raw_dim]
                mm_text_all, _ = model.project_modalities_all(text_feats=text_feats)
                mm_text_all = mm_text_all.detach()

            if use_image:
                img_feats = []
                for i in range(n_items):
                    codes = img_map.get(i)
                    vec = img_recon.reconstruct_from_codes(codes) if codes is not None else torch.zeros(img_recon.emb_dim, device=device)
                    img_feats.append(vec)
                img_feats = torch.stack(img_feats, dim=0)  # [N, raw_dim]
                _, mm_img_all = model.project_modalities_all(image_feats=img_feats)
                mm_img_all = mm_img_all.detach()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    print(f"Items: {n_items}, Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")
    # 创建优化器（仅创建一次，跨epoch保持状态）
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, device, optimizer, n_items, mm_text_all, mm_img_all)
        hr_v, ndcg_v = eval_epoch(model, valid_loader, device, 10, mm_text_all, mm_img_all)
        print(f"Epoch {ep}: loss={loss:.4f} | Valid HR@10={hr_v:.4f} NDCG@10={ndcg_v:.4f}")

    hr_t, ndcg_t = eval_epoch(model, test_loader, device, 10, mm_text_all, mm_img_all)
    print(f"Test HR@10={hr_t:.4f} NDCG@10={ndcg_t:.4f}")


if __name__ == "__main__":
    main()