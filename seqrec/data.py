import csv
import json
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from rqvae import RQVAEReconstructor, load_item_codes


def read_inter(path: str) -> List[Tuple[int, List[int], int]]:
    """
    Reads TSV with header:
    user_id:token \\t item_id_list:token_seq \\t item_id:token
    Returns list of (user_id, seq_items, target_item)
    """
    rows: List[Tuple[int, List[int], int]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 3:
                continue
            try:
                user_id = int(row[0])
                seq_str = row[1].strip()
                target_item = int(row[2])
                seq_items = [int(x) for x in seq_str.split(" ") if x != ""]
                rows.append((user_id, seq_items, target_item))
            except Exception:
                continue
    return rows


def build_item_set(all_rows: List[Tuple[int, List[int], int]]) -> List[int]:
    items = set()
    for _, seq, tgt in all_rows:
        for i in seq:
            items.add(i)
        items.add(tgt)
    # ensure padding id 0 exists; remap if 0 is used as real id
    if 0 not in items:
        items.add(0)
    return sorted(items)


class SeqDataset(Dataset):
    def __init__(self, rows: List[Tuple[int, List[int], int]], max_len: int, n_items: int,
                 item_to_text: Dict[int, List[int]] = None, item_to_image: Dict[int, List[int]] = None,
                 text_recon: RQVAEReconstructor = None, image_recon: RQVAEReconstructor = None,
                 device: torch.device = torch.device("cpu")):
        self.rows = rows
        self.max_len = max_len
        self.n_items = n_items
        self.item_to_text = item_to_text or {}
        self.item_to_image = item_to_image or {}
        self.text_recon = text_recon
        self.image_recon = image_recon
        self.device = device

        # cache reconstructed features for items to avoid repeated compute
        self.text_cache: Dict[int, torch.Tensor] = {}
        self.image_cache: Dict[int, torch.Tensor] = {}

    def __len__(self):
        return len(self.rows)

    def _pad_seq(self, seq: List[int]) -> List[int]:
        if len(seq) >= self.max_len:
            return seq[-self.max_len:]
        return [0] * (self.max_len - len(seq)) + seq

    def _get_mm(self, item_id: int):
        text_vec = None
        image_vec = None
        if self.text_recon is not None and item_id in self.item_to_text:
            if item_id in self.text_cache:
                text_vec = self.text_cache[item_id]
            else:
                codes = self.item_to_text[item_id]
                with torch.no_grad():
                    vec = self.text_recon.reconstruct_from_codes(codes)
                # 缓存为CPU且断开梯度
                vec = vec.detach().cpu()
                self.text_cache[item_id] = vec
                text_vec = vec
        if self.image_recon is not None and item_id in self.item_to_image:
            if item_id in self.image_cache:
                image_vec = self.image_cache[item_id]
            else:
                codes = self.item_to_image[item_id]
                with torch.no_grad():
                    vec = self.image_recon.reconstruct_from_codes(codes)
                vec = vec.detach().cpu()
                self.image_cache[item_id] = vec
                image_vec = vec
        return text_vec, image_vec

    def __getitem__(self, idx):
        user_id, seq, target = self.rows[idx]
        seq_padded = self._pad_seq(seq)
        seq_items = torch.tensor(seq_padded, dtype=torch.long)
        attn_mask = (seq_items == 0)

        # Build modality tensors [L, mm_dim]
        text_feats = None
        image_feats = None
        if self.text_recon is not None or self.image_recon is not None:
            text_list = []
            image_list = []
            for itm in seq_padded:
                tvec, ivec = self._get_mm(itm)
                text_list.append(tvec.cpu() if tvec is not None else None)
                image_list.append(ivec.cpu() if ivec is not None else None)

            # find dim from reconstructor
            if self.text_recon is not None:
                td = self.text_recon.emb_dim
                text_feats = torch.stack(
                    [tv if tv is not None else torch.zeros(td) for tv in text_list], dim=0
                )
            if self.image_recon is not None:
                idim = self.image_recon.emb_dim
                image_feats = torch.stack(
                    [iv if iv is not None else torch.zeros(idim) for iv in image_list], dim=0
                )

        return {
            "seq_items": seq_items,
            "attn_mask": attn_mask,
            "target": torch.tensor(target, dtype=torch.long),
            "seq_text": text_feats,
            "seq_image": image_feats,
        }


def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def prepare_datasets(
    train_path: str, valid_path: str, test_path: str,
    max_len: int,
    lemb_json: Optional[str],
    vitmb_json: Optional[str],
    text_ckpt: Optional[str],
    image_ckpt: Optional[str],
    device: torch.device
):
    train_rows = read_inter(train_path)
    valid_rows = read_inter(valid_path)
    test_rows = read_inter(test_path)

    all_rows = train_rows + valid_rows + test_rows
    item_list = build_item_set(all_rows)
    n_items = max(item_list) + 1  # assume contiguous starting at 0

    # load code mappings
    item_to_text = load_item_codes(lemb_json) if lemb_json else {}
    item_to_image = load_item_codes(vitmb_json) if vitmb_json else {}

    text_recon = RQVAEReconstructor(text_ckpt, device) if text_ckpt else None
    image_recon = RQVAEReconstructor(image_ckpt, device) if image_ckpt else None

    train_ds = SeqDataset(train_rows, max_len, n_items, item_to_text, item_to_image, text_recon, image_recon, device)
    valid_ds = SeqDataset(valid_rows, max_len, n_items, item_to_text, item_to_image, text_recon, image_recon, device)
    test_ds = SeqDataset(test_rows, max_len, n_items, item_to_text, item_to_image, text_recon, image_recon, device)

    return train_ds, valid_ds, test_ds, n_items