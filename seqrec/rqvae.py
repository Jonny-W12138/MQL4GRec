import json
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional


class SimpleDecoder(nn.Module):
    """
    Placeholder if checkpoint provides decoder weights in a simple form.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class RQVAEReconstructor:
    """
    RQ-VAE重建器，适配如下检查点结构：
    - obj['state_dict'] 含有多级 codebook：rq.vq_layers.{0..L-1}.embedding.weight
      每层维度为 e_dim（例如 32），共有 L 层（例如 4）
    - decoder MLP 权重：decoder.mlp_layers.{idx}.weight / bias
    - reconstruct_from_codes 接受长度为 L 的代码索引列表（或嵌套），逐层选取并融合
    """
    def __init__(self, ckpt_path: str, device: torch.device):
        self.device = device
        self.ckpt_path = ckpt_path
        # 多级 codebook
        self.codebooks: List[torch.Tensor] = []
        self.levels: int = 0
        self.e_dim: int = 32  # 每层embedding维度，若从权重推断则覆盖
        # decoder
        self.decoder: Optional[nn.Module] = None
        # 输出特征维度（若存在decoder，以最后线性层的out_dim为准；否则为融合后的维度）
        self.emb_dim: int = 256

        self._load_checkpoint()

    def _load_checkpoint(self):
        try:
            obj = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"[RQVAE] Failed to load checkpoint {self.ckpt_path}: {e}")
            obj = None

        state = None
        if isinstance(obj, dict):
            # 兼容不同键命名
            state = obj.get("state_dict") or obj.get("model_state_dict")
        # 收集多级codebook
        if isinstance(state, dict):
            # rq.vq_layers.{i}.embedding.weight
            code_keys = sorted([k for k in state.keys() if k.startswith("rq.vq_layers.") and k.endswith(".embedding.weight")])
            for ck in code_keys:
                wt = state[ck]
                if isinstance(wt, torch.Tensor):
                    self.codebooks.append(wt.to(self.device))
            self.levels = len(self.codebooks)
            if self.levels > 0:
                self.e_dim = int(self.codebooks[0].shape[1])

            # 构建decoder MLP：decoder.mlp_layers.{idx}.weight/bias
            # 这些idx通常为偶数索引间隔（1,4,7,...），保持排序
            lin_indices = []
            for k in state.keys():
                if k.startswith("decoder.mlp_layers.") and k.endswith(".weight"):
                    try:
                        idx = int(k.split("decoder.mlp_layers.")[1].split(".")[0])
                        lin_indices.append(idx)
                    except:
                        pass
            lin_indices = sorted(lin_indices)
            layers = []
            prev_out_dim = None
            for idx in lin_indices:
                w = state.get(f"decoder.mlp_layers.{idx}.weight")
                b = state.get(f"decoder.mlp_layers.{idx}.bias")
                if not isinstance(w, torch.Tensor) or not isinstance(b, torch.Tensor):
                    continue
                out_dim, in_dim = w.shape
                lin = nn.Linear(in_dim, out_dim)
                lin.weight.data.copy_(w)
                lin.bias.data.copy_(b)
                layers.append(lin)
                # 激活：使用GELU
                layers.append(nn.GELU())
                prev_out_dim = out_dim
            if len(layers) > 0:
                # 最后一层一般不需要激活，去掉末尾激活
                if isinstance(layers[-1], nn.GELU):
                    layers = layers[:-1]
                self.decoder = nn.Sequential(*layers).to(self.device)
                # 冻结decoder参数并设为eval，避免训练期间产生梯度
                for p in self.decoder.parameters():
                    p.requires_grad = False
                self.decoder.eval()
                self.emb_dim = prev_out_dim if prev_out_dim is not None else self.levels * self.e_dim

        # 如果没有解析到codebooks，则初始化一个退化的随机codebook以避免崩溃
        if self.levels == 0:
            print("[RQVAE] codebooks not found; initializing random 1-level codebook [K=1024, D=256]")
            self.codebooks = [torch.randn(1024, 256, device=self.device)]
            self.levels = 1
            self.e_dim = 256
            # 若没有decoder，输出维度与e_dim保持一致
            if self.decoder is None:
                self.emb_dim = self.e_dim

    def reconstruct_from_codes(self, codes: List[int]) -> torch.Tensor:
        """
        输入：
          codes: 长度为 levels 的列表（或嵌套），每层一个code索引
        策略：
          - 对每层取对应的 codebook 向量
          - 逐层相加得到融合向量（shape=[e_dim]）
          - 若存在decoder，则将融合向量作为输入，输出为重建特征（shape=[emb_dim]）
          - 若无decoder，直接返回融合向量
        """
        if codes is None or len(self.codebooks) == 0:
            return torch.zeros(self.emb_dim, device=self.device)

        # 将codes标准化为长度=levels
        code_list: List[int] = []
        if isinstance(codes, list):
            # 若是嵌套，取每层第一个索引
            for i in range(self.levels):
                if i < len(codes):
                    ci = codes[i]
                    if isinstance(ci, list) and len(ci) > 0:
                        code_list.append(int(ci[0]))
                    else:
                        try:
                            code_list.append(int(ci))
                        except:
                            code_list.append(0)
                else:
                    code_list.append(0)
        else:
            # 单值复用到各层
            try:
                val = int(codes)
            except:
                val = 0
            code_list = [val for _ in range(self.levels)]

        # 逐层选取并相加
        agg = None
        for i, cb in enumerate(self.codebooks):
            idx = max(0, min(int(code_list[i]), cb.shape[0] - 1))
            vec = cb[idx]  # [e_dim]
            agg = vec if agg is None else agg + vec
        if agg is None:
            agg = torch.zeros(self.e_dim, device=self.device)
        # decoder
        if self.decoder is not None:
            try:
                with torch.no_grad():
                    out = self.decoder(agg)
                return out
            except Exception as e:
                print(f"[RQVAE] decoder forward failed, using agg directly: {e}")
                # 若decoder失败，则回退到聚合向量；维度可能不匹配，需pad或截断
                if agg.shape[-1] != self.emb_dim:
                    # 简单对齐：若agg维度小于emb_dim则右侧补零，否则截断
                    if agg.shape[-1] < self.emb_dim:
                        pad = torch.zeros(self.emb_dim - agg.shape[-1], device=self.device)
                        return torch.cat([agg, pad], dim=0)
                    else:
                        return agg[:self.emb_dim]
                return agg
        # 无decoder，若输出维度定义为emb_dim且与e_dim不同，进行对齐
        if agg.shape[-1] != self.emb_dim:
            if agg.shape[-1] < self.emb_dim:
                pad = torch.zeros(self.emb_dim - agg.shape[-1], device=self.device)
                return torch.cat([agg, pad], dim=0)
            else:
                return agg[:self.emb_dim]
        return agg


def load_item_codes(json_path: str) -> Dict[int, List[int]]:
    """
    Robust JSON loader:
    Supports:
      - dict: {"item_id": [code1, code2, ...], ...}
      - list of objects: [{"id": item_id, "codes": [...]}, ...]
      - list of pairs: [[item_id, [codes...]], ...]
    Converts keys to int.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: Dict[int, List[int]] = {}

    if isinstance(data, dict):
        for k, v in data.items():
            try:
                item_id = int(k)
            except:
                continue
            if isinstance(v, list):
                mapping[item_id] = v
            elif isinstance(v, dict):
                # try common keys
                codes = v.get("codes") or v.get("index") or v.get("tokens")
                if isinstance(codes, list):
                    mapping[item_id] = codes
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                # expect keys like id/item_id and codes/index
                item_id = e.get("id") or e.get("item_id")
                codes = e.get("codes") or e.get("index") or e.get("tokens")
                if item_id is not None and isinstance(codes, list):
                    try:
                        mapping[int(item_id)] = codes
                    except:
                        pass
            elif isinstance(e, list) and len(e) == 2 and isinstance(e[1], list):
                try:
                    mapping[int(e[0])] = e[1]
                except:
                    pass
    return mapping