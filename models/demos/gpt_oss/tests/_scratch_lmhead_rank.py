# Scratch (host only): for the captured LM-head inputs (generated/dflash_lmhead_x.pt), where does the BF8 LM head's
# argmax rank among the BFP4 LM head's logits? (the candidate count a BFP4 prescreen + BF8 rescore needs)
from pathlib import Path

import torch

import ttnn


def quant(w, dtype):
    """[hidden, vocab] -> the device tile format's dequantized values (shared exponents along 16 vocab columns)."""
    out = torch.empty_like(w)
    step = 8192
    for c in range(0, w.shape[1], step):
        t = ttnn.from_torch(w[:, c : c + step].contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
        out[:, c : c + step] = ttnn.to_torch(t).float()[:, : min(step, w.shape[1] - c)]
    return out


def test_lmhead_rank():
    from models.demos.gpt_oss.tt.dflash.lm_head import _hf_lm_head

    root = Path(__file__).parents[4] / "generated"
    X = torch.load(root / "dflash_lmhead_x.pt").float()
    W = _hf_lm_head().float().T.contiguous()  # [hidden, vocab]
    W = torch.nn.functional.pad(W, (0, (-W.shape[1]) % 32))
    w8, w4 = quant(W, ttnn.bfloat8_b), quant(W, ttnn.bfloat4_b)
    V = 201088
    l8 = (X @ w8)[:, :V]
    l4 = (X @ w4)[:, :V]
    a8 = l8.argmax(-1)
    rank = (l4 > l4.gather(1, a8[:, None])).sum(-1)  # 0: the BF8 argmax is also the BFP4 argmax
    top2 = l8.topk(2, -1).values
    margin = top2[:, 0] - top2[:, 1]
    err = (l4 - l8).abs()
    print(f"LMR rows {X.shape[0]}; |l4 - l8| mean {err.mean():.4f} max {err.max():.3f}", flush=True)
    for k in (1, 2, 4, 8, 16, 32):
        print(f"LMR BF8 argmax within BFP4 top-{k}: {(rank < k).float().mean():.4f}", flush=True)
    print(
        f"LMR worst rank {int(rank.max())}; BF8 top-1 margin quantiles {margin.quantile(torch.tensor([0.01, 0.1, 0.5])).tolist()}",
        flush=True,
    )
