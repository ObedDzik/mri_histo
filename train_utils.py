import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

@dataclass
class EarlyStopper:
    patience: int
    best: float = float("-inf")
    num_bad: int = 0
    # NEW: keep best checkpoint in memory (CPU) for immediate eval
    best_state_cpu: dict | None = None

    def update(self, metric: float, model: torch.nn.Module, save_path: Path | None = None, tag: str = "model") -> bool:
        """
        If improved, snapshot the model to CPU memory and (optionally) save to disk.
        """
        if metric > self.best:
            self.best = metric
            # snapshot to CPU to avoid tying up GPU memory
            self.best_state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict()}, save_path)
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            return False

    def load_best_into(self, model: torch.nn.Module, strict: bool = False) -> bool:
        """
        Load the in-memory best state back into `model`. Returns True if loaded.
        """
        if self.best_state_cpu is not None:
            model.load_state_dict(self.best_state_cpu, strict=strict)
            return True
        return False

def save_embeddings(out_dir: Path | str, fname: str, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"embeddings": embeddings}
    if labels is not None:
        payload["labels"] = labels
    torch.save(payload, out_dir / fname)

def set_seed(seed: int, deterministic_cudnn: bool = True):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
