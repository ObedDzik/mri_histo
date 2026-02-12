import os
import sys
from warnings import warn

DINOV3_LIBRARY_PATH = os.getenv(
    "DINOV3_LIBRARY_PATH",
)
if DINOV3_LIBRARY_PATH is not None:
    sys.path.append(DINOV3_LIBRARY_PATH)

DINOV3_CHECKPOINTS_PATH = os.getenv(
    "DINOV3_CHECKPOINTS_PATH",
)
if DINOV3_CHECKPOINTS_PATH is None:
    warn(
        "DINOV3_CHECKPOINTS_PATH environment variable not set. Set this environment variable to the path where DINOv3 checkpoints are stored.",
    )

def dinov3_vitl16(**kwargs):
    import torch
    import os
    REPO_DIR = DINOV3_LIBRARY_PATH
    weight_basename = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    weights = os.path.join(DINOV3_CHECKPOINTS_PATH, weight_basename)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {weights}")
    kwargs["weights"] = weights
    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", **kwargs)
    return model