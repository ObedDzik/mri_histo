import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from create_dataset import LABEL_MAPPERS, PicaiSliceDataset
IMG_SIZE=256


def collate_resize_to_imgsize(batch):
    imgs, labels = [], []
    extras_keys = [k for k in batch[0].keys() if k not in ("image", "label")]
    extras = {k: [] for k in extras_keys}
    for s in batch:
        x = s["image"].unsqueeze(0)  # [1,C,H,W]
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0)
        imgs.append(x)
        labels.append(torch.as_tensor(s["label"], dtype=torch.long))
        for k in extras_keys:
            extras[k].append(s[k])
    return {"image": torch.stack(imgs, 0), "label": torch.stack(labels, 0), **extras}

# def make_pos_sampler(df: pd.DataFrame, pos_ratio: float = 0.33, seed: int = 42):
#     """Oversample lesion-intersecting slices."""
#     is_pos = df["has_lesion"].astype(int).values
#     n_pos = int(is_pos.sum()); n_neg = len(is_pos) - n_pos
#     assert n_pos > 0, "No positive slices in train folds."
#     w_neg = 1.0
#     w_pos = (pos_ratio/(1-pos_ratio)) * (n_neg/max(1,n_pos))
#     w = np.where(is_pos==1, w_pos, w_neg).astype(np.float64)
#     generator = torch.Generator().manual_seed(seed)
#     return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True, generator=generator)

def make_pos_sampler(df, pos_ratio = 0.33, seed = 42):
    """Oversample lesion slices AND balance across ISUP grades."""

    is_pos = df["has_lesion"].astype(int).values
    n_pos = int(is_pos.sum())
    n_neg = len(is_pos) - n_pos
    assert n_pos > 0, "No positive slices in train folds."
    
    w_neg = 1.0
    w_pos_base = (pos_ratio/(1-pos_ratio)) * (n_neg/max(1, n_pos))
    
    labels = df["merged_ISUP"].values
    pos_labels = labels[is_pos == 1]
    class_counts = np.bincount(pos_labels, minlength=6)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1)
    
    w = np.ones(len(df), dtype=np.float64)
    w[is_pos == 0] = w_neg
    for i in np.where(is_pos == 1)[0]:
        grade = labels[i]
        w[i] = w_pos_base * class_weights[grade]
    
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.from_numpy(w), 
        num_samples=len(w), 
        replacement=True, 
        generator=generator
    )

def class_weights_from_train(df, target="isup6"):
    label_column = df['merged_ISUP']
    label_mapper = LABEL_MAPPERS[target]
    y = label_column.map(label_mapper)
    classes = sorted(int(c) for c in y.unique())
    cnt = Counter(int(v) for v in y.tolist())
    K, N = len(classes), len(y)
    ws = [N / (K * max(1, cnt.get(c, 0))) for c in classes]
    m = sum(ws)/len(ws)
    ws = [w/m for w in ws]
    return torch.tensor(ws, dtype=torch.float32), classes

def get_dataloaders(
    manifest: str,
    folds_train, folds_val, folds_test=None,
    target: str = "isup6",
    use_skip: bool = False,
    channels=("path_T2","path_ADC","path_HBV"),
    missing_channel_mode="zeros",
    pct_lower: float = 0.5, pct_upper: float = 99.5,
    batch_size: int = 16,
    pos_ratio: float = 0.33,
    num_workers: int = 4,
    pin_memory: bool = True,
):

    train_ds = PicaiSliceDataset(
        manifest_csv=manifest,
        folds=folds_train,
        use_skip=use_skip,
        target=target,
        channels=channels,
        missing_channel_mode=missing_channel_mode,
        pct_lower=pct_lower, pct_upper=pct_upper,
        transform=None
    )

    val_ds = PicaiSliceDataset(
        manifest_csv=manifest,
        folds=folds_val,
        use_skip=use_skip,
        target=target,
        channels=channels,
        missing_channel_mode=missing_channel_mode,
        pct_lower=pct_lower, pct_upper=pct_upper,
        transform=None
    )

    w_ce, classes_present = class_weights_from_train(train_ds.df, target=target)
    print("!!!!!! classes_present", classes_present)
    n_classes = len(classes_present)
    print("!!!!! number of classes", n_classes)

    sampler = make_pos_sampler(train_ds.df, pos_ratio=pos_ratio)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_resize_to_imgsize
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_resize_to_imgsize
    )

    test_ds, test_loader = None, None
    if folds_test is not None:
        test_ds = PicaiSliceDataset(
            manifest_csv=manifest,
            folds=folds_test,
            use_skip=use_skip,
            target=target,
            channels=channels,
            missing_channel_mode=missing_channel_mode,
            pct_lower=pct_lower, pct_upper=pct_upper,
            transform=None
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=collate_resize_to_imgsize
        )

    return (train_loader, val_loader, test_loader, w_ce, classes_present, n_classes)