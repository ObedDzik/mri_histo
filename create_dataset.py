from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Dict, Any, Callable
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset


@lru_cache(maxsize=128)
def _load_nifti_volume(path: str):
    """
    Load NIfTI volume and return as (Z, H, W) float32 array.
    Cached globally for efficiency across train/val datasets.
    """
    if not path:
        raise FileNotFoundError("Empty path")
    
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    
    img = nib.load(str(p))
    arr = img.get_fdata(dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape}")
    
    # Move last axis (Z) to front if needed
    return np.moveaxis(arr, -1, 0)

def percentile_normalize(x: np.ndarray, p_low: float = 0.5, p_high: float = 99.5):
    """Percentile windowing normalization to [0, 1]."""
    lo, hi = np.percentile(x, [p_low, p_high])
    x = np.clip(x, lo, hi)
    
    if hi > lo:
        x = (x - lo) / (hi - lo)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    
    return x.astype(np.float32)

def validate_label6(y):
    """Validate 6-class ISUP grade."""
    if y < 0 or y > 5:
        raise ValueError(f"Invalid merged_ISUP={y}, expected int in [0, 5]")
    return int(y)

def map_isup3(y6):
    """Map 6-class ISUP (0-5) to 3-class: {0,1}->0, {2,3}->1, {4,5}->2"""
    y6 = validate_label6(y6)
    return y6 // 2

def map_isupc3(y6):
    """Map 6-class ISUP to clinical 3-class: 0->0, {1,2}->1, {3,4,5}->2"""
    y6 = validate_label6(y6)
    if y6 == 0:
        return 0
    elif y6 <= 2:
        return 1
    else:
        return 2

def map_binary_all(y6):
    """Binary: {0,1}->0 (benign), {2,3,4,5}->1 (cancer)"""
    y6 = validate_label6(y6)
    return 0 if y6 <= 1 else 1

LABEL_MAPPERS = {"isup6": lambda y: validate_label6(y), "isup3": map_isup3, "isupc3": map_isupc3, "binary_all": map_binary_all}


class PicaiSliceDataset(Dataset):
    """
    Loads pre-extracted 2D slices from multi-parametric MRI (T2, ADC, HBV)
    with patient-level fold splits and configurable label schemes.
    Args:
        manifest_csv: Path to CSV with columns:
            - case_id, fold, z: identifiers
            - merged_ISUP, label3, has_lesion: labels
            - path_T2, path_ADC, path_HBV: NIfTI paths
            - bbox_prostate_{z0,z1,h0,h1,w0,w1}: crop coordinates
            - skip (optional): filter flag
        folds: Keep only these folds (e.g., [0, 1] for train). None = all.
        use_skip: If True, filter out rows where skip==1.
        target: Label scheme - 'isup6', 'isup3', 'isupc3', 'binary_low_high', 'binary_all'
        channels: Which image paths to load (in order).
        missing_channel_mode: How to handle empty paths - 'zeros' or 'repeat_t2'.
        pct_lower/pct_upper: Percentile normalization bounds per channel.
        transform: Optional callable(image, label, meta) -> (image, label, meta).
    Returns:
        Dict with keys: image, label, merged_ISUP, case_id, fold, z, has_lesion, bbox
    """
    
    REQUIRED_COLS = {
        "case_id", "fold", "z", "merged_ISUP", "has_lesion",
        "bbox_prostate_z0", "bbox_prostate_z1",
        "bbox_prostate_h0", "bbox_prostate_h1",
        "bbox_prostate_w0", "bbox_prostate_w1",
        "path_T2", "path_ADC",
    }
    
    def __init__(
        self,
        manifest_csv: str | Path,
        folds: Optional[Sequence[int | str]] = None,
        use_skip: bool = False,
        target: str = "isup6",
        channels: Sequence[str] = ("path_T2", "path_ADC", "path_HBV"),
        missing_channel_mode: str = "zeros",
        pct_lower: float = 0.5,
        pct_upper: float = 99.5,
        transform: Optional[Callable] = None,
    ):
        if target not in LABEL_MAPPERS:
            raise ValueError(f"Invalid target='{target}', choose from {list(LABEL_MAPPERS.keys())}")
        if missing_channel_mode not in ("zeros", "repeat_t2"):
            raise ValueError(f"Invalid missing_channel_mode='{missing_channel_mode}'")
        
        self.manifest_csv = Path(manifest_csv)
        df = pd.read_csv(self.manifest_csv)
        missing_cols = self.REQUIRED_COLS - set(df.columns)
        if missing_cols:
            raise ValueError(f"Manifest missing required columns: {missing_cols}")
        
        df["fold"] = df["fold"].astype(str).str.strip()
        df.loc[df["fold"].isin(["", "nan", "NaN"]), "fold"] = "NA"
        if use_skip and "skip" in df.columns:
            df = df[df["skip"] == 0].copy()
        if folds is not None:
            folds = [str(f) for f in folds]
            df = df[df["fold"].isin(folds)].copy()

        df["path_T2"] = df["path_T2"].astype(str).str.strip()
        df["path_ADC"] = df["path_ADC"].astype(str).str.strip()
        df = df[(df["path_T2"].str.len() > 0) & (df["path_ADC"].str.len() > 0)].copy()
        keep_cols = list(self.REQUIRED_COLS | set(channels))
        self.df = df[keep_cols].reset_index(drop=True)
        for col in ["case_id", "fold"]:
            self.df[col] = self.df[col].astype("category")
        
        self.target = target
        self.label_mapper = LABEL_MAPPERS[target]
        self.channels = tuple(channels)
        self.missing_channel_mode = missing_channel_mode
        self.pct_lower = pct_lower
        self.pct_upper = pct_upper
        self.transform = transform
        print(f"[{self.__class__.__name__}] Loaded {len(self.df)} slices "
              f"with target='{target}', {len(self.channels)} channels")
    
    def _extract_slice(self, path: str, z: int, h0: int, h1: int, w0: int, w1: int, fallback: Optional[np.ndarray] = None):
        """Extract cropped slice from volume or return fallback."""
        if not path:
            if self.missing_channel_mode == "repeat_t2" and fallback is not None:
                return fallback.copy()
            elif fallback is not None:
                return np.zeros_like(fallback, dtype=np.float32)
            else:
                raise ValueError("Missing path with no fallback provided")
        vol = _load_nifti_volume(path)
        return vol[z, h0:h1, w0:w1].astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        z = int(row["z"])
        z0, z1 = int(row["bbox_prostate_z0"]), int(row["bbox_prostate_z1"])
        h0, h1 = int(row["bbox_prostate_h0"]), int(row["bbox_prostate_h1"])
        w0, w1 = int(row["bbox_prostate_w0"]), int(row["bbox_prostate_w1"])
        
        path_t2 = str(row["path_T2"]).strip()
        t2_slice = self._extract_slice(path_t2, z, h0, h1, w0, w1)
        
        slices = []
        for ch_name in self.channels:
            if ch_name == "path_T2":
                slices.append(t2_slice)
            else:
                path = str(row.get(ch_name, "")).strip()
                ch_slice = self._extract_slice(path, z, h0, h1, w0, w1, fallback=t2_slice)
                slices.append(ch_slice)
        
        slices = [percentile_normalize(s, self.pct_lower, self.pct_upper) for s in slices]
        image = torch.from_numpy(np.stack(slices, axis=0)) #(C, H, W)
        merged_ISUP = int(row["merged_ISUP"])
        label = self.label_mapper(merged_ISUP)
        meta = {"case_id":str(row["case_id"]), "fold":str(row["fold"]), "z":z, "has_lesion":int(row["has_lesion"]), "bbox":(z0, z1, h0, h1, w0, w1)}
        
        if self.transform is not None:
            result = self.transform(image, label, meta)
            if isinstance(result, tuple):
                if len(result) == 3:
                    image, label, meta = result
                elif len(result) == 2:
                    image, label = result
                else:
                    image = result
        
        return {
            "image": image,
            "label": int(label),
            "merged_ISUP": merged_ISUP,
            "case_id": meta["case_id"],
            "fold": meta["fold"],
            "z": meta["z"],
            "has_lesion": meta["has_lesion"],
            "bbox": meta["bbox"],
        }
    
    def get_label_distribution(self) -> Dict[int, int]:
        """Get distribution of mapped labels in this dataset."""
        labels = [self.label_mapper(int(y6)) for y6 in self.df["merged_ISUP"]]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


if __name__ == "__main__":

    manifest="/project/aip-medilab/shared/picai/manifests/slices_manifest.csv"

    dataset = PicaiSliceDataset(
        manifest_csv=manifest,
        folds=None,
        use_skip=True,
        target="isup6",
        channels=("path_T2", "path_ADC", "path_HBV"),
        missing_channel_mode="zeros",
    )

    print(f"Dataset size: {len(dataset)}") #28139
    print(f"Label distribution: {dataset.get_label_distribution()}") # {0: 25990, 2: 1082, 3: 559, 4: 206, 5: 302}

    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}") #torch.Size([3, 200, 140])
    print(f"Label: {sample['label']} (from merged_ISUP={sample['merged_ISUP']})") #Label: 0 (from merged_ISUP=0)