"""
Two-stage training for MRI-to-Histopathology alignment.
Stage 1 (Triplet): Align MRI encoder embeddings to histopathology reference
Stage 2 (CE): Fine-tune classification head on aligned embeddings
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tempfile import mkdtemp
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix)
from sklearn.linear_model import LogisticRegression
import wandb
from omegaconf import OmegaConf
import os
from tqdm import tqdm

from create_dataset import PicaiSliceDataset
from create_dataloader import get_dataloaders
from create_model_wrapper import DINOModelWrapper
from get_dino_model import dinov3_vitl16
from triplet_loss_utils import get_histo_by_isup, triplet_loss_batch
from train_utils import (EarlyStopper, set_seed)

def extract_patient_ids(df, patient_col = "patient_id", case_col = "case_id"):
    """Extract patient IDs from dataframe, inferring from case_id if needed."""
    if patient_col in df.columns:
        return df[patient_col].astype(str).to_numpy()
    if case_col in df.columns:
        # Common format: "patient_study" -> extract "patient"
        return df[case_col].astype(str).str.split("_").str[0].to_numpy()
    raise ValueError(
        f"Cannot infer patient IDs: neither '{patient_col}' nor '{case_col}' found in "
        f"columns: {list(df.columns)}"
    )

def validate_patient_splits(train_df, val_df, test_df = None, patient_col= "patient_id", case_col = "case_id"):
    """Ensure no patient appears in multiple splits (critical for valid evaluation)."""
    train_pts = set(extract_patient_ids(train_df, patient_col, case_col))
    val_pts = set(extract_patient_ids(val_df, patient_col, case_col))
    test_pts = set(extract_patient_ids(test_df, patient_col, case_col)) if test_df is not None else set()
    
    overlaps = {"train∩val": train_pts & val_pts, "train∩test": train_pts & test_pts, "val∩test": val_pts & test_pts}
    print(f"[Split Validation] Patients: train={len(train_pts)}, val={len(val_pts)}, test={len(test_pts)}")
    print(f"[Split Validation] Overlaps: {', '.join(f'{k}={len(v)}' for k, v in overlaps.items())}")
    
    if any(overlaps.values()):
        samples = {k: sorted(v)[:5] for k, v in overlaps.items() if v}
        raise ValueError(
            f"DATA LEAK: Patient overlap detected across splits!\n"
            f"{samples}\n"
            "Fix fold assignment at patient level before training."
        )

def run_triplet_epoch(loader, model, triplet_loss_fn, optimizer = None, device = "cuda", desc='train'):
    """Run one epoch of triplet loss training/validation."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_samples = 0
    with torch.set_grad_enabled(is_training):
        for batch in tqdm(loader, desc=desc):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
        
            _, embeddings = model(images)
            loss = triplet_loss_fn(embeddings, labels)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def extract_embeddings(loader, model, device="cuda", desc='extract_emb_train'):
    """Extract all embeddings and labels from a dataloader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    for batch in tqdm(loader, desc=desc):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        _, embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels.cpu())
    if not all_embeddings:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    
    X = torch.cat(all_embeddings, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    return X, y


def evaluate_embeddings_with_logreg(X_train, y_train, X_val, y_val, n_classes, max_iter, class_weights=class_weights):
    """Train logistic regression on embeddings and evaluate."""
    clf = LogisticRegression(max_iter=max_iter, solver="saga", class_weight = class_weights, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    metrics = {
        "acc": float(accuracy_score(y_val, y_pred)),
        "bacc": float(balanced_accuracy_score(y_val, y_pred)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro")),
        "clf": clf,
    }
    # accuracy
    per_acc = {}
    for c in range(n_classes):
        mask = y_val == c
        if mask.any():
            per_acc[c] = float((y_pred[mask] == c).mean())
        else:
            per_acc[c] = float("nan")
    metrics["per_acc"] = per_acc
    # AUC
    per_auc = {c: float("nan") for c in range(n_classes)}
    macro_auc = float("nan")
    
    try:
        probs = clf.predict_proba(X_val)
        auc_scores = []
        for c in range(n_classes):
            y_binary = (y_val == c).astype(int)
            # Need positive and negative samples
            if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
                auc = roc_auc_score(y_binary, probs[:, c])
                per_auc[c] = float(auc)
                auc_scores.append(auc)
        if auc_scores:
            macro_auc = float(np.mean(auc_scores))
    except Exception as e:
        print(f"[Warning] AUC calculation failed: {e}")
    
    metrics["per_auc"] = per_auc
    metrics["macro_auc"] = macro_auc
    return metrics

def run_ce_epoch(loader, model, class_weights = None, optimizer = None, device = "cuda"):
    """Run one epoch of cross-entropy training/validation."""
    is_training = optimizer is not None
    model.train(is_training)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    all_preds = []
    all_labels = []
    
    with torch.set_grad_enabled(is_training):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits, _ = model(images)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    
    if all_preds:
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        f1_macro = float(f1_score(y_true, y_pred, average="macro"))
        bacc = float(balanced_accuracy_score(y_true, y_pred))
    else:
        f1_macro = 0.0
        bacc = 0.0
    return {"loss": avg_loss, "acc": acc, "f1_macro": f1_macro, "bacc": bacc}

def setup_dataloaders(cfg):
    """Create train/val/test datasets and loaders."""
    print("\n" + "="*80)
    print("SETTING UP DATA")
    print("="*80)
    print(f"Folds: train={cfg.folds_train}, val={cfg.folds_val}, test={cfg.folds_test}")

    train_loader, val_loader, test_loader, class_weights, classes_present, n_classes = get_dataloaders(
        manifest = cfg.mri_manifest,
        folds_train = cfg.folds_train, 
        folds_val = cfg.folds_val,
        folds_test= cfg.folds_test,
        target = cfg.target,
        use_skip = cfg.use_skip,
        channels = cfg.channels,
        missing_channel_mode = cfg.missing_channel_mode,
        pct_lower = cfg.pct_lower, 
        pct_upper = cfg.pct_upper,
        batch_size = cfg.batch_size,
        pos_ratio = cfg.pos_ratio,
        num_workers = cfg.num_workers,
        pin_memory = True,
    )
    
    validate_patient_splits(train_loader.dataset.df, val_loader.dataset.df, test_loader.dataset.df if test_loader.dataset else None)
    n_classes = len(train_loader.dataset.get_label_distribution())
    print(f"\nClasses: {n_classes}")
    print(f"Train distribution: {train_loader.dataset.get_label_distribution()}")
    print(f"Val distribution: {val_loader.dataset.get_label_distribution()}")
    if test_loader.dataset:
        print(f"Test distribution: {test_loader.dataset.get_label_distribution()}")
    return train_loader, val_loader, test_loader, class_weights, classes_present, n_classes

def setup_histo_buckets(cfg):
    """Load histopathology reference embeddings."""
    print("\n" + "="*80)
    print("LOADING HISTOPATHOLOGY REFERENCE")
    print("="*80)
    # For triplet training, we want to align to train+val histo embeddings
    # For validation metric, we use held-out test histo embeddings
    train_csv = str(Path(cfg.histo_marksheet_dir) / "train.csv")
    val_csv = str(Path(cfg.histo_marksheet_dir) / "val.csv")
    test_csv = str(Path(cfg.histo_marksheet_dir) / "test.csv")
    # Combine train+val for training buckets
    df_train = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv),], ignore_index=True)
    df_test = pd.read_csv(test_csv)
    
    train_buckets = get_histo_by_isup(
        encodings_dir=str(cfg.histo_emb_dir),
        marksheet_csv=df_train,
        num_classes=cfg.n_classes,
        provider=cfg.histo_provider,
    )
    val_buckets = get_histo_by_isup(
        encodings_dir=str(cfg.histo_emb_dir),
        marksheet_csv=df_test,
        num_classes=cfg.n_classes,
        provider=cfg.histo_provider,
    )
    for label, bucket in enumerate(train_buckets):
        if len(bucket) == 0:
            raise ValueError(f"Train histo bucket for class {label} is empty!")
    for label, bucket in enumerate(val_buckets):
        if len(bucket) == 0:
            print(f"[Warning] Val histo bucket for class {label} is empty")
    print(f"Train buckets: {', '.join(f'c{k}={len(v)}' for k, v in enumerate(train_buckets))}")
    print(f"Val buckets: {', '.join(f'c{k}={len(v)}' for k, v in enumerate(val_buckets))}")
    return train_buckets, val_buckets

def setup_model(cfg, device):
    """Initialize DINOv2-based model."""
    print("\n" + "="*80)
    print("SETTING UP MODEL")
    print("="*80)
    
    dino_backbone = dinov3_vitl16()
    model = DINOModelWrapper(
        backbone=dino_backbone,
        num_classes=cfg.data.n_classes,
        img_size=cfg.data.img_size,
        proj_dim=cfg.model.proj_dim,
        attn_dim=cfg.model.attn_dim,
        head_hidden=cfg.head.head_hidden,
        head_dropout=cfg.head.head_dropout,
        pixel_mean_std=(cfg.model.mean, cfg.model.std),
    )
    model = model.to(device)
    print(f"Embedding dim: {model.C} -> {cfg.model.proj_dim}")
    print(f"Num classes: {cfg.data.n_classes}")
    return model

def stage1_triplet_alignment(model, train_loader, val_loader, train_buckets, val_buckets, cfg, device, class_weights=class_weights):
    """Stage 1: Align encoder embeddings to histopathology reference."""
    print("\n" + "="*80)
    print("STAGE 1: TRIPLET ALIGNMENT")
    print("="*80)
    # Freeze everything except encoder and projection
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.proj.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.proj.parameters(), "lr": cfg.triplet_lr, "weight_decay": cfg.triplet_wd},
        {"params": model.encoder.parameters(), "lr": cfg.triplet_lr * cfg.enc_lr_mult, "weight_decay": cfg.triplet_wd},
    ])
    print(f"Optimizer: proj_lr={cfg.triplet_lr:.2e}, enc_lr={cfg.triplet_lr * cfg.enc_lr_mult:.2e}")
    print(f"Margin: {cfg.triplet_margin}")

    def train_triplet_loss(embeddings, labels):
        return triplet_loss_batch(
            embeddings = embeddings,
            labels = labels, 
            histo_dict = train_buckets, 
            margin = cfg.triplet_margin,
            reduction = 'mean',
            num_classes = cfg.n_classes, 
        )
    
    def val_triplet_loss(embeddings, labels):
        return triplet_loss_batch(
            embeddings = embeddings, 
            labels = labels, 
            histo_dict = val_buckets,
            margin = cfg.triplet_margin,
            reduction = 'mean',
            num_classes = cfg.n_classes
        )
    
    early_stopper = EarlyStopper(patience=cfg.triplet_patience)
    for epoch in range(1, cfg.triplet_epochs + 1):
        train_loss = run_triplet_epoch(train_loader, model, train_triplet_loss, optimizer, device, desc='train')
        val_loss = run_triplet_epoch(val_loader, model, val_triplet_loss, None, device, desc='eval')
        X_train, y_train = extract_embeddings(train_loader, model, device, desc='extract_emb_train')
        X_val, y_val = extract_embeddings(val_loader, model, device, desc='extract_emb_val')
        lr_metrics = evaluate_embeddings_with_logreg(X_train, y_train, X_val, y_val, cfg.n_classes, cfg.logreg_max_iter, class_weights=class_weights)
        
        print(f"\nEpoch {epoch}/{cfg.triplet_epochs}")
        print(f" Loss: train={train_loss:.4f}, val={val_loss:.4f}")
        print(f" LogReg: acc={lr_metrics['acc']:.4f}, bacc={lr_metrics['bacc']:.4f}, "f"f1={lr_metrics['f1_macro']:.4f}, auc={lr_metrics['macro_auc']:.4f}")
        log_dict = {
            "epoch": epoch,
            "stage1/train_loss": train_loss,
            "stage1/val_loss": val_loss,
            "stage1/lr_acc": lr_metrics["acc"],
            "stage1/lr_bacc": lr_metrics["bacc"],
            "stage1/lr_f1": lr_metrics["f1_macro"],
            "stage1/lr_auc": lr_metrics["macro_auc"],
        }
        for c in range(cfg.n_classes):
            log_dict[f"stage1/lr_acc_c{c}"] = lr_metrics["per_acc"][c]
            log_dict[f"stage1/lr_auc_c{c}"] = lr_metrics["per_auc"][c]
        wandb.log(log_dict)
        
        # Early stopping based on macro AUC
        checkpoint_path = str(Path(cfg.checkpoint_dir) / "stage1_best.pt")
        if early_stopper.update(lr_metrics["macro_auc"], model, save_path=checkpoint_path):
            print(f" New best model (auc={early_stopper.best:.4f})")
        else:
            print(f" No improvement ({early_stopper.num_bad}/{early_stopper.patience})")
            if early_stopper.should_stop():
                print(f"\nEarly stopping at epoch {epoch}")
                break
    if early_stopper.load_best_into(model):
        print(f"\nLoaded best model from epoch with auc={early_stopper.best:.4f}")
    else:
        print("\n[Warning] No improvement during training, using final model")
    model.eval()
    return model

def stage2_classification(model, train_loader, val_loader, class_weights, cfg, device):
    """Stage 2: Train classification head on aligned embeddings."""
    print("\n" + "="*80)
    print("STAGE 2: CLASSIFICATION HEAD")
    print("="*80)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if cfg.head_scope == "head_only":
        for param in model.head.parameters():
            param.requires_grad = True
        trainable = [model.head.parameters()]
        print("Training: head only")

    elif cfg.head_scope == "head_and_proj":
        for param in model.head.parameters():
            param.requires_grad = True
        for param in model.proj.parameters():
            param.requires_grad = True
        trainable = [model.head.parameters(), model.proj.parameters()]
        print("Training: head + projection")

    elif cfg.head_scope == "all":
        for param in model.parameters():
            param.requires_grad = True
        trainable = [
            {"params": model.head.parameters(), "lr": cfg.head_lr},
            {"params": model.proj.parameters(), "lr": cfg.head_lr},
            {"params": model.encoder.parameters(), "lr": cfg.head_lr * cfg.enc_lr_mult},
        ]
        print("Training: all parameters (encoder uses reduced LR)")
    else:
        raise ValueError(f"Unknown head_scope: {cfg.head_scope}")
    
    if cfg.head_scope == "all":
        optimizer = torch.optim.AdamW(trainable, weight_decay=cfg.head_wd)
    else:
        params = []
        for p_group in trainable:
            params.extend(p_group)
        optimizer = torch.optim.AdamW(params, lr=cfg.head_lr, weight_decay=cfg.head_wd)
    print(f"Optimizer: lr={cfg.head_lr:.2e}, wd={cfg.head_wd:.2e}")
    class_weights = class_weights.to(device)
    early_stopper = EarlyStopper(patience=cfg.head_patience)
    
    for epoch in range(1, cfg.head_epochs + 1):
        train_metrics = run_ce_epoch(train_loader, model, class_weights, optimizer, device)
        val_metrics = run_ce_epoch(val_loader, model, class_weights, None, device)
        print(f"\nEpoch {epoch}/{cfg.head_epochs}")
        print(f"  Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['acc']:.4f}, "
              f"bacc={train_metrics['bacc']:.4f}, f1={train_metrics['f1_macro']:.4f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}, acc={val_metrics['acc']:.4f}, "
              f"bacc={val_metrics['bacc']:.4f}, f1={val_metrics['f1_macro']:.4f}")
        wandb.log({
            "epoch": epoch,
            "stage2/train_loss": train_metrics["loss"],
            "stage2/train_acc": train_metrics["acc"],
            "stage2/train_bacc": train_metrics["bacc"],
            "stage2/train_f1": train_metrics["f1_macro"],
            "stage2/val_loss": val_metrics["loss"],
            "stage2/val_acc": val_metrics["acc"],
            "stage2/val_bacc": val_metrics["bacc"],
            "stage2/val_f1": val_metrics["f1_macro"],
        })
        checkpoint_path = str(Path(cfg.checkpoint_dir) / "stage2_best.pt")
        if early_stopper.update(val_metrics["bacc"], model, save_path=checkpoint_path):
            print(f"  New best model (bacc={early_stopper.best:.4f})")
        else:
            print(f"  No improvement ({early_stopper.num_bad}/{early_stopper.patience})")
            if early_stopper.should_stop():
                print(f"\nEarly stopping at epoch {epoch}")
                break
    if early_stopper.load_best_into(model):
        print(f"\nLoaded best model with bacc={early_stopper.best:.4f}")
    else:
        print("\n[Warning] No improvement during training, using final model")
    model.eval()
    return model


@torch.no_grad()
def evaluate_final(model, test_loader, class_weights, device):
    """Evaluation on test set."""
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    
    model.eval()
    all_logits = []
    all_labels = []
    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits, _ = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    logits = torch.cat(all_logits, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = logits.argmax(axis=1)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = {
        "acc": float(accuracy_score(labels, preds)),
        "bacc": float(balanced_accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
    }
    # Per-class metrics
    per_acc = {}
    per_auc = {}
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            per_acc[c] = float((preds[mask] == c).mean())
        else:
            per_acc[c] = float("nan")
        # AUC
        y_binary = (labels == c).astype(int)
        if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
            per_auc[c] = float(roc_auc_score(y_binary, probs[:, c]))
        else:
            per_auc[c] = float("nan")
    metrics["per_acc"] = per_acc
    metrics["per_auc"] = per_auc
    metrics["macro_auc"] = float(np.nanmean(list(per_auc.values())))
    metrics["cm"] = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['acc']:.4f}")
    print(f"  Balanced Accuracy: {metrics['bacc']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  AUC (macro): {metrics['macro_auc']:.4f}")
    print(f"\nPer-Class Performance:")
    for c in range(n_classes):
        print(f"  Class {c}: acc={per_acc[c]:.4f}, auc={per_auc[c]:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics["cm"])
    return metrics

def main(cfg):
    """Main training pipeline."""
    print("\n" + "="*80)
    print("MRI-HISTOPATHOLOGY ALIGNMENT")
    print("="*80)
    print(f"Target: {cfg.data.target}")
    print(f"Seed: {cfg.seed}")

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(config=OmegaConf.to_object(cfg), **cfg.get("wandb", {}))
    _tmpdir = mkdtemp()
    OmegaConf.save(cfg, os.path.join(_tmpdir, "train_config.yaml"), resolve=True)
    wandb.save(os.path.join(_tmpdir, "train_config.yaml"), base_path=_tmpdir, policy="now")

    (train_loader, val_loader, test_loader, class_weights, classes_present, n_classes) = setup_dataloaders(cfg.data)
    train_buckets, val_buckets = setup_histo_buckets(cfg.data)
    model = setup_model(cfg, device)

    # Stage 1: Triplet alignment
    model = stage1_triplet_alignment(model, train_loader, val_loader, train_buckets, val_buckets, cfg.align, device, class_weights=class_weights)
    if cfg.train_mode == "train_classifier":
        # Stage 2: Classification head
        model = stage2_classification(model, train_loader, val_loader, class_weights, cfg.head, device)
        # Final evaluation on test set
        if test_loader is not None:
            test_metrics = evaluate_final(model, test_loader, class_weights, device)
            wandb.log({
                "test/acc": test_metrics["acc"],
                "test/bacc": test_metrics["bacc"],
                "test/f1_macro": test_metrics["f1_macro"],
                "test/macro_auc": test_metrics["macro_auc"],
            })
        final_path = str(Path(cfg.chkpt_dir) / "final_model.pt")
        torch.save(model.state_dict(), final_path)
        print(f"\nFinal model saved to: {final_path}")

    wandb.finish()
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)

def load_config(path):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    return cfg

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train MRI_Histo alignment model")
    p.add_argument("--config", "-c", help="Path to config file")
    cfg = p.parse_args()
    cfg = load_config(cfg.config)
    main(cfg)