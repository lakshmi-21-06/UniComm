
from pathlib import Path
import json
import time
import threading
import traceback

import numpy as np

_state = {
    "running": False,
    "status": "idle",
    "logs": [],
    "metrics": None,
    "error": None,
}
_lock = threading.Lock()


def log(message):
    with _lock:
        _state["logs"].append(
            f"[{time.strftime('%H:%M:%S')}] {message}"
        )
        _state["logs"] = _state["logs"][-500:]


def job_status():
    with _lock:
        return dict(_state)


def train_amrita(
    npz_path: Path,
    checkpoint_dir: Path,
    epochs=60,
    batch_size=16,
    lr=0.0005,
    sequence_length=32,
):
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Processed data missing: {npz_path}. Run preprocessing first."
        )

    with _lock:
        _state.update({
            "running": True,
            "status": "training",
            "logs": [],
            "metrics": None,
            "error": None,
        })

    try:
        import torch
        from torch import nn
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            confusion_matrix,
            classification_report,
        )

        torch.manual_seed(42)
        np.random.seed(42)

        data = np.load(npz_path, allow_pickle=True)
        raw = data["X"]
        labels = data["y"].astype(str)

        classes = sorted(set(labels.tolist()))
        class_to_id = {name: i for i, name in enumerate(classes)}
        y = np.array(
            [class_to_id[label] for label in labels],
            dtype=np.int64,
        )

        if len(classes) < 2:
            raise RuntimeError("At least two classes are required.")

        counts = np.bincount(y, minlength=len(classes))
        if np.any(counts < 3):
            raise RuntimeError(
                "Each class needs at least 3 videos for stratified "
                "train/validation/test evaluation."
            )

        # Split videos while preserving class proportions.
        all_idx = np.arange(len(y))
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.15,
            random_state=43,
            stratify=y[train_val_idx],
        )

        dim = int(np.asarray(raw[0]).shape[1])

        # Resample every video to the same number of frames.
        def pack(index):
            sequence = np.asarray(raw[index], dtype=np.float32)

            if sequence.ndim != 2 or sequence.shape[1] != dim:
                raise ValueError(
                    f"Unexpected sequence shape at index {index}: "
                    f"{sequence.shape}"
                )

            if len(sequence) == 0 or not np.isfinite(sequence).all():
                raise ValueError(
                    f"Empty or invalid sequence at index {index}"
                )

            positions = np.linspace(
                0, len(sequence) - 1, sequence_length
            )
            left = np.floor(positions).astype(int)
            right = np.ceil(positions).astype(int)
            fraction = (positions - left)[:, None]

            return (
                sequence[left] * (1.0 - fraction)
                + sequence[right] * fraction
            ).astype(np.float32)

        X = np.stack([pack(i) for i in range(len(y))])

        # Calculate normalization statistics from TRAINING data only.
        train_frames = X[train_idx].reshape(-1, dim)
        mean = train_frames.mean(axis=0).astype(np.float32)
        std = train_frames.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0

        X = ((X - mean[None, None, :]) / std[None, None, :])
        X = np.clip(X, -10.0, 10.0).astype(np.float32)

        class SignDataset(Dataset):
            def __init__(self, indices):
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, item):
                i = self.indices[item]
                return (
                    torch.from_numpy(X[i]),
                    torch.tensor(y[i], dtype=torch.long),
                )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(dim, 128)
                self.pos = nn.Parameter(
                    torch.zeros(1, sequence_length, 128)
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=128,
                    nhead=4,
                    dim_feedforward=256,
                    batch_first=True,
                    dropout=0.1,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Sequential(
                    nn.LayerNorm(128),
                    nn.Linear(128, len(classes)),
                )

            def forward(self, x):
                x = self.proj(x) + self.pos
                x = self.enc(x)
                return self.head(x.mean(dim=1))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Net().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )
        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(
            SignDataset(train_idx),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            SignDataset(val_idx),
            batch_size=batch_size,
            shuffle=False,
        )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "amrita_transformer.pt"

        log(
            f"{len(y)} videos; {len(classes)} classes; device={device}"
        )
        log(
            f"Split: train={len(train_idx)}, validation={len(val_idx)}, "
            f"test={len(test_idx)}"
        )
        log("Feature normalization fitted on training data only.")

        best_val_loss = float("inf")
        best_epoch = 0
        patience = 12
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            model.train()
            train_losses = []

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            val_preds = []
            val_actual = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    val_losses.append(loss_fn(logits, yb).item())
                    val_preds.extend(logits.argmax(1).cpu().tolist())
                    val_actual.extend(yb.cpu().tolist())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            val_acc = accuracy_score(val_actual, val_preds)

            log(
                f"epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_accuracy={val_acc:.1%}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "classes": classes,
                        "feature_dim": dim,
                        "sequence_length": sequence_length,
                        "mean": mean,
                        "std": std,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                log(f"Early stopping at epoch {epoch}.")
                break

        # Reload the best validation checkpoint before testing.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        test_loader = DataLoader(
            SignDataset(test_idx),
            batch_size=batch_size,
            shuffle=False,
        )
        predictions = []
        actual = []

        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.to(device))
                predictions.extend(logits.argmax(1).cpu().tolist())
                actual.extend(yb.tolist())

        precision, recall, f1, _ = precision_recall_fscore_support(
            actual,
            predictions,
            labels=list(range(len(classes))),
            average="macro",
            zero_division=0,
        )

        metrics = {
            "test_accuracy": float(
                accuracy_score(actual, predictions)
            ),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "test_samples": len(test_idx),
            "classes": classes,
            "confusion_matrix": confusion_matrix(
                actual,
                predictions,
                labels=list(range(len(classes))),
            ).tolist(),
            "classification_report": classification_report(
                actual,
                predictions,
                labels=list(range(len(classes))),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
            "device": device,
            "split": "Stratified train/validation/test video split",
            "best_epoch": best_epoch,
            "best_validation_loss": float(best_val_loss),
        }

        (checkpoint_dir / "evaluation.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        with _lock:
            _state.update({
                "running": False,
                "status": "completed",
                "metrics": metrics,
                "error": None,
            })

        log(
            f"Completed. Best epoch={best_epoch}; "
            f"test accuracy={metrics['test_accuracy']:.1%}; "
            f"macro F1={metrics['macro_f1']:.1%}"
        )
        return metrics

    except Exception as exc:
        with _lock:
            _state.update({
                "running": False,
                "status": "failed",
                "error": str(exc),
            })
        log(traceback.format_exc())
        raise


def predict_sequence(sequence, checkpoint_path: Path):
    import torch
    from torch import nn

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    dim = checkpoint["feature_dim"]
    length = checkpoint["sequence_length"]
    classes = checkpoint["classes"]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dim, 128)
            self.pos = nn.Parameter(torch.zeros(1, length, 128))
            layer = nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,
                dim_feedforward=256,
                batch_first=True,
                dropout=0.1,
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(
                nn.LayerNorm(128),
                nn.Linear(128, len(classes)),
            )

        def forward(self, x):
            x = self.proj(x) + self.pos
            x = self.enc(x)
            return self.head(x.mean(dim=1))

    model = Net()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    a = np.asarray(sequence, dtype=np.float32)

    if a.ndim != 2 or a.shape[1] != dim or len(a) == 0:
        raise ValueError(
            f"Expected a non-empty sequence with {dim} features per frame."
        )
    if not np.isfinite(a).all():
        raise ValueError("Sequence contains invalid numeric values.")

    positions = np.linspace(0, len(a) - 1, length)
    left = np.floor(positions).astype(int)
    right = np.ceil(positions).astype(int)
    fraction = (positions - left)[:, None]
    a = a[left] * (1.0 - fraction) + a[right] * fraction

    # Apply the same training-set normalization.
    mean = checkpoint.get("mean")
    std = checkpoint.get("std")
    if mean is not None and std is not None:
        a = (a - mean) / np.where(std < 1e-6, 1.0, std)
        a = np.clip(a, -10.0, 10.0)

    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor(a[None], dtype=torch.float32)), dim=1
        )[0].numpy()


    index = int(probabilities.argmax())
    confidence = float(probabilities[index])

    # A provisional threshold; calibrate it using validation data.
    CONFIDENCE_THRESHOLD = 0.70

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "label": None,
            "confidence": confidence,
            "predicted_class": classes[index],
            "accepted": False,
            "warning": "Uncertain prediction. Please perform the sign again.",
        }

    return {
        "label": classes[index],
        "confidence": confidence,
        "accepted": True,
        "warning": (
            "Model prediction; verify before relying on its interpretation."
        ),
    }

