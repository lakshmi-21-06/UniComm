from pathlib import Path
import json

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def inventory(root: Path):
    if not root.exists():
        return {
            "exists": False,
            "path": str(root),
            "message": "Path does not exist or is not configured.",
        }

    counts, examples = {}, []

    for p in root.rglob("*"):
        if p.is_file():
            ext = p.suffix.lower() or "[no extension]"
            counts[ext] = counts.get(ext, 0) + 1

            if len(examples) < 20:
                examples.append(str(p.relative_to(root)))

    return {
        "exists": True,
        "path": str(root),
        "file_count": sum(counts.values()),
        "extension_counts": counts,
        "sample_files": examples,
    }


def find_amrita_classes(root: Path):
    """Find folders containing videos directly."""

    video_folders = []

    for folder in root.rglob("*"):
        if not folder.is_dir():
            continue

        direct_videos = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        ]

        if direct_videos:
            video_folders.append((folder, len(direct_videos)))

    return sorted(
        video_folders,
        key=lambda item: str(item[0]).lower(),
    )


def inspect_amrita(root: Path):
    base = inventory(root)

    if not base["exists"]:
        return {
            **base,
            "dataset": "amrita",
            "classes": [],
            "video_count": 0,
        }

    classes = []
    total = 0

    for folder, count in find_amrita_classes(root):
        classes.append({
            "label": folder.name,
            "video_count": count,
            "examples": [
                str(v.relative_to(root))
                for v in sorted(folder.iterdir())
                if v.is_file() and v.suffix.lower() in VIDEO_EXTS
            ][:3],
        })

        total += count

    return {
        **base,
        "dataset": "amrita",
        "classes": classes,
        "class_count": len(classes),
        "video_count": total,
        "note": (
            "Classes are folders containing videos directly. "
            "Verify the reported class names and counts before training."
        ),
    }


def extract_mediapipe_sequence(video_path: Path, max_frames=120):
    try:
        import mediapipe as mp
    except Exception as e:
        raise RuntimeError(f"MediaPipe unavailable: {e}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    seq = []

    try:
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:

            for _ in range(max_frames):
                ok, frame = cap.read()

                if not ok:
                    break

                res = holistic.process(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )

                def flat(landmarks, count, dims, visibility=False):
                    if not landmarks:
                        return [0.0] * (count * dims)

                    out = []

                    for lm in landmarks.landmark[:count]:
                        out.extend([lm.x, lm.y, lm.z])

                        if visibility:
                            out.append(getattr(lm, "visibility", 0.0))

                    return out

                seq.append(
                    flat(res.pose_landmarks, 33, 4, True)
                    + flat(res.left_hand_landmarks, 21, 3)
                    + flat(res.right_hand_landmarks, 21, 3)
                )

    finally:
        cap.release()

    if not seq:
        raise RuntimeError(f"No readable frames in {video_path}")

    arr = np.asarray(seq, dtype=np.float32)
    indices = np.linspace(
        0,
        len(arr) - 1,
        min(64, len(arr)),
    ).astype(int)

    return arr[indices]


def preprocess_amrita(
    root: Path,
    output: Path,
    max_videos=None,
):
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {root}"
        )

    output.mkdir(parents=True, exist_ok=True)

    class_folders = find_amrita_classes(root)
    samples = []

    for folder, _ in class_folders:
        for video in sorted(folder.iterdir()):
            if video.is_file() and video.suffix.lower() in VIDEO_EXTS:
                samples.append((video, folder.name))

    if max_videos is not None:
        samples = samples[:max_videos]

    if len(set(label for _, label in samples)) < 2:
        raise ValueError(
            "Found fewer than two classes with directly contained videos. "
            "Check the dataset folder structure."
        )

    X, y, groups, errors = [], [], [], []

    for index, (video, label) in enumerate(samples, start=1):
        try:
            X.append(extract_mediapipe_sequence(video))
            y.append(label)
            groups.append(str(video.resolve()))

        except Exception as e:
            errors.append({
                "video": str(video),
                "error": str(e),
            })

        if index % 25 == 0:
            print(
                f"Processed {index}/{len(samples)} videos; "
                f"successful={len(X)}, failed={len(errors)}"
            )

    if len(X) < 4 or len(set(y)) < 2:
        raise RuntimeError(
            f"Not enough valid videos processed ({len(X)}). "
            f"Errors: {errors[:5]}"
        )

    np.savez_compressed(
        output / "amrita_sequences.npz",
        X=np.array(X, dtype=object),
        y=np.array(y),
        groups=np.array(groups),
    )

    manifest = {
        "dataset": "amrita",
        "samples": len(X),
        "classes": sorted(set(y)),
        "class_count": len(set(y)),
        "feature_dim": int(X[0].shape[1]),
        "successful_videos": len(X),
        "failed_videos": len(errors),
        "errors": errors,
        "label_mapping": {
            c: i for i, c in enumerate(sorted(set(y)))
        },
    }

    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest