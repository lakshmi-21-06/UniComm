from collections import deque
from typing import Dict
from pathlib import Path
import time
import threading
import subprocess
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import settings, ROOT
from .schemas import (
    DatasetConfig,
    InspectRequest,
    TrainRequest,
    TranslateRequest,
    TTSRequest,
)
from .datasets import (
    inspect_amrita,
    preprocess_amrita,
    extract_mediapipe_sequence,
)
from .modeling import train_amrita, job_status, predict_sequence


app = FastAPI(title=settings.app_name, version="0.1.0")

# Reuse one ASR model and prevent concurrent transcription calls.
_asr_lock = threading.Lock()
INDIC_MODEL_PATHS = {
    "te": "/home/lakshmi/indicconformer_stt_te_hybrid_rnnt_large.nemo",
    "hi": "/home/lakshmi/indicconformer_stt_hi_hybrid_rnnt_large.nemo",
}
_indic_models = {}


def get_indic_model(language: str):
    if language not in INDIC_MODEL_PATHS:
        raise ValueError(f"Unsupported Indic language: {language}")

    if language not in _indic_models:
        from nemo.collections.asr.models import EncDecRNNTBPEModel

        model = EncDecRNNTBPEModel.restore_from(
            restore_path=INDIC_MODEL_PATHS[language],
            map_location=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
        )
        model.eval()
        _indic_models[language] = model

    return _indic_models[language]

def convert_to_mono_wav(input_path: Path) -> Path:
    output_path = input_path.with_name(
        f"{input_path.stem}_mono.wav"
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return output_path
# --------------------------------------------------
# Continuous webcam recognition state
# --------------------------------------------------

_live_lock = threading.Lock()
_live_holistic = None
_live_sessions = {}


@lru_cache(maxsize=1)
def get_asr_model():
    from faster_whisper import WhisperModel

    return WhisperModel(
        settings.asr_model,
        device="cpu",
        compute_type="int8",
    )



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

paths = {
    "amrita": (
        settings.resolve(settings.amrita_dataset_path)
        if settings.amrita_dataset_path
        else None
    )
}

proc = settings.resolve(settings.processed_data_dir)
ckpt = settings.resolve(settings.checkpoint_dir)

@app.post("/api/avatar/signs")
def get_avatar_signs(req: dict):
    import re
    import numpy as np

    text = str(req.get("text", "")).strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Enter a word or message first.",
        )

    dataset_file = proc / "amrita" / "amrita_sequences.npz"

    if not dataset_file.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Processed dataset not found: {dataset_file}",
        )

    try:
        data = np.load(dataset_file, allow_pickle=True)
        X = data["X"]
        y = data["y"]

        def normalize(value):
            value = str(value).strip().lower()

            # Remove numeric class prefixes, e.g. 9.hello
            value = re.sub(r"^\d+\.", "", value)

            return "".join(
                char for char in value if char.isalnum()
            )

        label_lookup = {}

        for index, label in enumerate(y):
            label_lookup.setdefault(normalize(label), []).append(index)

        # Dataset spelling corrections and common phrases.
        aliases = {
            "tomorrow": "tommorow",
            "umbrella": "umberlla",
            "thank you": "thankyou",
            "good morning": "goodmorning",
            "you are perfect": "youareperfect",
        }

        tokens = re.findall(r"[a-z0-9]+", text.lower())
        signs = []
        unsupported = []
        position = 0

        while position < len(tokens):
            matched_key = None
            matched_count = 0

            # Prefer the longest supported phrase.
            for count in (3, 2, 1):
                phrase = " ".join(
                    tokens[position:position + count]
                )

                if len(tokens[position:position + count]) != count:
                    continue

                key = normalize(aliases.get(phrase, phrase))

                if key in label_lookup:
                    matched_key = key
                    matched_count = count
                    break

            if matched_key is None:
                unsupported.append(tokens[position])
                position += 1
                continue

            # Use the first recorded sequence for this class.
            sample_index = label_lookup[matched_key][0]
            frames = np.asarray(
                X[sample_index], dtype=np.float32
            )

            signs.append({
                "word": " ".join(
                    tokens[position:position + matched_count]
                ),
                "label": str(y[sample_index]),
                "frames": frames.tolist(),
            })

            position += matched_count

        return {
            "input_text": text,
            "signs": signs,
            "unsupported_words": unsupported,
            "supported_sign_count": len(signs),
            "message": (
                "Recorded dataset movements returned."
                if signs
                else "No supported signs found for this input."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load avatar movements: {exc}",
        )


# --------------------------------------------------
# Health and dataset configuration
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "time": time.time(),
        "configured_datasets": {
            key: str(value) if value else ""
            for key, value in paths.items()
        },
        "features": {
            "asr": "optional faster-whisper",
            "translation": "optional transformers",
            "tts": "optional edge-tts",
            "sign_model": (ckpt / "amrita_transformer.pt").exists(),
        },
    }


@app.get("/api/datasets")
def get_datasets():
    return {
        key: str(value) if value else ""
        for key, value in paths.items()
    }


@app.post("/api/datasets/configure")
def configure(req: DatasetConfig):
    if req.amrita_path is not None:
        paths["amrita"] = (
            Path(req.amrita_path).expanduser().resolve()
            if req.amrita_path
            else None
        )

    return {
        "configured": {
            key: str(value) if value else ""
            for key, value in paths.items()
        }
    }


@app.post("/api/datasets/inspect")
def inspect(req: InspectRequest):
    root = paths["amrita"]

    if not root:
        raise HTTPException(
            status_code=400,
            detail="Configure the Amrita dataset path first.",
        )

    return inspect_amrita(root)


@app.post("/api/datasets/preprocess/amrita")
def preprocess():
    if not paths["amrita"]:
        raise HTTPException(
            status_code=400,
            detail="Configure AMRITA_DATASET_PATH first.",
        )

    try:
        return preprocess_amrita(paths["amrita"], proc / "amrita")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --------------------------------------------------
# Model training and evaluation
# --------------------------------------------------

@app.post("/api/train")
def train(req: TrainRequest):
    try:
        return train_amrita(
            proc / "amrita" / "amrita_sequences.npz",
            ckpt,
            req.epochs,
            req.batch_size,
            req.learning_rate,
            req.sequence_length,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/train/status")
def train_status():
    return job_status()


@app.get("/api/evaluation")
def evaluation():
    evaluation_path = ckpt / "evaluation.json"

    if not evaluation_path.exists():
        return {
            "available": False,
            "message": "No saved evaluation. Preprocess and train first.",
        }

    import json

    return {
        "available": True,
        **json.loads(evaluation_path.read_text(encoding="utf-8")),
    }


# --------------------------------------------------
# Sign language video prediction
# --------------------------------------------------

@app.post("/api/sign/predict-video")
async def predict_video(file: UploadFile = File(...)):
    temp = ROOT / "data" / "uploads"
    temp.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    video_path = temp / f"clip_{int(time.time() * 1000)}{suffix}"

    try:
        video_path.write_bytes(await file.read())

        checkpoint = ckpt / "amrita_transformer.pt"

        if not checkpoint.exists():
            raise HTTPException(
                status_code=400,
                detail="No trained checkpoint. Preprocess and train first.",
            )

        sequence = extract_mediapipe_sequence(video_path)

        return predict_sequence(sequence, checkpoint)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        video_path.unlink(missing_ok=True)


# --------------------------------------------------
# Real-time webcam sign prediction
# --------------------------------------------------


@app.post("/api/sign/predict-frames")
async def predict_frames(file: UploadFile = File(...)):
    import cv2
    import mediapipe as mp

    temp = ROOT / "data" / "uploads"
    temp.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "webcam.webm").suffix or ".webm"
    video_path = temp / f"webcam_{int(time.time() * 1000)}{suffix}"

    try:
        video_path.write_bytes(await file.read())

        checkpoint = ckpt / "amrita_transformer.pt"

        if not checkpoint.exists():
            raise HTTPException(
                status_code=400,
                detail="No trained checkpoint. Preprocess and train first.",
            )

        # Check whether at least one hand is visible in this clip.
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open the webcam clip.",
            )

        hand_detected = False

        try:
            with mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                checked = 0

                while checked < 20:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    checked += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if result.multi_hand_landmarks:
                        hand_detected = True
                        break
        finally:
            cap.release()

        if not hand_detected:
            return {
                "detected": False,
                "label": None,
                "confidence": None,
                "message": "No hand detected — prediction skipped.",
            }

        # Only classify after hand detection succeeds.
        sequence = extract_mediapipe_sequence(video_path)
        prediction = predict_sequence(sequence, checkpoint)

        return {
            **prediction,
            "detected": True,
            "message": "Hand detected. Sign prediction completed.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        video_path.unlink(missing_ok=True)

# --------------------------------------------------
# Hand landmark detection
# --------------------------------------------------

@app.post("/api/sign/hand-landmarks")
async def detect_hand_landmarks(file: UploadFile = File(...)):
    import cv2
    import mediapipe as mp
    import numpy as np

    temp = ROOT / "data" / "uploads"
    temp.mkdir(parents=True, exist_ok=True)

    video_path = temp / f"hands_{int(time.time() * 1000)}.webm"

    try:
        video_path.write_bytes(await file.read())

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open the webcam clip.",
            )

        hands_module = mp.solutions.hands
        detected = []

        try:
            with hands_module.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                frame_count = 0
                while frame_count < 20:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    frame_count += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if result.multi_hand_landmarks:
                        for hand, handedness in zip(
                            result.multi_hand_landmarks,
                            result.multi_handedness,
                        ):
                            detected.append({
                                "hand": handedness.classification[0].label,
                                "landmarks": [
                                    {
                                        "x": float(point.x),
                                        "y": float(point.y),
                                    }
                                    for point in hand.landmark
                                ],
                            })

            return {
                "detected": bool(detected),
                "hands": detected[-2:],
                "frames_checked": frame_count,
            }
        finally:
            cap.release()

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        video_path.unlink(missing_ok=True)

# --------------------------------------------------
# Hand landmark detection from a single webcam frame
# --------------------------------------------------

# --------------------------------------------------
# Hand landmark detection from a single webcam frame
# --------------------------------------------------

@app.post("/api/sign/hand-landmarks-image")
async def detect_hand_landmarks_image(file: UploadFile = File(...)):
    import cv2
    import mediapipe as mp
    import numpy as np

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image received.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image received.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5,
    ) as hands:
        result = hands.process(rgb)

    detected_hands = []

    if result.multi_hand_landmarks:
        handedness_list = result.multi_handedness or []

        for index, hand in enumerate(result.multi_hand_landmarks):
            handedness = "Unknown"

            if index < len(handedness_list):
                handedness = handedness_list[index].classification[0].label

            detected_hands.append({
                "hand": handedness,
                "landmarks": [
                    {
                        "x": float(point.x),
                        "y": float(point.y),
                    }
                    for point in hand.landmark
                ],
            })

    return {
        "detected": bool(detected_hands),
        "hands": detected_hands,
    }


# --------------------------------------------------
# Continuous live webcam sign recognition
# --------------------------------------------------

# --------------------------------------------------
# Continuous live webcam sign recognition
# --------------------------------------------------

@app.post("/api/sign/live-reset")
def reset_live_session(session_id: str = Form(...)):
    """Reset the rolling frame buffer when live recognition starts."""

    with _live_lock:
        _live_sessions[session_id] = {
            "frames": deque(maxlen=32),
            "hand_detected": deque(maxlen=32),
            "frame_count": 0,
        }

    return {
        "success": True,
        "message": "Live recognition session reset.",
    }


@app.post("/api/sign/live-frame")
async def predict_live_frame(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Accept one webcam image, extract 258 Holistic features,
    and predict from a rolling sequence of 32 frames.
    """

    import cv2
    import mediapipe as mp
    import numpy as np

    global _live_holistic

    checkpoint = ckpt / "amrita_transformer.pt"

    if not checkpoint.exists():
        raise HTTPException(
            status_code=400,
            detail="No trained checkpoint. Preprocess and train first.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded webcam frame is empty.",
        )

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the webcam image.",
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        with _live_lock:

            # Create one persistent Holistic tracker.
            if _live_holistic is None:
                _live_holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                )

            result = _live_holistic.process(rgb)

            def flatten_landmarks(
                landmarks,
                count,
                dims,
                visibility=False,
            ):
                if not landmarks:
                    return [0.0] * (count * dims)

                values = []

                for landmark in landmarks.landmark[:count]:
                    values.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ])

                    if visibility:
                        values.append(
                            getattr(landmark, "visibility", 0.0)
                        )

                # Keep the expected feature length.
                expected = count * dims
                return values[:expected] + [0.0] * max(
                    0, expected - len(values)
                )

            # Preserve the exact training feature order.
            features = (
                flatten_landmarks(
                    result.pose_landmarks, 33, 4, True
                )
                + flatten_landmarks(
                    result.left_hand_landmarks, 21, 3
                )
                + flatten_landmarks(
                    result.right_hand_landmarks, 21, 3
                )
            )

            if len(features) != 258:
                raise RuntimeError(
                    f"Expected 258 features, got {len(features)}."
                )

            hand_detected = bool(
                result.left_hand_landmarks
                or result.right_hand_landmarks
            )

            # Initialize the session if it has not been reset.
            if session_id not in _live_sessions:
                _live_sessions[session_id] = {
                    "frames": deque(maxlen=32),
                    "hand_detected": deque(maxlen=32),
                    "frame_count": 0,
                }

            session = _live_sessions[session_id]

            session["frames"].append(features)
            session["hand_detected"].append(hand_detected)
            session["frame_count"] += 1

            frame_count = session["frame_count"]
            has_hand = any(session["hand_detected"])

            # Wait until a full rolling window is available.
            if len(session["frames"]) < 32 or not has_hand:
                return {
                    "detected": hand_detected,
                    "ready": False,
                    "label": None,
                    "confidence": None,
                    "frames_collected": len(session["frames"]),
                    "message": (
                        "Collecting webcam frames..."
                        if has_hand
                        else "Show your hands to the camera."
                    ),
                }

            # Predict every 4 incoming frames to reduce CPU load.
            if frame_count % 4 != 0:
                return {
                    "detected": hand_detected,
                    "ready": True,
                    "label": None,
                    "confidence": None,
                    "frames_collected": len(session["frames"]),
                    "message": "Collecting the next prediction window.",
                }

            sequence = np.asarray(
                list(session["frames"]),
                dtype=np.float32,
            )

            prediction = predict_sequence(sequence, checkpoint)

            return {
                **prediction,
                "detected": hand_detected,
                "ready": True,
                "frames_collected": len(session["frames"]),
                "message": (
                    "Live sign prediction updated."
                    if prediction.get("accepted", False)
                    else "Prediction uncertain. Continue signing."
                ),
            }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Live prediction failed: {exc}",
        )
# --------------------------------------------------
# Speech recognition
# Supported language codes:
# en = English, te = Telugu, hi = Hindi
# Language is optional for backward compatibility.
# --------------------------------------------------

@app.post("/api/speech/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
):
    allowed_languages = {"en", "te", "hi"}

    if language is not None and language not in allowed_languages:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language. Choose en, te, or hi.",
        )

    temp = ROOT / "data" / "uploads"
    temp.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    audio_path = temp / f"audio_{int(time.time() * 1000)}{suffix}"

    try:
        audio_path.write_bytes(await file.read())

        with _asr_lock:
            if language in {"te", "hi"}:
                model = get_indic_model(language)
                mono_path = convert_to_mono_wav(audio_path)
                print("Selected language:", repr(language))
                print("Model type:", type(model).__name__)
                print("Decoder type:", type(model.ctc_decoder).__name__)
                print("Language masks:", list(model.ctc_decoder.language_masks.keys()))

                results = model.transcribe(
                [str(mono_path)],
                batch_size=1,
                language_id=language,
            )

                result = results[0]

                if isinstance(result, (list, tuple)):
                    result = result[0] if result else ""

                text = (
                    result.text
                    if hasattr(result, "text")
                    else str(result)
                )

                return {
                    "text": text.strip(),
                    "language": language,
                    "language_probability": None,
                }

            try:
                model = get_asr_model()
            except ImportError:
                raise HTTPException(
                    status_code=503,
                    detail="ASR unavailable. Install faster-whisper.",
                )

            segments, info = model.transcribe(
                str(audio_path),
                task="transcribe",
                language=language,
                vad_filter=True,
                beam_size=3,
                condition_on_previous_text=False,
            )

            text = " ".join(
                segment.text.strip()
                for segment in segments
            ).strip()

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
        }

    except HTTPException:
        raise
    
    except Exception as exc:
        import traceback

        print("SPEECH TRANSCRIPTION ERROR:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {exc}",
        )

    finally:
        audio_path.unlink(missing_ok=True)


# --------------------------------------------------
# Translation
# --------------------------------------------------
@lru_cache(maxsize=1)
def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained(
        settings.translation_model
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.translation_model
    )
    model.eval()
    return tokenizer, model
@app.post("/api/translate")
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is required.",
        )

    try:
        start_time = time.perf_counter()

        tokenizer, model = load_translation_model()

        print(
            f"Translation model load/cache time: "
            f"{time.perf_counter() - start_time:.2f} seconds"
        )

        tokenizer.src_lang = req.source_language

        encoded = tokenizer(
            req.text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        target_token_id = tokenizer.convert_tokens_to_ids(
            req.target_language
        )

        generation_start = time.perf_counter()

        with torch.no_grad():
            output = model.generate(
                **encoded,
                forced_bos_token_id=target_token_id,
                max_new_tokens=128,
            )

        print(
            f"Translation generation time: "
            f"{time.perf_counter() - generation_start:.2f} seconds"
        )

        translated_text = tokenizer.batch_decode(
            output,
            skip_special_tokens=True,
        )[0]

        return {
            "text": translated_text,
            "source_language": req.source_language,
            "target_language": req.target_language,
            "model": settings.translation_model,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Translation unavailable/model load failed: {exc}",
        )


# --------------------------------------------------
# Text-to-speech
# --------------------------------------------------

@app.post("/api/tts")
async def tts(req: TTSRequest):
    try:
        import edge_tts
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail=(
                "TTS unavailable. Install edge-tts "
                "or use browser speech synthesis."
            ),
        )

    voices = {
        "en": "en-US-AriaNeural",
        "te": "te-IN-ShrutiNeural",
        "hi": "hi-IN-SwaraNeural",
    }

    output_dir = ROOT / "data" / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_path = output_dir / f"tts_{int(time.time() * 1000)}.mp3"
    voice = voices.get(req.language, voices["en"])

    try:
        await edge_tts.Communicate(
            req.text,
            voice,
        ).save(str(audio_path))

        return FileResponse(
            path=audio_path,
            media_type="audio/mpeg",
            filename=audio_path.name,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
