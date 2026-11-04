# UniComm — Universal AI Communication Layer

Local-first React + FastAPI project for a shared speech/sign/text/translation workspace.

**Honest status:** this is a runnable foundation, not a claim that every abstract feature is already complete. Dataset inspection, Amrita landmark extraction, Transformer training/evaluation, video prediction endpoints, and optional ASR/translation/TTS integrations are included. Webcam preview works, but continuous rolling-window webcam inference is not wired yet. Dataset 2 training intentionally waits for validated annotation mappings. No pretrained ISL weights or fake metrics are included. A signing avatar needs verified ISL animation data and is not bundled.

## Setup (Windows / VS Code)
1. Extract ZIP and open `UniComm` in VS Code.
2. Run `scripts\setup_windows.bat`.
3. Edit `backend\.env`; set `AMRITA_DATASET_PATH` and `ISL_CSLRT_DATASET_PATH` to your local dataset roots.
4. Run `scripts\start_backend.bat`, then `scripts\start_frontend.bat` in another terminal.
5. Open `http://127.0.0.1:5173`; API docs are at `http://127.0.0.1:8000/docs`.

Python 3.10/3.11 and Node.js 20+ recommended. Optional engines: `pip install -r backend/requirements-optional.txt`. Large models may download on first use.

## Dataset handling
- **Amrita SLR:** class-folder video inventory → MediaPipe Holistic landmarks → per-video temporal sequences → Transformer training → held-out evaluation. Confirm detected folder labels before training.
- **ISL-CSLRT:** inventories `corpus_csv_files`, `Frames_Sentence_Level`, `Frames_Word_Level`, and `Videos_Sentence_Level`; reports spreadsheet headers and samples. It does not guess annotation mappings. Validate these before adding its word/sentence training adapter.
- Do not merge label spaces unless verified. Metrics are generated only by actual evaluation.

See `docs/DATASET_GUIDE.md`, `docs/ARCHITECTURE.md`, and `docs/API.md`.
