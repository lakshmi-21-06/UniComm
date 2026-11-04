# Architecture
React/Vite UI → FastAPI REST API → dataset adapters, optional ASR/translation/TTS, and PyTorch sequence classifier.

The shared workspace routes text to translation and speech output. Dataset-specific pipelines remain separate. Optional engines return explicit errors when unavailable rather than fake outputs.

Not yet included as verified capabilities: continuous webcam rolling-window inference, Dataset 2 training (requires annotation mapping), and a validated ISL signing avatar. Avatar animation needs suitable sign-animation data.
