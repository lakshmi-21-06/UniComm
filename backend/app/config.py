
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "UniComm API"
    host: str = "127.0.0.1"
    port: int = 8000
    amrita_dataset_path: str = ""
    processed_data_dir: str = "data/processed"
    checkpoint_dir: str = "data/checkpoints"
    asr_model: str = "medium"
    translation_model: str = "facebook/nllb-200-distilled-600M"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def resolve(self, value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else (ROOT / p).resolve()


settings = Settings()
