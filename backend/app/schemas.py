
from pydantic import BaseModel, Field
from typing import Optional


class DatasetConfig(BaseModel):
    amrita_path: Optional[str] = None


class InspectRequest(BaseModel):
    dataset: str = Field(pattern="^amrita$")


class TrainRequest(BaseModel):
    dataset: str = Field(default="amrita", pattern="^amrita$")
    epochs: int = Field(default=60, ge=1, le=200)
    batch_size: int = Field(default=16, ge=1, le=256)
    learning_rate: float = Field(default=0.0005, gt=0, le=0.1)
    sequence_length: int = Field(default=32, ge=8, le=128)


class TranslateRequest(BaseModel):
    text: str
    source_language: str = "eng_Latn"
    target_language: str = "tel_Telu"


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
