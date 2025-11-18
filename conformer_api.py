# cursor 

from __future__ import annotations

import io
from functools import lru_cache
from typing import Literal

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModel

LANG_CODE = "hi"
TARGET_SAMPLE_RATE = 16_000

@lru_cache(maxsize=1)
def load_model() -> AutoModel:
    try:
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
        )
        model.eval()
        return model
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("Failed to load ASR model") from exc


def _load_audio(file_bytes: bytes) -> tuple[torch.Tensor, int]:
    with io.BytesIO(file_bytes) as buffer:
        wav, sr = torchaudio.load(buffer)
    wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=TARGET_SAMPLE_RATE,
        )
        wav = resampler(wav)
        sr = TARGET_SAMPLE_RATE
    return wav, sr


app = FastAPI(title="Indic Conformer ASR Service")


@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: str = LANG_CODE,
    decoder: Literal["ctc", "rnnt"] = "ctc",
) -> JSONResponse:
    if decoder not in {"ctc", "rnnt"}:
        raise HTTPException(status_code=400, detail="decoder must be 'ctc' or 'rnnt'")

    file_bytes = await audio.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        waveform, _ = _load_audio(file_bytes)
        model = load_model()
        transcription = model(waveform, language, decoder)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "decoder": decoder,
            "language": language,
            "transcription": transcription,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "a:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )

