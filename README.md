# Indic Conformer ASR Service

A FastAPI-based speech-to-text API service for Indian languages using AI4Bharat's Indic Conformer multilingual model.

## Features

- Supports multiple Indian languages
- REST API endpoint for audio transcription
- Supports both CTC and RNN-T decoders
- Automatic audio resampling to 16kHz
- Handles various audio formats via torchaudio

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
python conformer_api.py
```

The API will be available at `http://localhost:8000`

### API Endpoint

**POST** `/transcribe`

**Parameters:**
- `audio`: Audio file (form-data)
- `language`: Language code (default: "hi" for Hindi)
- `decoder`: Decoder type - "ctc" or "rnnt" (default: "ctc")

**Example:**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "audio=@sample.wav" \
  -F "language=hi" \
  -F "decoder=ctc"
```

## Model

Uses the [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) model from Hugging Face.
