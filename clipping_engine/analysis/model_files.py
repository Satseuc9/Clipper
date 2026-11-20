"""
Small model files, downloaded once into MODEL_DIR (git-ignored).

- YAMNet (Google, Apache-2.0): AudioSet sound-event classifier, 521 classes,
  used for laughter / screams / cheering / gasps. TFLite, ~4 MB.
- YuNet (OpenCV Zoo, MIT): face detector used for vertical cropping. ONNX, ~230 KB.

Whisper weights are downloaded by faster-whisper itself (HuggingFace) into
MODEL_DIR/whisper the first time transcription runs.
"""

import logging
import os
from pathlib import Path

import requests

from clipping_engine import config

log = logging.getLogger(__name__)

MODELS = {
    "yamnet.tflite": (
        "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/latest/yamnet.tflite",
        3_000_000,
    ),
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        100_000,
    ),
}


class ModelUnavailable(RuntimeError):
    pass


def model_path(name):
    """Local path of a model file, downloading it on first use."""
    url, min_size = MODELS[name]
    folder = Path(config.MODEL_DIR)
    path = folder / name
    if path.exists() and path.stat().st_size >= min_size:
        return path
    folder.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ModelUnavailable(f"could not download {name}: {type(exc).__name__}") from None
    if len(resp.content) < min_size:
        raise ModelUnavailable(f"download of {name} looks wrong ({len(resp.content)} bytes)")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    os.replace(tmp, path)
    return path
