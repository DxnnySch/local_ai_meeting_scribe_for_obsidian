from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import json
import urllib.request
import wave
import importlib
import base64
import io
import gc
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - optional at import time
    sd = None

try:
    import pyaudiowpatch as pyaudio
except Exception:  # pragma: no cover - optional at import time
    pyaudio = None

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional at import time
    WhisperModel = None

try:
    from faster_whisper.tokenizer import LANGUAGES as WHISPER_LANGUAGES
except Exception:  # pragma: no cover - optional at import time
    WHISPER_LANGUAGES = None

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None

try:
    import stopwordsiso as stopwords_iso
except Exception:  # pragma: no cover - optional at import time
    stopwords_iso = None

try:
    from langdetect import DetectorFactory
    from langdetect import detect as detect_language

    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - optional at import time
    detect_language = None

try:
    from pyannote.audio import Pipeline as PyannotePipeline
except Exception:  # pragma: no cover - optional at import time
    PyannotePipeline = None

try:
    from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
except Exception:  # pragma: no cover - optional at import time
    AudioUtilities = None
    IAudioMeterInformation = None


APP = FastAPI(title="Local Meeting Scribe Companion", version="0.1.0")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
SUPPORTED_PROFILES = ("fast", "balanced", "pristine")


class SummarizeRequest(BaseModel):
    text: str
    language: str | None = None
    model: str | None = None
    ollama_url: str | None = None


class StartRequest(BaseModel):
    profile: Literal["fast", "balanced", "pristine"] = "balanced"
    ai_quality_mode: Literal["efficient", "full_vram_12gb"] = "full_vram_12gb"
    prefer_loopback: bool = True
    enable_diarization: bool = True
    expected_speakers: int = 3
    language: str = "auto"
    app_audio_only: bool = False
    include_mic: bool = True
    self_speaker_name: str = "Me"
    mic_device_contains: str = ""
    target_apps: list[str] = []


class StartResponse(BaseModel):
    ok: bool
    message: str


class TestMicRequest(BaseModel):
    mic_device_contains: str = ""
    duration_seconds: float = 1.8


class TestMicResponse(BaseModel):
    ok: bool
    device_name: str
    rms: float
    peak: float
    error: str | None = None


class SegmentOut(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None


class StopResponse(BaseModel):
    segments: list[SegmentOut]
    speakers: list[str]
    samples: dict[str, str]
    audio_samples: dict[str, str]


@dataclass
class RecordingState:
    is_recording: bool = False
    start_time: float | None = None
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    frames: list[bytes] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    output_wav: Path | None = None
    backend: str = "none"
    current_profile: str = "balanced"
    ai_quality_mode: str = "full_vram_12gb"
    diarization_enabled: bool = True
    expected_speakers: int = 3
    meeting_language: str = "auto"
    prefer_loopback: bool = True
    app_audio_only: bool = False
    include_mic: bool = True
    self_speaker_name: str = "Me"
    mic_device_contains: str = ""
    target_apps: set[str] = field(default_factory=set)
    last_target_audio_active: bool = True
    last_target_audio_check_at: float = 0.0
    stop_event: threading.Event = field(default_factory=threading.Event)
    worker_thread: threading.Thread | None = None
    # sounddevice backend handle
    stream: Any | None = None
    # pyaudiowpatch backend handles
    pa: Any | None = None
    mic_stream: Any | None = None
    loopback_stream: Any | None = None
    mic_rate: int = 16000
    loopback_rate: int = 16000
    mic_channels: int = 1
    loopback_channels: int = 1
    target_mix_rate: int = 16000
    chunk_duration_s: float = 0.1
    recorded_seconds: float = 0.0
    chunk_levels: list[tuple[float, float, float, float]] = field(default_factory=list)


STATE = RecordingState()
WHISPER_MODELS: dict[str, WhisperModel] = {}
PYANNOTE_PIPELINE: Any | None = None
STOPWORD_CACHE: dict[str, set[str]] = {}


def release_gpu_resources(unload_models: bool = True) -> dict[str, Any]:
    global PYANNOTE_PIPELINE
    released: dict[str, Any] = {"unloaded_models": bool(unload_models), "whisper_models": 0, "pyannote_unloaded": False}
    if unload_models:
        released["whisper_models"] = len(WHISPER_MODELS)
        WHISPER_MODELS.clear()
        if PYANNOTE_PIPELINE is not None:
            PYANNOTE_PIPELINE = None
            released["pyannote_unloaded"] = True
    gc.collect()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                released["cuda_cleared"] = True
            else:
                released["cuda_cleared"] = False
        except Exception as exc:
            released["cuda_error"] = str(exc)
    return released


def get_supported_languages() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if isinstance(WHISPER_LANGUAGES, dict) and WHISPER_LANGUAGES:
        for code, name in sorted(WHISPER_LANGUAGES.items(), key=lambda item: str(item[1]).lower()):
            items.append({"code": str(code), "label": f"{str(name).title()} ({str(code)})"})
    else:
        fallback = {
            "de": "German",
            "en": "English",
            "fr": "French",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "uk": "Ukrainian",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
        }
        for code, name in sorted(fallback.items(), key=lambda item: item[1].lower()):
            items.append({"code": code, "label": f"{name} ({code})"})
    return [
        {"code": "auto", "label": "Auto detect (recommended for mixed language / Denglisch)"},
        *items,
    ]


def _normalize_meeting_language(value: str) -> str:
    language = (value or "auto").strip().lower()
    if language in ("", "auto", "auto-multilingual", "auto_multilingual", "mixed", "bilingual"):
        return "auto"
    supported = {item["code"] for item in get_supported_languages()}
    return language if language in supported else "auto"


def _stopwords_for_language(lang_code: str) -> set[str]:
    code = str(lang_code or "").strip().lower()
    if not code:
        return set()
    if code in STOPWORD_CACHE:
        return STOPWORD_CACHE[code]
    words: set[str] = set()
    if stopwords_iso is not None:
        try:
            words = {str(item).lower() for item in stopwords_iso.stopwords(code)}
        except Exception:
            words = set()
    STOPWORD_CACHE[code] = words
    return words


def _detect_languages_from_utterances(utterances: list[str]) -> set[str]:
    detected: set[str] = set()
    if detect_language is None:
        return detected
    for utterance in utterances[:120]:
        text = re.sub(r"\s+", " ", utterance).strip()
        if len(text) < 20:
            continue
        try:
            lang = str(detect_language(text)).lower().strip()
        except Exception:
            continue
        if len(lang) >= 2:
            detected.add(lang[:2])
    return detected


def _normalize_ollama_base_url(value: str | None) -> str:
    raw = str(value or os.getenv("MSCRIBE_OLLAMA_URL", "http://127.0.0.1:11434")).strip()
    if not raw:
        return "http://127.0.0.1:11434"
    lowered = raw.lower()
    if lowered.endswith("/api/generate"):
        raw = raw[: -len("/api/generate")]
    if lowered.endswith("/api/tags"):
        raw = raw[: -len("/api/tags")]
    return raw.rstrip("/")


def list_ollama_models(ollama_url: str | None = None) -> list[dict[str, str]]:
    base_url = _normalize_ollama_base_url(ollama_url)
    request = urllib.request.Request(f"{base_url}/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
    except Exception as exc:
        print(f"ollama model list failed: {exc}")
        return []

    models = data.get("models", [])
    out: list[dict[str, str]] = []
    for model in models:
        name = str(model.get("name", "")).strip()
        if name:
            out.append({"id": name, "name": name})
    out.sort(key=lambda item: item["name"].lower())
    return out


def profile_to_asr_config(profile: str, ai_quality_mode: str) -> dict[str, Any]:
    if ai_quality_mode == "full_vram_12gb":
        if profile == "fast":
            return {
                "model": os.getenv("MSCRIBE_ASR_MODEL_FAST_12GB", os.getenv("MSCRIBE_ASR_MODEL_FAST", "large-v3")),
                "beam_size": int(os.getenv("MSCRIBE_BEAM_FAST_12GB", "5")),
                "vad_filter": True,
            }
        if profile == "pristine":
            return {
                "model": os.getenv("MSCRIBE_ASR_MODEL_PRISTINE_12GB", os.getenv("MSCRIBE_ASR_MODEL_PRISTINE", "large-v3")),
                "beam_size": int(os.getenv("MSCRIBE_BEAM_PRISTINE_12GB", "9")),
                "vad_filter": True,
            }
        return {
            "model": os.getenv("MSCRIBE_ASR_MODEL_BALANCED_12GB", os.getenv("MSCRIBE_ASR_MODEL_BALANCED", "large-v3")),
            "beam_size": int(os.getenv("MSCRIBE_BEAM_BALANCED_12GB", "7")),
            "vad_filter": True,
        }

    if profile == "fast":
        return {
            "model": os.getenv("MSCRIBE_ASR_MODEL_FAST", "medium"),
            "beam_size": int(os.getenv("MSCRIBE_BEAM_FAST", "2")),
            "vad_filter": True,
        }
    if profile == "pristine":
        return {
            "model": os.getenv("MSCRIBE_ASR_MODEL_PRISTINE", "large-v3"),
            "beam_size": int(os.getenv("MSCRIBE_BEAM_PRISTINE", "7")),
            "vad_filter": True,
        }
    return {
        "model": os.getenv("MSCRIBE_ASR_MODEL_BALANCED", os.getenv("MSCRIBE_ASR_MODEL", "large-v3")),
        "beam_size": int(os.getenv("MSCRIBE_BEAM_BALANCED", os.getenv("MSCRIBE_BEAM_SIZE", "5"))),
        "vad_filter": True,
    }


def load_whisper_model(profile: str, ai_quality_mode: str) -> WhisperModel | None:
    if WhisperModel is None:
        return None

    cfg = profile_to_asr_config(profile, ai_quality_mode)
    model_name = cfg["model"]
    cache_key = f"{ai_quality_mode}:{profile}:{model_name}"
    if cache_key in WHISPER_MODELS:
        return WHISPER_MODELS[cache_key]

    if ai_quality_mode == "full_vram_12gb":
        compute_type = os.getenv("MSCRIBE_ASR_COMPUTE_12GB", "float16")
        device = os.getenv("MSCRIBE_ASR_DEVICE_12GB", os.getenv("MSCRIBE_ASR_DEVICE", "cuda"))
    else:
        compute_type = os.getenv("MSCRIBE_ASR_COMPUTE", "int8_float16")
        device = os.getenv("MSCRIBE_ASR_DEVICE", "auto")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    WHISPER_MODELS[cache_key] = model
    return model


def _audio_callback(indata, frames, time_info, status) -> None:
    if status:
        # Avoid crashing on transient buffer events.
        print(f"audio status: {status}")
    with STATE.lock:
        if STATE.is_recording:
            if not STATE.include_mic:
                return
            mono = np.mean(indata, axis=1, keepdims=True)
            pcm16 = np.int16(np.clip(mono, -1.0, 1.0) * 32767.0)
            STATE.frames.append(pcm16.tobytes())
            mic_rms = float(np.sqrt(np.mean(np.square(pcm16.astype(np.float32))) + 1e-12))
            start_s = STATE.recorded_seconds
            end_s = start_s + (int(frames) / max(1, STATE.sample_rate))
            STATE.chunk_levels.append((start_s, end_s, mic_rms, 0.0))
            STATE.recorded_seconds = end_s


def _record_loop_pyaudio() -> None:
    loop_stream = STATE.loopback_stream
    mic_stream = STATE.mic_stream
    if loop_stream is None or mic_stream is None:
        return
    chunks_written = 0
    while STATE.is_recording and not STATE.stop_event.is_set():
        loop_started_at = time.time()
        try:
            loop_frames = max(1, int(STATE.loopback_rate * STATE.chunk_duration_s))
            mic_frames = max(1, int(STATE.mic_rate * STATE.chunk_duration_s))
            target_frames = max(1, int(STATE.target_mix_rate * STATE.chunk_duration_s))

            loop_arr = _read_stream_chunk(loop_stream, loop_frames, STATE.loopback_channels)
            mic_arr = _read_stream_chunk(mic_stream, mic_frames, STATE.mic_channels)
            loop_arr = _to_mono_int16(loop_arr, STATE.loopback_channels)
            mic_arr = _to_mono_int16(mic_arr, STATE.mic_channels)

            loop_arr = _resample_int16(loop_arr, STATE.loopback_rate, STATE.target_mix_rate, target_frames)
            mic_arr = _resample_int16(mic_arr, STATE.mic_rate, STATE.target_mix_rate, target_frames)

            if STATE.app_audio_only and not _target_app_audio_active_cached():
                loop_arr = np.zeros_like(loop_arr)
            if not STATE.include_mic:
                mic_arr = np.zeros_like(mic_arr)

            mic_gain = max(0.0, float(os.getenv("MSCRIBE_MIC_GAIN", "2.0")))
            if mic_gain != 1.0 and mic_arr.size > 0:
                mic_float = np.clip(mic_arr.astype(np.float32) * mic_gain, -32768.0, 32767.0)
                mic_arr = mic_float.astype(np.int16)

            mic_rms = float(np.sqrt(np.mean(np.square(mic_arr.astype(np.float32))) + 1e-12))
            loop_rms = float(np.sqrt(np.mean(np.square(loop_arr.astype(np.float32))) + 1e-12))

            has_loop = bool(np.any(loop_arr))
            has_mic = bool(np.any(mic_arr))
            if has_loop and has_mic:
                mix_noise_floor = max(1.0, float(os.getenv("MSCRIBE_MIX_NOISE_FLOOR", "45.0")))
                dominance_ratio = max(1.05, float(os.getenv("MSCRIBE_SOURCE_DOMINANCE_RATIO", "1.7")))
                if mic_rms >= max(mix_noise_floor, loop_rms * dominance_ratio):
                    # Keep mic intelligibility when loopback only has low-level noise.
                    mixed = mic_arr
                elif loop_rms >= max(mix_noise_floor, mic_rms * dominance_ratio):
                    mixed = loop_arr
                else:
                    mixed = ((loop_arr.astype(np.int32) + mic_arr.astype(np.int32)) // 2).astype(np.int16)
            elif has_mic:
                mixed = mic_arr
            else:
                mixed = loop_arr
            with STATE.lock:
                if STATE.is_recording:
                    STATE.frames.append(mixed.tobytes())
                    start_s = STATE.recorded_seconds
                    end_s = start_s + (target_frames / max(1, STATE.target_mix_rate))
                    STATE.chunk_levels.append((start_s, end_s, mic_rms, loop_rms))
                    STATE.recorded_seconds = end_s
                    chunks_written += 1
                    if chunks_written == 1:
                        print(
                            "first capture chunk written: "
                            f"mic_rms={mic_rms:.1f}, loop_rms={loop_rms:.1f}, "
                            f"target_frames={target_frames}"
                        )
        except Exception as exc:
            if STATE.stop_event.is_set():
                break
            print(f"pyaudio loop error: {exc}")
        elapsed = time.time() - loop_started_at
        sleep_for = max(0.0, STATE.chunk_duration_s - elapsed)
        if sleep_for > 0.0:
            time.sleep(sleep_for)


def _target_app_audio_active_cached() -> bool:
    now = time.time()
    with STATE.lock:
        if now - STATE.last_target_audio_check_at < 0.3:
            return STATE.last_target_audio_active
        target_apps = set(STATE.target_apps)
    active = _target_app_audio_active(target_apps)
    with STATE.lock:
        STATE.last_target_audio_active = active
        STATE.last_target_audio_check_at = now
    return active


def _target_app_audio_active(target_apps: set[str]) -> bool:
    if not target_apps:
        return True
    if AudioUtilities is None:
        return True
    try:
        sessions = AudioUtilities.GetAllSessions()
        for session in sessions:
            process = getattr(session, "Process", None)
            if process is None:
                continue
            name = str(getattr(process, "name", lambda: "")()).lower()
            if name not in target_apps:
                continue
            ctl = getattr(session, "_ctl", None)
            if ctl is None or IAudioMeterInformation is None:
                continue
            meter = ctl.QueryInterface(IAudioMeterInformation)
            peak = meter.GetPeakValue()
            if float(peak) > 0.01:
                return True
    except Exception:
        return True
    return False


def _resample_int16(samples: np.ndarray, src_rate: int, dst_rate: int, dst_len: int) -> np.ndarray:
    if dst_len <= 0:
        return np.zeros((0,), dtype=np.int16)
    if samples.size == 0:
        return np.zeros((dst_len,), dtype=np.int16)
    if src_rate == dst_rate and samples.size == dst_len:
        return samples.astype(np.int16, copy=False)

    src_positions = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    dst_positions = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    interp = np.interp(dst_positions, src_positions, samples.astype(np.float32))
    return np.int16(np.clip(interp, -32768, 32767))


def _to_mono_int16(samples: np.ndarray, channels: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((0,), dtype=np.int16)
    safe_channels = max(1, int(channels))
    if safe_channels <= 1:
        return samples.astype(np.int16, copy=False)
    frame_count = samples.size // safe_channels
    if frame_count <= 0:
        return samples.astype(np.int16, copy=False)
    trimmed = samples[: frame_count * safe_channels].reshape(frame_count, safe_channels)
    return np.int16(np.mean(trimmed.astype(np.float32), axis=1))


def _read_stream_chunk(stream: Any, frames: int, channels: int) -> np.ndarray:
    safe_frames = max(1, int(frames))
    safe_channels = max(1, int(channels))
    target_samples = safe_frames * safe_channels
    if stream is None:
        return np.zeros((target_samples,), dtype=np.int16)

    to_read = safe_frames
    try:
        available = int(stream.get_read_available())
        if available <= 0:
            return np.zeros((target_samples,), dtype=np.int16)
        to_read = max(1, min(safe_frames, available))
    except Exception:
        to_read = safe_frames

    try:
        raw = stream.read(to_read, exception_on_overflow=False)
    except Exception:
        return np.zeros((target_samples,), dtype=np.int16)

    arr = np.frombuffer(raw, dtype=np.int16)
    expected_read_samples = to_read * safe_channels
    if arr.size < expected_read_samples:
        arr = np.pad(arr, (0, expected_read_samples - arr.size), mode="constant")
    elif arr.size > expected_read_samples:
        arr = arr[:expected_read_samples]

    if to_read < safe_frames:
        arr = np.pad(arr, (0, target_samples - arr.size), mode="constant")
    return arr.astype(np.int16, copy=False)


def _pick_supported_channels(pa: Any, device_index: int, preferred_max: int = 2) -> int:
    device_info = pa.get_device_info_by_index(device_index)
    max_input = int(device_info.get("maxInputChannels", 1) or 1)
    max_input = max(1, min(preferred_max, max_input))
    for channels in range(max_input, 0, -1):
        try:
            pa.is_format_supported(
                int(device_info.get("defaultSampleRate", 48000)),
                input_device=device_index,
                input_channels=channels,
                input_format=pyaudio.paInt16,
            )
            return channels
        except Exception:
            continue
    return 1


def _pick_active_mic_device(pa: Any) -> Any | None:
    best_info = None
    best_rms = 0.0
    disallowed_tokens = (
        "loopback",
        "stereo mix",
        "what u hear",
        "wave out",
        "voicemeeter out",
        "cable output",
        "monitor of",
    )
    for device_idx in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(device_idx)
            name = str(info.get("name", ""))
            lowered = name.lower()
            if any(token in lowered for token in disallowed_tokens):
                continue
            if int(info.get("maxInputChannels", 0) or 0) <= 0:
                continue
            rate = int(info.get("defaultSampleRate", 48000) or 48000)
            channels = _pick_supported_channels(pa, info["index"], preferred_max=2)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=max(256, int(rate * 0.05)),
                input_device_index=info["index"],
            )
            try:
                raw = stream.read(max(256, int(rate * 0.15)), exception_on_overflow=False)
                arr = _to_mono_int16(np.frombuffer(raw, dtype=np.int16), channels).astype(np.float32)
                rms = float(np.sqrt(np.mean(np.square(arr)) + 1e-12))
                if rms > best_rms:
                    best_rms = rms
                    best_info = info
            finally:
                stream.stop_stream()
                stream.close()
        except Exception:
            continue
    # Conservative threshold to ignore completely silent inputs.
    if best_info is not None and best_rms > 50.0:
        print(f"auto-selected active mic device: {best_info.get('name')} (rms={best_rms:.2f})")
        return best_info
    return None


def _resolve_mic_device(pa: Any, mic_filter: str, include_mic: bool) -> Any:
    mic_filter = (mic_filter or "").strip().lower()
    if mic_filter:
        for device_idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(device_idx)
            name = str(info.get("name", ""))
            if mic_filter in name.lower() and int(info.get("maxInputChannels", 0) or 0) > 0:
                return info
        print("No mic device matched configured filter, falling back to default input device.")
        return pa.get_default_input_device_info()

    if not include_mic:
        return pa.get_default_input_device_info()

    auto_pick = os.getenv("MSCRIBE_AUTO_PICK_ACTIVE_MIC", "0").strip().lower() in ("1", "true", "yes", "on")
    if auto_pick:
        auto_mic = _pick_active_mic_device(pa)
        if auto_mic is not None:
            return auto_mic

    return pa.get_default_input_device_info()


def _start_pyaudio_capture() -> None:
    if pyaudio is None:
        raise RuntimeError("pyaudiowpatch not available")

    pa = pyaudio.PyAudio()
    try:
        loopback_filter = os.getenv("MSCRIBE_LOOPBACK_DEVICE_CONTAINS", "").strip().lower()
        if loopback_filter:
            loopback_device = None
            for device_idx in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(device_idx)
                name = str(info.get("name", ""))
                if loopback_filter in name.lower() and "loopback" in name.lower():
                    loopback_device = info
                    break
            if loopback_device is None:
                print(
                    "No loopback device matched MSCRIBE_LOOPBACK_DEVICE_CONTAINS, "
                    "falling back to default WASAPI loopback."
                )
                loopback_device = pa.get_default_wasapi_loopback()
        else:
            loopback_device = pa.get_default_wasapi_loopback()
        mic_filter = STATE.mic_device_contains or os.getenv("MSCRIBE_MIC_DEVICE_CONTAINS", "")
        mic_device = _resolve_mic_device(pa, mic_filter, include_mic=STATE.include_mic)
        print(
            f"selected devices: loopback='{loopback_device.get('name', 'unknown')}', "
            f"mic='{mic_device.get('name', 'unknown')}'"
        )
    except Exception as exc:
        pa.terminate()
        raise RuntimeError(f"Unable to resolve WASAPI devices: {exc}") from exc

    loopback_rate = int(loopback_device.get("defaultSampleRate", 48000) or 48000)
    mic_rate = int(mic_device.get("defaultSampleRate", 48000) or 48000)
    loopback_channels = _pick_supported_channels(pa, loopback_device["index"], preferred_max=2)
    mic_channels = _pick_supported_channels(pa, mic_device["index"], preferred_max=2)
    target_mix_rate = int(os.getenv("MSCRIBE_MIX_RATE", "16000"))
    chunk_duration_s = float(os.getenv("MSCRIBE_CHUNK_DURATION_S", "0.1"))

    loopback_stream = pa.open(
        format=pyaudio.paInt16,
        channels=loopback_channels,
        rate=loopback_rate,
        input=True,
        frames_per_buffer=max(256, int(loopback_rate * chunk_duration_s)),
        input_device_index=loopback_device["index"],
    )
    mic_stream = pa.open(
        format=pyaudio.paInt16,
        channels=mic_channels,
        rate=mic_rate,
        input=True,
        frames_per_buffer=max(256, int(mic_rate * chunk_duration_s)),
        input_device_index=mic_device["index"],
    )

    with STATE.lock:
        STATE.pa = pa
        STATE.loopback_stream = loopback_stream
        STATE.mic_stream = mic_stream
        STATE.backend = "pyaudiowpatch"
        STATE.is_recording = True
        STATE.stop_event.clear()
        STATE.loopback_rate = loopback_rate
        STATE.mic_rate = mic_rate
        STATE.loopback_channels = loopback_channels
        STATE.mic_channels = mic_channels
        STATE.target_mix_rate = target_mix_rate
        STATE.chunk_duration_s = chunk_duration_s
        STATE.sample_rate = target_mix_rate
        STATE.worker_thread = threading.Thread(target=_record_loop_pyaudio, daemon=True)
        STATE.worker_thread.start()

    print(
        "loopback capture active: "
        f"loopback={loopback_rate}Hz/{loopback_channels}ch, "
        f"mic={mic_rate}Hz/{mic_channels}ch, "
        f"mix={target_mix_rate}Hz, "
        f"include_mic={STATE.include_mic}"
    )


def _start_sounddevice_capture() -> None:
    if sd is None:
        raise RuntimeError("sounddevice not available")
    input_info = sd.query_devices(kind="input")
    selected_rate = int(input_info.get("default_samplerate", STATE.sample_rate) or STATE.sample_rate)
    stream = sd.InputStream(
        samplerate=selected_rate,
        channels=STATE.channels,
        dtype="float32",
        callback=_audio_callback,
    )
    stream.start()
    with STATE.lock:
        STATE.stream = stream
        STATE.backend = "sounddevice"
        STATE.is_recording = True
        STATE.sample_rate = selected_rate


def start_audio_capture(profile: str, prefer_loopback: bool, enable_diarization: bool) -> None:
    if profile not in SUPPORTED_PROFILES:
        raise RuntimeError(f"Unsupported profile: {profile}")

    with STATE.lock:
        if STATE.is_recording:
            raise RuntimeError("capture already running")
        STATE.frames.clear()
        STATE.start_time = time.time()
        STATE.current_profile = profile
        STATE.prefer_loopback = prefer_loopback
        STATE.diarization_enabled = enable_diarization
        STATE.stop_event.clear()
        STATE.output_wav = Path(tempfile.gettempdir()) / f"meeting_scribe_{int(time.time())}.wav"
        STATE.backend = "none"
        STATE.last_target_audio_active = True
        STATE.last_target_audio_check_at = 0.0
        STATE.recorded_seconds = 0.0
        STATE.chunk_levels.clear()

    if prefer_loopback and pyaudio is not None and os.name == "nt":
        try:
            _start_pyaudio_capture()
            return
        except Exception as exc:
            print(f"loopback capture unavailable, falling back to sounddevice: {exc}")

    _start_sounddevice_capture()


def stop_audio_capture() -> Path:
    with STATE.lock:
        if not STATE.is_recording:
            raise RuntimeError("capture is not running")
        backend = STATE.backend
        stream = STATE.stream
        worker_thread = STATE.worker_thread
        loopback_stream = STATE.loopback_stream
        mic_stream = STATE.mic_stream
        pa = STATE.pa
        STATE.is_recording = False
        STATE.stop_event.set()

    if backend == "sounddevice" and stream is not None:
        stream.stop()
        stream.close()
        with STATE.lock:
            STATE.stream = None
            STATE.backend = "none"
    elif backend == "pyaudiowpatch":
        # Stop input streams first so blocking read() returns.
        if loopback_stream is not None:
            try:
                loopback_stream.stop_stream()
            except Exception:
                pass
        if mic_stream is not None:
            try:
                mic_stream.stop_stream()
            except Exception:
                pass

        if worker_thread is not None:
            worker_thread.join(timeout=10.0)

        if loopback_stream is not None:
            loopback_stream.close()
        if mic_stream is not None:
            mic_stream.close()
        if pa is not None:
            pa.terminate()
        with STATE.lock:
            STATE.worker_thread = None
            STATE.loopback_stream = None
            STATE.mic_stream = None
            STATE.pa = None
            STATE.backend = "none"

    with STATE.lock:
        output_wav = STATE.output_wav
        if output_wav is None:
            raise RuntimeError("capture output path missing")
        if len(STATE.frames) == 0:
            silence = np.zeros((STATE.sample_rate,), dtype=np.int16).tobytes()
            STATE.frames = [silence]
        raw = b"".join(STATE.frames)
        STATE.frames.clear()

    # Normalize captured audio so low-level mic speech is not discarded by ASR/VAD.
    audio_int16 = np.frombuffer(raw, dtype=np.int16)
    if audio_int16.size > 0:
        audio_float = audio_int16.astype(np.float32)
        peak = float(np.max(np.abs(audio_float)))
        rms = float(np.sqrt(np.mean(np.square(audio_float)) + 1e-12))
        if peak > 0.0:
            # Target a healthy peak while avoiding heavy clipping.
            target_peak = 12000.0
            gain = min(10.0, max(1.0, target_peak / peak))
            # If the signal is very quiet, allow additional boost.
            if rms < 220.0:
                gain = min(12.0, gain * 1.8)
            audio_float = np.clip(audio_float * gain, -32768.0, 32767.0)
            raw = audio_float.astype(np.int16).tobytes()
            print(f"audio normalize applied: peak={peak:.1f}, rms={rms:.1f}, gain={gain:.2f}")

    with wave.open(str(output_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(STATE.sample_rate)
        wav_file.writeframes(raw)
    return output_wav


def assign_speakers_with_whisperx(audio_path: Path, segments: list[SegmentOut]) -> list[SegmentOut]:
    whisperx = None
    try:
        whisperx = importlib.import_module("whisperx")
    except Exception:
        whisperx = None
    if whisperx is None:
        return segments

    hf_token = os.getenv("MSCRIBE_HF_TOKEN", "").strip()
    if not hf_token:
        return segments

    diarize_device = os.getenv("MSCRIBE_DIARIZATION_DEVICE", os.getenv("MSCRIBE_ASR_DEVICE", "cpu"))
    try:
        diarizer = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=diarize_device)
        diarized = diarizer(str(audio_path))
    except Exception as exc:
        print(f"whisperx diarization failed: {exc}")
        return segments

    diarized_rows: list[dict[str, Any]] = []
    if hasattr(diarized, "to_dict"):
        try:
            diarized_rows = diarized.to_dict("records")
        except Exception:
            diarized_rows = []
    if not diarized_rows:
        return segments

    def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    for seg in segments:
        winner = seg.speaker or "SPEAKER_01"
        best = 0.0
        for row in diarized_rows:
            row_start = float(row.get("start", 0.0))
            row_end = float(row.get("end", 0.0))
            row_speaker = str(row.get("speaker", winner))
            score = overlap(seg.start, seg.end, row_start, row_end)
            if score > best:
                best = score
                winner = row_speaker
        seg.speaker = winner
    return segments


def assign_speakers_with_pyannote(audio_path: Path, segments: list[SegmentOut]) -> list[SegmentOut]:
    global PYANNOTE_PIPELINE
    if PyannotePipeline is None:
        return segments

    hf_token = os.getenv("MSCRIBE_HF_TOKEN", "").strip()
    if not hf_token:
        return segments

    if PYANNOTE_PIPELINE is None:
        try:
            PYANNOTE_PIPELINE = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
            )
            if torch is not None:
                target_device = os.getenv("MSCRIBE_DIARIZATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
                PYANNOTE_PIPELINE.to(torch.device(target_device))
        except Exception as exc:
            print(f"pyannote diarization init failed: {exc}")
            return segments

    with STATE.lock:
        expected_speakers = max(1, int(STATE.expected_speakers))

    try:
        diarized = PYANNOTE_PIPELINE(
            str(audio_path),
            min_speakers=max(1, expected_speakers - 1),
            max_speakers=min(8, expected_speakers + 1),
        )
    except Exception as exc:
        print(f"pyannote diarization failed: {exc}")
        return segments

    rows: list[dict[str, Any]] = []
    for turn, _, speaker in diarized.itertracks(yield_label=True):
        rows.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
    if not rows:
        return segments

    def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    for seg in segments:
        winner = seg.speaker or "SPEAKER_01"
        best = 0.0
        for row in rows:
            score = overlap(seg.start, seg.end, row["start"], row["end"])
            if score > best:
                best = score
                winner = row["speaker"]
        seg.speaker = winner
    return segments


def _load_wav_mono(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio.astype(np.float32) / 32768.0, sample_rate


def _simple_segment_features(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((8,), dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2.0)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
    denom = float(np.sum(spectrum) + 1e-9)
    centroid = float(np.sum(freqs * spectrum) / denom)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / denom))
    rolloff_idx = np.searchsorted(np.cumsum(spectrum), 0.85 * np.sum(spectrum))
    rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])
    peaks = np.partition(spectrum, -4)[-4:] if spectrum.size >= 4 else spectrum
    peak_mean = float(np.mean(peaks)) if peaks.size > 0 else 0.0
    peak_std = float(np.std(peaks)) if peaks.size > 0 else 0.0
    return np.array([rms, zcr, centroid, bandwidth, rolloff, peak_mean, peak_std, float(samples.size)], dtype=np.float32)


def _kmeans(features: np.ndarray, k: int, max_iter: int = 30) -> np.ndarray:
    if features.shape[0] <= k:
        return np.arange(features.shape[0], dtype=np.int32)
    centers = features[:k].copy()
    labels = np.zeros((features.shape[0],), dtype=np.int32)
    for _ in range(max_iter):
        distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = features[labels == idx]
            if members.size == 0:
                centers[idx] = features[np.random.randint(0, features.shape[0])]
            else:
                centers[idx] = np.mean(members, axis=0)
    return labels


def assign_speakers_with_feature_clustering(audio_path: Path, segments: list[SegmentOut], expected_speakers: int) -> list[SegmentOut]:
    audio, sample_rate = _load_wav_mono(audio_path)
    usable_indices: list[int] = []
    feature_rows: list[np.ndarray] = []

    for idx, seg in enumerate(segments):
        start_idx = max(0, int(seg.start * sample_rate))
        end_idx = min(audio.shape[0], int(seg.end * sample_rate))
        if end_idx - start_idx < int(0.45 * sample_rate):
            continue
        chunk = audio[start_idx:end_idx]
        usable_indices.append(idx)
        feature_rows.append(_simple_segment_features(chunk, sample_rate))

    if len(feature_rows) < 2:
        return segments

    features = np.vstack(feature_rows)
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-6
    norm_features = (features - mean) / std

    k = max(2, min(expected_speakers, len(feature_rows), 8))
    labels = _kmeans(norm_features, k=k)
    for feature_idx, seg_idx in enumerate(usable_indices):
        segments[seg_idx].speaker = f"SPEAKER_{int(labels[feature_idx]) + 1:02d}"

    # Fill short segments based on nearest previous assigned speaker.
    current = "SPEAKER_01"
    for seg in segments:
        if seg.speaker:
            current = seg.speaker
        else:
            seg.speaker = current
    return segments


def transcribe_file(
    audio_path: Path,
    profile: str,
    ai_quality_mode: str,
    enable_diarization: bool,
    language: str = "auto",
    disable_vad: bool = False,
) -> list[SegmentOut]:
    model = load_whisper_model(profile, ai_quality_mode)
    if model is None:
        # Fallback for first-run before models are installed.
        return [
            SegmentOut(
                start=0.0,
                end=0.0,
                text="Whisper model not available yet. Install dependencies and model, then retry.",
                speaker="SPEAKER_01",
            )
        ]

    asr_config = profile_to_asr_config(profile, ai_quality_mode)
    vad_filter = False if disable_vad else bool(asr_config["vad_filter"])
    normalized_language = _normalize_meeting_language(language)
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=asr_config["beam_size"],
        vad_filter=vad_filter,
        language=None if normalized_language == "auto" else normalized_language,
        word_timestamps=False,
    )
    out: list[SegmentOut] = []
    for seg in segments:
        out.append(
            SegmentOut(
                start=float(seg.start),
                end=float(seg.end),
                text=seg.text.strip(),
                speaker="SPEAKER_01",
            )
        )
    if not out:
        out = [SegmentOut(start=0.0, end=0.0, text="No speech detected.", speaker="SPEAKER_01")]
    if enable_diarization:
        with STATE.lock:
            expected_speakers = max(1, int(STATE.expected_speakers))
        diarized = assign_speakers_with_pyannote(audio_path, out)
        if len({seg.speaker for seg in diarized if seg.speaker}) <= 1:
            diarized = assign_speakers_with_whisperx(audio_path, diarized)
        if len({seg.speaker for seg in diarized if seg.speaker}) <= 1:
            diarized = assign_speakers_with_feature_clustering(audio_path, diarized, expected_speakers=expected_speakers)
        out = diarized
    return out


def build_speaker_samples(segments: list[SegmentOut]) -> tuple[list[str], dict[str, str]]:
    speakers: dict[str, str] = {}
    for seg in segments:
        speaker_id = seg.speaker or "SPEAKER_01"
        if speaker_id not in speakers and seg.text.strip():
            speakers[speaker_id] = seg.text.strip()
    if not speakers:
        speakers["SPEAKER_01"] = "No sample available."
    return list(speakers.keys()), speakers


def build_speaker_audio_samples(audio_path: Path, segments: list[SegmentOut]) -> dict[str, str]:
    audio, sample_rate = _load_wav_mono(audio_path)
    by_speaker: dict[str, SegmentOut] = {}
    for seg in segments:
        speaker_id = seg.speaker or "SPEAKER_01"
        duration = max(0.0, seg.end - seg.start)
        if speaker_id not in by_speaker:
            by_speaker[speaker_id] = seg
            continue
        prev = by_speaker[speaker_id]
        prev_duration = max(0.0, prev.end - prev.start)
        if duration > prev_duration:
            by_speaker[speaker_id] = seg

    samples: dict[str, str] = {}
    for speaker_id, seg in by_speaker.items():
        start_idx = max(0, int(seg.start * sample_rate))
        end_idx = min(audio.shape[0], int(seg.end * sample_rate))
        max_len = int(4.0 * sample_rate)
        if end_idx - start_idx > max_len:
            end_idx = start_idx + max_len
        chunk = audio[start_idx:end_idx]
        if chunk.size < int(0.25 * sample_rate):
            continue

        pcm16 = np.int16(np.clip(chunk, -1.0, 1.0) * 32767.0)
        buff = io.BytesIO()
        with wave.open(buff, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        samples[speaker_id] = base64.b64encode(buff.getvalue()).decode("ascii")
    return samples


def apply_self_speaker_tag(
    segments: list[SegmentOut], chunk_levels: list[tuple[float, float, float, float]]
) -> list[SegmentOut]:
    if not chunk_levels:
        return segments
    for seg in segments:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        if seg_end <= seg_start:
            continue
        overlap_total = 0.0
        mic_dominant_total = 0.0
        for c_start, c_end, mic_rms, loop_rms in chunk_levels:
            overlap = max(0.0, min(seg_end, c_end) - max(seg_start, c_start))
            if overlap <= 0:
                continue
            overlap_total += overlap
            if mic_rms > 200.0 and mic_rms > (loop_rms * 1.35):
                mic_dominant_total += overlap
        if overlap_total <= 0:
            continue
        if (mic_dominant_total / overlap_total) >= 0.6:
            seg.speaker = "SELF_USER"
    return segments


def generate_structured_summary(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    speaker_entries: list[tuple[str, str]] = []
    for line in lines:
        match = re.match(r"^\*\*(.+?):\*\*\s*(.+)$", line)
        if match:
            speaker_entries.append((match.group(1).strip(), match.group(2).strip()))

    if not speaker_entries:
        cleaned = " ".join(text.split())
        return (
            "### Meeting Summary\n"
            "- Could not parse speaker structure reliably.\n"
            f"- Transcript preview: {cleaned[:260] if cleaned else 'No transcript content.'}\n"
        )

    participants = sorted({speaker for speaker, _ in speaker_entries})
    speaker_counts = Counter(speaker for speaker, _ in speaker_entries)
    utterances = [entry[1] for entry in speaker_entries]

    action_item_triggers = (
        "bis zum nächsten mal",
        "bis zum naechsten mal",
        "bis zum nächsten meeting",
        "bis zum naechsten meeting",
        "beim nächsten mal",
        "beim naechsten mal",
        "für nächstes mal",
        "fuer naechstes mal",
        "for next meeting",
        "until next meeting",
        "by next meeting",
    )
    decision_keywords = (
        "entschieden",
        "beschlossen",
        "wir machen",
        "agreed",
        "final",
        "decide",
    )

    action_items = []
    decisions = []
    for speaker, utterance in speaker_entries:
        lower = utterance.lower()
        if any(keyword in lower for keyword in action_item_triggers):
            action_items.append(f"- [ ] {speaker}: {utterance}")
        if any(keyword in lower for keyword in decision_keywords):
            decisions.append(f"- {speaker}: {utterance}")

    detected_languages = _detect_languages_from_utterances(utterances)
    # Keep German + English stopwords active to better handle Denglisch meetings.
    stopwords = set()
    for lang_code in (detected_languages | {"de", "en"}):
        stopwords.update(_stopwords_for_language(lang_code))
    stopwords.update(
        {
            "und",
            "oder",
            "aber",
            "dass",
            "dies",
            "diese",
            "dieser",
            "jetzt",
            "noch",
            "mal",
            "okay",
            "capturing",
            "capture",
            "meeting",
            "mikrofon",
            "microphone",
            "audio",
            "plugin",
        }
    )
    topic_noise_terms = {
        "capture",
        "capturing",
        "meeting",
        "mikrofon",
        "microphone",
        "audio",
        "transkript",
        "transcript",
        "plugin",
        "funktioniert",
        "funktionieren",
        "test",
        "testen",
    }
    word_counter: Counter[str] = Counter()
    for utterance in utterances:
        for word in re.findall(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß\-]{2,}", utterance.lower()):
            if len(word) < 4:
                continue
            if word in stopwords or word in topic_noise_terms:
                continue
            word_counter[word] += 1
    top_topics = [word for word, count in word_counter.most_common(12) if count >= 2][:6]
    topic_line = ", ".join(top_topics) if top_topics else "No clear recurring topics detected."

    has_issue_signal = any(
        keyword in utterance.lower()
        for utterance in utterances
        for keyword in ("problem", "issue", "fehler", "bug", "risk", "risiko", "blocker")
    )
    has_planning_signal = any(
        keyword in utterance.lower()
        for utterance in utterances
        for keyword in ("nächste schritte", "naechste schritte", "plan", "timeline", "deadline", "priorität")
    )
    highlights: list[str] = []
    if decisions:
        highlights.append("- A concrete decision was made during the meeting.")
    if action_items:
        highlights.append("- Follow-up tasks for a next step were identified.")
    if has_issue_signal:
        highlights.append("- Risks or blockers were explicitly mentioned.")
    if has_planning_signal:
        highlights.append("- The team discussed execution planning for upcoming work.")
    if top_topics and (decisions or action_items or has_issue_signal or has_planning_signal):
        highlights.append(f"- Discussion centered on recurring themes: {topic_line}.")
    highlights = highlights[:4]

    section_decisions = decisions[:6]
    section_actions = action_items[:6]
    section_highlights = highlights
    participant_line = ", ".join(participants)
    top_speaker = speaker_counts.most_common(1)[0][0]

    summary = "### Meeting Summary\n"
    summary += f"- Participants detected: {participant_line}\n"
    summary += f"- Most active speaker: {top_speaker}\n"
    summary += f"- Main topics: {topic_line}\n"
    if section_highlights:
        summary += "\n### Highlights\n" + "\n".join(section_highlights) + "\n"
    if section_decisions:
        summary += "\n### Decisions\n" + "\n".join(section_decisions) + "\n"
    if section_actions:
        summary += "\n### Action Items\n" + "\n".join(section_actions) + "\n"
    return summary


def summarize_with_ollama(
    text: str,
    language: str = "auto",
    model: str | None = None,
    ollama_url: str | None = None,
) -> str | None:
    selected_model = str(model or os.getenv("MSCRIBE_SUMMARY_MODEL", "")).strip()
    if selected_model:
        os.environ["MSCRIBE_SUMMARY_MODEL"] = selected_model
    model = selected_model
    if not model:
        return None

    base_url = _normalize_ollama_base_url(ollama_url)
    if ollama_url:
        os.environ["MSCRIBE_OLLAMA_URL"] = base_url
    endpoint = f"{base_url}/api/generate"
    normalized_language = _normalize_meeting_language(language)
    language_instruction = (
        "Use the same language as the transcript."
        if normalized_language == "auto"
        else f"Write the summary in language code '{normalized_language}'."
    )
    prompt = (
        "You are a meeting assistant. Create a concise, useful, paraphrased meeting summary in markdown.\n"
        "Do not copy long direct quotes from transcript. Use the transcript only as source facts.\n"
        f"{language_instruction}\n"
        "Output the following sections, and omit any section that has no actual content:\n"
        "### Meeting Summary\n"
        "- 3-5 bullets with paraphrased core outcomes\n"
        "### Highlights\n"
        "- paraphrased important points\n"
        "### Decisions (optional)\n"
        "- only explicit decisions\n"
        "### Action Items (optional)\n"
        "- ONLY include tasks explicitly framed as due by next meeting/next time\n\n"
        f"Transcript:\n{text}\n"
    )
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "0s"}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read().decode("utf-8")
            data = json.loads(raw)
            result = str(data.get("response", "")).strip()
            return result or None
    except Exception as exc:
        print(f"ollama summarize failed: {exc}")
        return None


@APP.post("/start", response_model=StartResponse)
def start(req: StartRequest | None = None) -> StartResponse:
    payload = req or StartRequest()
    try:
        if payload.app_audio_only and AudioUtilities is None:
            raise RuntimeError("app_audio_only requested but pycaw is not installed")
        with STATE.lock:
            STATE.ai_quality_mode = payload.ai_quality_mode
            STATE.expected_speakers = max(1, min(8, int(payload.expected_speakers)))
            STATE.meeting_language = _normalize_meeting_language(payload.language or "auto")
            STATE.app_audio_only = bool(payload.app_audio_only)
            STATE.include_mic = bool(payload.include_mic)
            STATE.self_speaker_name = (payload.self_speaker_name or "Me").strip() or "Me"
            STATE.mic_device_contains = (payload.mic_device_contains or "").strip()
            STATE.target_apps = {str(app).strip().lower() for app in payload.target_apps if str(app).strip()}
        start_audio_capture(
            profile=payload.profile,
            prefer_loopback=payload.prefer_loopback,
            enable_diarization=payload.enable_diarization,
        )
        return StartResponse(ok=True, message=f"Capture started ({payload.profile})")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@APP.post("/stop", response_model=StopResponse)
def stop() -> StopResponse:
    audio_path: Path | None = None
    try:
        audio_path = stop_audio_capture()
        with STATE.lock:
            profile = STATE.current_profile
            ai_quality_mode = STATE.ai_quality_mode
            enable_diarization = STATE.diarization_enabled
            language = STATE.meeting_language
            include_mic = STATE.include_mic
            chunk_levels = list(STATE.chunk_levels)
        mic_rms_recent = 0.0
        loop_rms_recent = 0.0
        mic_rms_peak = 0.0
        loop_rms_peak = 0.0
        if chunk_levels:
            sample = chunk_levels[-40:]
            mic_rms_recent = float(sum(item[2] for item in sample) / len(sample))
            loop_rms_recent = float(sum(item[3] for item in sample) / len(sample))
            mic_rms_peak = float(max(item[2] for item in sample))
            loop_rms_peak = float(max(item[3] for item in sample))
        print(
            "capture levels: "
            f"chunks={len(chunk_levels)}, mic_recent={mic_rms_recent:.1f}, loop_recent={loop_rms_recent:.1f}, "
            f"mic_peak={mic_rms_peak:.1f}, loop_peak={loop_rms_peak:.1f}, include_mic={include_mic}"
        )
        mic_dominant = mic_rms_recent > 85.0 and (
            loop_rms_recent < 80.0 or mic_rms_recent >= (loop_rms_recent * 1.35)
        )
        disable_vad = include_mic and mic_dominant
        if disable_vad:
            print(
                "mic-dominant capture detected; disabling VAD for ASR "
                f"(mic_rms={mic_rms_recent:.1f}, loop_rms={loop_rms_recent:.1f})"
            )
        segments = transcribe_file(
            audio_path,
            profile=profile,
            ai_quality_mode=ai_quality_mode,
            enable_diarization=enable_diarization,
            language=language,
            disable_vad=disable_vad,
        )
        if include_mic:
            segments = apply_self_speaker_tag(segments, chunk_levels)
        speakers, samples = build_speaker_samples(segments)
        audio_samples = build_speaker_audio_samples(audio_path, segments)
        return StopResponse(segments=segments, speakers=speakers, samples=samples, audio_samples=audio_samples)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass


@APP.post("/summarize")
def summarize(req: SummarizeRequest) -> dict[str, str]:
    llm_summary = summarize_with_ollama(
        req.text,
        language=(req.language or "auto").strip().lower(),
        model=(req.model or "").strip() or None,
        ollama_url=(req.ollama_url or "").strip() or None,
    )
    if llm_summary:
        return {"summary": llm_summary}
    return {"summary": generate_structured_summary(req.text)}


@APP.post("/cleanup")
def cleanup() -> dict[str, Any]:
    released = release_gpu_resources(unload_models=True)
    return {"ok": True, "released": released}


@APP.get("/llm/models")
def llm_models(ollama_url: str | None = None) -> dict[str, list[dict[str, str]]]:
    models = list_ollama_models(ollama_url=ollama_url)
    return {"models": models}


@APP.post("/test-mic", response_model=TestMicResponse)
def test_mic(req: TestMicRequest) -> TestMicResponse:
    if pyaudio is None:
        return TestMicResponse(ok=False, device_name="", rms=0.0, peak=0.0, error="pyaudiowpatch not available")

    pa = pyaudio.PyAudio()
    try:
        mic_device = _resolve_mic_device(pa, req.mic_device_contains, include_mic=True)
        device_name = str(mic_device.get("name", "unknown"))
        sample_rate = int(mic_device.get("defaultSampleRate", 48000) or 48000)
        duration = max(0.4, min(4.0, float(req.duration_seconds)))
        frames_to_read = max(256, int(sample_rate * duration))

        channels = _pick_supported_channels(pa, mic_device["index"], preferred_max=2)
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=max(256, int(sample_rate * 0.05)),
            input_device_index=mic_device["index"],
        )
        try:
            raw = stream.read(frames_to_read, exception_on_overflow=False)
        finally:
            stream.stop_stream()
            stream.close()

        arr = _to_mono_int16(np.frombuffer(raw, dtype=np.int16), channels).astype(np.float32)
        if arr.size == 0:
            return TestMicResponse(ok=False, device_name=device_name, rms=0.0, peak=0.0, error="No samples captured")

        rms = float(np.sqrt(np.mean(np.square(arr)) + 1e-12))
        peak = float(np.max(np.abs(arr)))
        detected = rms > 50.0 or peak > 700.0
        return TestMicResponse(ok=detected, device_name=device_name, rms=rms, peak=peak)
    except Exception as exc:
        return TestMicResponse(ok=False, device_name="", rms=0.0, peak=0.0, error=str(exc))
    finally:
        pa.terminate()


@APP.get("/devices/mics")
def list_mic_devices() -> dict[str, list[dict[str, str]]]:
    devices: list[dict[str, str]] = []
    seen: set[str] = set()

    if pyaudio is not None:
        pa = pyaudio.PyAudio()
        try:
            for device_idx in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(device_idx)
                name = str(info.get("name", "")).strip()
                if not name:
                    continue
                if "loopback" in name.lower():
                    continue
                if int(info.get("maxInputChannels", 0) or 0) <= 0:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                devices.append({"id": name, "name": name})
        finally:
            pa.terminate()

    if not devices and sd is not None:
        try:
            sd_devices = sd.query_devices()
            for info in sd_devices:
                name = str(info.get("name", "")).strip()
                if not name:
                    continue
                if int(info.get("max_input_channels", 0) or 0) <= 0:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                devices.append({"id": name, "name": name})
        except Exception:
            pass

    devices.sort(key=lambda item: item["name"].lower())
    return {"devices": devices}


@APP.get("/languages")
def list_supported_languages() -> dict[str, list[dict[str, str]]]:
    return {"languages": get_supported_languages()}


@APP.get("/status")
def status() -> dict[str, Any]:
    with STATE.lock:
        recent = STATE.chunk_levels[-25:]
        mic_rms_recent = 0.0
        loop_rms_recent = 0.0
        if recent:
            mic_rms_recent = float(sum(item[2] for item in recent) / len(recent))
            loop_rms_recent = float(sum(item[3] for item in recent) / len(recent))
        return {
            "recording": STATE.is_recording,
            "backend": STATE.backend,
            "profile": STATE.current_profile,
            "ai_quality_mode": STATE.ai_quality_mode,
            "diarization_enabled": STATE.diarization_enabled,
            "meeting_language": STATE.meeting_language,
            "include_mic": STATE.include_mic,
            "target_apps": sorted(STATE.target_apps),
            "sample_rate": STATE.sample_rate,
            "channels": STATE.channels,
            "recent_mic_rms": round(mic_rms_recent, 2),
            "recent_loop_rms": round(loop_rms_recent, 2),
            "elapsed_seconds": round(time.time() - STATE.start_time, 1) if STATE.start_time else 0.0,
            "output_wav": str(STATE.output_wav) if STATE.output_wav else None,
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "companion_app:APP",
        host=os.getenv("MSCRIBE_HOST", "127.0.0.1"),
        port=int(os.getenv("MSCRIBE_PORT", "8000")),
        reload=False,
    )
