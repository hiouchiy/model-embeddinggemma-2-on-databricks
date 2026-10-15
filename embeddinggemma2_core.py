"""EmbeddingGemma 2 core, shared by the notebook, the AI Runtime batch job and the serving model.

One place for: the runtime pins, the pinned checkpoint, the encoder sets (MODALITIES), and the request contract
(`text` + `params_json` -> one embedding; docs/SERVING-API.md). The three ways to run (notebooks/generate.py,
air_job/generate.yaml via generate.py, serving/embeddinggemma2_pyfunc.py) only add their own input and output handling
on top.

Importing this module needs neither torch nor sentence-transformers; they are imported when an Embedder is built.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

# Serverless GPU (notebooks and AI Runtime) presets HF_HUB_ENABLE_HF_TRANSFER=1 without installing hf_transfer, and
# every Hub download then raises. huggingface_hub reads it at import, so switch it off before anything imports it.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Apache 2.0, not gated. 740M in total: a 270M text model plus a 170M vision encoder (images and video) and a 300M
# audio encoder, which can each be left unloaded (MODALITIES).
HF_MODEL = "google/embeddinggemma-2"
HF_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"   # 2026-10 (pinned)

# Which encoders are loaded: the `config_kwargs` of SentenceTransformer (model card, "Selective Encoder Loading").
# Video goes through the vision encoder, so "text+vision" embeds images and video.
MODALITIES = {
    "all": dict(config_kwargs={}, inputs=("text", "image", "video", "audio"), params="740M"),
    "text": dict(config_kwargs={"vision_config": None, "audio_config": None}, inputs=("text",), params="270M"),
    "text+vision": dict(config_kwargs={"audio_config": None}, inputs=("text", "image", "video"), params="440M"),
    "text+audio": dict(config_kwargs={"vision_config": None}, inputs=("text", "audio"), params="570M"),
}
DEFAULT_MODALITIES = "all"
# Matryoshka (MRL) output sizes. Re-normalised after truncation unless `normalize` is false.
DIMS = (768, 512, 256, 128)
AUDIO_SAMPLE_RATE = 16000   # the audio encoder takes mono 16 kHz

# The runtime: torch 2.10 / cu128 (the stack the other model repos run on serverless GPU), plus the first releases that
# know the model. scripts/install.sh installs it.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# - transformers 5.19.0 is the first release with `embedding_gemma2` (5.18.0 does not have it).
# - sentence-transformers>=6.1.0 is the model's own requirement (config_sentence_transformers.json): older versions
#   ignore the order of multimodal dict inputs.
# - torchvision is pinned to the build for torch 2.10: the GPU image ships one for its base torch 2.7.1 that is
#   ABI-broken against 2.10 and cannot be uninstalled from the base layer.
# - av (PyAV, bundles FFmpeg) decodes video (decode_video); librosa decodes and resamples audio.
# - databricks-sdk>=0.102.0: env_pack uploads a large environment tar; older sdks time out after 5 min.
# - protobuf>=5.29.5,<6 keeps the serverless notebook kernel (pyspark) alive.
RUNTIME_REQUIREMENTS = [
    "torch==2.10.0", "torchvision==0.25.0", "transformers==5.19.0", "sentence-transformers==6.1.0",
    "huggingface-hub>=1.31.0,<2", "safetensors>=0.8.0", "pillow>=10", "av==16.1.0",
    "librosa==1.0.0", "soundfile>=0.12.0", "pyarrow>=15", "pandas>=2.2",
    "mlflow==3.12.0", "databricks-sdk>=0.102.0", "protobuf>=5.29.5,<6",
]
# The serving model's own pip requirement; env_pack restores the rest of the registering environment as is.
SERVING_PIP_REQUIREMENTS = ["mlflow==3.12.0"]

MEDIA = ("image", "video", "audio")
PLACEHOLDER = {"image": "<|image|>", "video": "<|video|>", "audio": "<|audio|>"}
# params_json keys that change how a batch is encoded; rows that agree on them are encoded together.
_BATCH_KEYS = ("truncate_dim", "normalize", "processing_kwargs")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def modalities_name(name: str | None = None) -> str:
    """A key of MODALITIES: `name`, else $EMBEDDINGGEMMA_MODALITIES, else the default."""
    name = name or os.environ.get("EMBEDDINGGEMMA_MODALITIES") or DEFAULT_MODALITIES
    if name not in MODALITIES:
        raise ValueError(f"modalities must be one of {sorted(MODALITIES)}")
    return name


def parse_params(params_json: Any) -> dict:
    """The `params_json` column: a JSON object string, a dict, or empty."""
    if isinstance(params_json, str) and params_json.strip():
        return json.loads(params_json)
    if isinstance(params_json, dict):
        return params_json
    return {}


def _as_list(v: Any) -> list:
    if v is None or v == "":
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _read_bytes(path_or_url: str) -> bytes:
    """A local or /Volumes path, or an http(s) URL."""
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=120) as r:
            return r.read()
    return Path(path_or_url).read_bytes()


def decode_video(data: bytes, fps: float) -> dict:
    """A video file's bytes -> frames sampled at `fps`, as sentence-transformers' video dict (array + metadata).

    Decoded here with PyAV, keeping only the sampled frames. transformers' own path-based loading falls back to the
    deprecated torchvision decoder (torchcodec needs system FFmpeg), which breaks across PyAV releases. The processor
    then samples at the same fps from this metadata, so it keeps these frames (max_frames caps them, uniformly).
    """
    import av
    import numpy as np
    frames, next_t = [], 0.0
    with av.open(io.BytesIO(data)) as c:
        stream = c.streams.video[0]
        for frame in c.decode(stream):
            t = float(frame.time or 0.0)
            if t + 1e-6 >= next_t:
                frames.append(frame.to_ndarray(format="rgb24"))
                next_t += 1.0 / fps
    if not frames:
        raise ValueError("no video frames decoded")
    return {"array": np.stack(frames), "video_metadata": {
        "fps": fps, "total_num_frames": len(frames), "duration": len(frames) / fps,
        "frames_indices": list(range(len(frames)))}}


class Embedder:
    """One loaded EmbeddingGemma 2. `handle_rows` returns the endpoint's response rows."""

    def __init__(self, modalities: str | None = None, precision: str | None = None):
        """modalities: a key of MODALITIES; default $EMBEDDINGGEMMA_MODALITIES, else "all"."""
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 or fp32 only: the model's activations overflow float16 and come back NaN or silently wrong (model card).
        if precision is None:
            precision = "bf16" if self.device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False) else "fp32"
        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32 (float16 breaks this model)")
        self.precision = precision
        self.modalities = modalities_name(modalities)
        spec = MODALITIES[self.modalities]
        self.inputs = spec["inputs"]
        self._st = SentenceTransformer(
            HF_MODEL, revision=HF_REVISION, device=self.device,
            model_kwargs={"dtype": torch.bfloat16 if precision == "bf16" else torch.float32},
            config_kwargs=spec["config_kwargs"])
        self.prompts = dict(self._st.prompts)
        self.dim = int(self._st.get_embedding_dimension())
        video_processor = getattr(getattr(self._st[0], "processor", None), "video_processor", None)
        self.video_fps = float(getattr(video_processor, "fps", None) or 1.0)   # 1 fps (processor_config.json)

    def list_params(self) -> dict:
        """The `{"__list_params__": true}` response row (no embedding)."""
        return self._row([], messages={
            "model": HF_MODEL, "revision": HF_REVISION, "modalities": self.modalities, "inputs": list(self.inputs),
            "precision": self.precision, "dims": list(DIMS), "prompts": self.prompts,
            "supported_params": ["prompt_name", "prompt", "title", "truncate_dim", "normalize", "processing_kwargs",
                                 *[f"{m}_b64" for m in MEDIA], *[f"{m}_path" for m in MEDIA]],
            "defaults": {"prompt_name": None, "truncate_dim": DIMS[0], "normalize": True}})

    @staticmethod
    def _row(embedding: list, modality: str = "", seconds: float = 0.0, applied: dict | None = None,
             messages: Any = None, error: str = "") -> dict:
        return {"embedding": embedding, "dim": len(embedding), "modality": modality,
                "embed_seconds": round(seconds, 4),
                "messages": json.dumps(messages if messages is not None else [], ensure_ascii=False, default=str),
                "applied_params": json.dumps(applied or {}, ensure_ascii=False, default=str), "error": error}

    # -- one row -> one model input -------------------------------------------------------------------------------

    def _decode(self, kind: str, data: bytes, video_fps: float) -> Any:
        if kind == "image":
            from PIL import Image
            return Image.open(io.BytesIO(data)).convert("RGB")
        if kind == "audio":
            import librosa
            wav, _ = librosa.load(io.BytesIO(data), sr=AUDIO_SAMPLE_RATE, mono=True)
            return {"array": wav, "sampling_rate": AUDIO_SAMPLE_RATE}
        return decode_video(data, video_fps)

    def build_input(self, text: str, params: dict) -> tuple[Any, str, dict, list[str]]:
        """(model input, modality label, applied params, notes) for one row. Raises ValueError on a bad request."""
        notes: list[str] = []
        media: dict[str, list] = {}
        video_fps = float(((params.get("processing_kwargs") or {}).get("video") or {}).get("fps") or self.video_fps)
        for kind in MEDIA:
            items = [base64.b64decode(b) for b in _as_list(params.get(f"{kind}_b64"))]
            items += [_read_bytes(p) for p in _as_list(params.get(f"{kind}_path"))]
            if items:
                if kind not in self.inputs:
                    raise ValueError(f"{kind} input needs an encoder this model was loaded without "
                                     f"(modalities={self.modalities}; see MODALITIES)")
                media[kind] = [self._decode(kind, d, video_fps) for d in items]
        text = "" if text is None else str(text)
        if not text and not media:
            raise ValueError("empty input: give `text` and/or one of image_b64 / video_b64 / audio_b64 (or *_path)")

        # Task prefixes apply to text only (model card): a title builds the document prefix, `prompt` is a raw prefix,
        # `prompt_name` a key of the model's prompts.
        prefix, prompt_name = "", params.get("prompt_name")
        if params.get("title"):
            prefix = f"title: {params['title']} | text: "
        elif params.get("prompt") is not None:
            prefix = str(params["prompt"])
        elif prompt_name:
            if prompt_name not in self.prompts:
                raise ValueError(f"unknown prompt_name {prompt_name!r}; one of {sorted(self.prompts)}")
            prefix = self.prompts[prompt_name]
        if prefix and not text:
            notes.append("prompt ignored: media-only input takes no task prefix")
            prefix = ""

        applied = {k: v for k, v in params.items() if not k.endswith("_b64")}
        applied.update({f"{k}_count": len(v) for k, v in media.items()})
        if prefix:
            applied["prefix"] = prefix
        modality = "+".join((["text"] if text else []) + list(media))

        if not media:
            return prefix + text, modality, applied, notes
        if not text and len(media) == 1 and len(next(iter(media.values()))) == 1:
            return next(iter(media.values()))[0], modality, applied, notes
        # Interleaved: each placeholder in the text is filled from its modality's list, in order. Placeholders the text
        # does not have are appended, so text + media without placeholders still makes one joint embedding.
        body = prefix + text
        for kind, items in media.items():
            missing = len(items) - body.count(PLACEHOLDER[kind])
            if missing < 0:
                raise ValueError(f"text has more {PLACEHOLDER[kind]} placeholders than {kind} items ({len(items)})")
            if missing:
                body = (body + " " + " ".join([PLACEHOLDER[kind]] * missing)).strip()
                notes.append(f"appended {missing} {PLACEHOLDER[kind]} placeholder(s)")
        return {"text": body, **{k: (v if len(v) > 1 else v[0]) for k, v in media.items()}}, modality, applied, notes

    # -- encoding -------------------------------------------------------------------------------------------------

    def _encode(self, inputs: list, key: tuple) -> tuple[list[list[float]], float]:
        import torch
        truncate_dim, normalize, pk = key
        t0 = time.perf_counter()
        with torch.inference_mode():
            emb = self._st.encode(inputs, normalize_embeddings=False, convert_to_tensor=True, show_progress_bar=False,
                                  **({"processing_kwargs": json.loads(pk)} if pk else {}))
            # Truncate and re-normalise in fp32: in bf16 the unit length is only good to ~1e-2.
            emb = emb.float()[:, :truncate_dim]
            if normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu().tolist(), time.perf_counter() - t0

    def handle_rows(self, rows: list[tuple[Any, Any]]) -> list[dict]:
        """(text, params_json) request rows -> response rows, in order. Errors come back in each row's `error`.

        Rows that agree on truncate_dim / normalize / processing_kwargs are encoded as one batch; if a batch fails,
        its rows are retried one by one so only the bad row reports the error.
        """
        out: list[dict | None] = [None] * len(rows)
        groups: dict[tuple, list[tuple[int, Any, str, dict, list]]] = {}
        for i, (text, params_json) in enumerate(rows):
            try:
                p = parse_params(params_json)
                if p.get("__list_params__"):  # agent self-discovery, no embedding
                    out[i] = self.list_params()
                    continue
                dim = int(p.get("truncate_dim") or DIMS[0])
                if dim not in DIMS:
                    raise ValueError(f"truncate_dim must be one of {list(DIMS)}")
                pk = p.get("processing_kwargs")
                key = (dim, bool(p.get("normalize", True)), json.dumps(pk, sort_keys=True) if pk else "")
                x, modality, applied, notes = self.build_input(text, p)
                applied["truncate_dim"] = dim
                groups.setdefault(key, []).append((i, x, modality, applied, notes))
            except Exception as e:
                out[i] = self._row([], error=f"{type(e).__name__}: {e}")

        for key, items in groups.items():
            try:
                embs, dt = self._encode([x for _, x, *_ in items], key)
                for (i, _, modality, applied, notes), e in zip(items, embs):
                    out[i] = self._row(e, modality, dt, applied, notes)
            except Exception:
                for i, x, modality, applied, notes in items:
                    try:
                        (e,), dt = self._encode([x], key)
                        out[i] = self._row(e, modality, dt, applied, notes)
                    except Exception as e:
                        out[i] = self._row([], modality, applied=applied, error=f"{type(e).__name__}: {e}")
        return out  # type: ignore[return-value]

    def handle(self, text: str, params_json: Any) -> dict:
        """One request row: `text` + `params_json` -> one response row."""
        return self.handle_rows([(text, params_json)])[0]
