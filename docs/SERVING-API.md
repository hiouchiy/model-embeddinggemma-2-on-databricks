# Serving API — EmbeddingGemma 2 (multimodal embeddings)

The single source for the API of the Model Serving endpoint this repo deploys (e.g. `embeddinggemma2`). How to
register and deploy it: [README](../README.md) "3. Model Serving".

Model: [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) (740M, Apache 2.0). It maps text
(incl. code), images, video and audio, alone or interleaved, into one 768-dimensional space, with Matryoshka
truncation to 512 / 256 / 128. It is served with env_pack (GPU Express Deployment), which restores the torch 2.10 /
cu128 environment it was registered from. The PyFunc is
[`serving/embeddinggemma2_pyfunc.py`](../serving/embeddinggemma2_pyfunc.py); the request handling is
`embeddinggemma2_core.Embedder` ([`embeddinggemma2_core.py`](../embeddinggemma2_core.py)).

## 0. Calling it

```
POST https://<WORKSPACE_HOST>/serving-endpoints/<ENDPOINT>/invocations
Authorization: Bearer <TOKEN>
Content-Type: application/json
```

- **Input format:** MLflow PyFunc `dataframe_records`. **One row is one embedding.** Rows that share `truncate_dim`,
  `normalize` and `processing_kwargs` are encoded as one batch.
- **Output format:** `{"predictions": [...]}`, one element per row, in order.
- **Errors:**
  - A bad row (unknown prompt, undecodable image, a modality the endpoint did not load, ...) still returns HTTP 200,
    with the reason in that row's `error`; the other rows are unaffected.
  - HTTP 4xx / 5xx come from the platform: auth, starting up, schema mismatch, timeout, payload too large.
  - A `params_json` that is not valid JSON fails only its own row.
- **Cold start:** the first request after scale-to-zero starts the container and downloads the checkpoint (≈ 1.5 GB).
  - Use a client timeout of 600–900 s.
  - On 503 / 504 / 429, wait 60 s and retry.
  - `databricks serving-endpoints query` times out after about 60 s, so the first call often fails; call REST directly.
- **Model Serving in general** (auth, input formats): Databricks docs,
  [Query serving endpoints for custom models](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints).

## 1. Request

| Column | Type | Required | Description |
|---|---|---|---|
| `text` | string | ✅ (as a column) | The text to embed. `""` for a media-only row. May contain `<\|image\|>` / `<\|video\|>` / `<\|audio\|>` placeholders to place media inside the text. |
| `params_json` | string (JSON) | ✅ (as a column) | Task prefix, size and media as a JSON string (table below). `""` or `"{}"` embeds `text` as is at 768 d. |

`params_json` is a single JSON string so that MLflow schema enforcement never drops a key.

### Listing the parameters (no embedding)

```json
{"dataframe_records": [{"text": "", "params_json": "{\"__list_params__\": true}"}]}
```

The response's `messages` holds `{"model", "revision", "modalities", "inputs", "precision", "dims", "prompts",
"supported_params", "defaults"}`. `inputs` tells which media this endpoint accepts; `prompts` is the model's full
prompt table (**the source for prompt names**).

### `params_json` keys

**Task prefix (text only)**

The model is trained with short task prefixes. Use the query prefix for queries and `Document` (or `title`) for the
corpus; use the same symmetric prefix on both sides for classification, clustering and similarity. Media take no
prefix: for a media-only row the prefix is dropped (noted in `messages`).

| Key | Type | Default | Description |
|---|---|---|---|
| `prompt_name` | string | none | A name from the model's prompts. Queries: `SearchQuery`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`. Corpus: `Document` (`title: none \| text: `). Symmetric: `Classification`, `Clustering`, `SentenceSimilarity`. The full table: `__list_params__`. |
| `title` | string | — | A corpus item with a real title: the prefix becomes `title: {title} \| text: `. Overrides `prompt_name`. |
| `prompt` | string | — | A raw prefix, prepended as is. Overrides `prompt_name`. |

The prefix actually used comes back in `applied_params.prefix`.

**Media** (each key takes one base64 string or a list of them)

| Key | Description |
|---|---|
| `image_b64` | Images (PNG, JPEG, WebP, ...). 280 tokens each by default. Needs the vision encoder. |
| `video_b64` | Videos (MP4, MOV, WebM, ... anything PyAV decodes). Sampled at 1 frame per second, up to 32 frames (uniform beyond that), 140 tokens per frame. Needs the vision encoder. |
| `audio_b64` | Audio (WAV, FLAC, OGG, MP3). Decoded to mono and resampled to 16 kHz; 25 tokens per second. Needs the audio encoder. |
| `image_path` / `video_path` / `audio_path` | A path (local, `/Volumes/...`) or an http(s) URL **as seen by the server**. For the notebook and the AIR job; from an endpoint only public URLs work. |

How a row is combined:
- **One media item, no text:** that item's embedding.
- **Text and media, or several items:** **one joint embedding** of all of them. Each placeholder in `text` is filled from
  its modality's list, in order. Placeholders the text lacks are appended at the end (noted in `messages`); more
  placeholders than items is an error.
- All modalities share one 8,192-token context per row.

**Size and processing**

| Key | Type | Default | Description |
|---|---|---|---|
| `truncate_dim` | int | 768 | Matryoshka size: `768`, `512`, `256` or `128`. Near-lossless down to 256; 128 is best kept to text-only workloads (model card). **Queries and corpus must use the same size.** |
| `normalize` | bool | true | L2-normalise (after truncation). Leave it on for cosine / dot-product search. |
| `processing_kwargs` | object | — | Per-modality processor overrides, passed to sentence-transformers. E.g. `{"image": {"max_soft_tokens": 560}}` (vision budget 70–1120 tokens per image: more is finer and slower), `{"video": {"fps": 2, "max_frames": 58}}`. |

Unknown keys are ignored (they show up in `applied_params`).

## 2. Response (one per row)

| Field | Type | Description |
|---|---|---|
| `embedding` | array of float | The vector, `truncate_dim` long, unit length unless `normalize` is false. Empty on error. |
| `dim` | int | Its length |
| `modality` | string | What was embedded: `text`, `image`, `text+image`, `video`, `audio`, ... |
| `embed_seconds` | float | Encode time of the batch this row was in (on the GPU) |
| `messages` | string (JSON) | Notes (prefix dropped, placeholders appended). For `__list_params__`, the parameter list. |
| `applied_params` | string (JSON) | The parameters applied, with `prefix`, `truncate_dim` and `<kind>_count`; base64 payloads are left out |
| `error` | string | `""` on success; `"<ExceptionType>: <message>"` on failure |

## 3. Examples

```bash
curl -sS -X POST "https://<WORKSPACE_HOST>/serving-endpoints/<ENDPOINT>/invocations" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"dataframe_records":[
        {"text":"What causes the northern lights?","params_json":"{\"prompt_name\":\"SearchQuery\"}"},
        {"text":"オーロラは太陽からの荷電粒子が原因で光る。","params_json":"{\"title\":\"オーロラ\"}"}]}' \
  | python3 -c "import sys,json;p=json.load(sys.stdin)['predictions'];print([r['error'] or r['dim'] for r in p], sum(a*b for a,b in zip(p[0]['embedding'],p[1]['embedding'])))"
```

Cross-modal search (Python; `invoke()` wraps the call in §0):

```python
import base64, json
b64 = lambda path: base64.b64encode(open(path, "rb").read()).decode()
P = json.dumps
res = invoke("<ENDPOINT>", {"dataframe_records": [
    {"text": "a dog running on the beach", "params_json": P({"prompt_name": "SearchQuery", "truncate_dim": 256})},
    {"text": "", "params_json": P({"image_b64": b64("dog.jpg"), "truncate_dim": 256})},
    {"text": "", "params_json": P({"video_b64": b64("beach.mp4"), "truncate_dim": 256})},
    {"text": "", "params_json": P({"audio_b64": b64("waves.wav"), "truncate_dim": 256})},
    {"text": "Waterproof running shoes. <|image|> Grip test on wet rock: <|video|>",   # one joint embedding
     "params_json": P({"title": "Trail shoe", "image_b64": b64("shoe.jpg"), "video_b64": b64("grip.mp4"), "truncate_dim": 256})},
]})
q, *docs = [r["embedding"] for r in res["predictions"]]
scores = [sum(a * b for a, b in zip(q, d)) for d in docs]
```

## 4. Performance and limits

- **Measured** (A10, warm, 2026-10-15): 1 text row 0.24–0.32 s round trip, 64 text rows 0.54 s, 1 image 0.32 s, a 3 s
  video 0.41 s. GPU memory peak ≈ 1.4–1.7 GiB.

- **Context:** 8,192 tokens per row, shared by all modalities: ≈ 29 images, ≈ 58 video frames or ≈ 327 s of audio at
  the defaults.
- **Request size:** media travel as base64 in the JSON body, which Model Serving caps (16 MB per request at the time of
  writing). Send large videos one per request, or embed them with the notebook / AIR job from a Volume (`video_path`).
- **Scores across modalities:** a text query scores text documents higher than media. In the local test (query
  "a red circle"), the red-circle image scored 0.758, below the *blue*-circle text "青い円が描かれた紙。" at 0.776, though
  above the blue-circle image (0.684); the endpoint gives the same (0.757 / 0.684). Rank within one modality, or
  calibrate thresholds per modality.
- **Weights:** downloaded from the HF Hub at load, so the serving environment needs outbound access.

## 5. Re-registering

env_pack packs the environment of a serverless GPU notebook, so registration cannot run locally. Run
[`notebooks/register_serving.py`](../notebooks/register_serving.py) as a serverless GPU job (`GPU_1xA10`, client
`4`); the `uc_model` parameter sets the target and `modalities` the encoders. Steps: [README](../README.md)
"3. Model Serving".
