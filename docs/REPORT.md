# EmbeddingGemma 2 on Databricks — build report

**Model:** [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (740M: 270M text model + 170M vision
+ 300M audio encoders; one 768-d space for text, images, video and audio; Matryoshka 512 / 256 / 128; Apache 2.0)
**Environment:** serverless GPU (A10, environment 4), AI Runtime CLI, GPU Model Serving with env_pack; workspace
`fevm-lxejgv`
**Date:** 2026-10-14 – 2026-10-15

What went wrong, how it was diagnosed, and why the code is the way it is. The measured numbers live in the
[README](../README.md) "Measured results"; the pins live in `embeddinggemma2_core.py`.

---

## 0. Summary

| Phase | Result | In one line |
|---|---|---|
| Runtime | ✅ | The first transformers release with the model is **5.19.0**; the model card's `5.18.0.dev0` does not exist as a release with it. |
| Media inputs | ✅ | Video through transformers' path loader breaks on current PyAV; the core decodes video itself. |
| GPU test | ✅ on the 2nd run | **bf16 vectors were not unit length** (±2e-3); truncation and normalisation now run in fp32. |
| Serving | ✅ env_pack + custom PyFunc | MLflow's flavors cannot take images / video / audio; a PyFunc keeps the multimodal API. |
| Ways 1 and 2 | ✅ | Notebook job and AIR batch, both reading media from a Volume. |

**The main technical points:**
- check that a model is in a *released* library version by looking inside the wheel, not at the model card;
- with bf16, renormalise in fp32 after any truncation;
- for multimodal models, own the media decoding instead of relying on the library's file loaders.

---

## 1. Runtime versions

- **What the model card says:** `config_sentence_transformers.json` records `transformers 5.18.0.dev0`,
  `sentence_transformers 6.1.0`, `pytorch 2.14.0+cu130`, and requires `sentence-transformers>=6.1.0` ("older versions
  ignore the order of multimodal dict inputs").
- **What was checked:** the released wheels from the PyPI mirror were unzipped and searched for
  `transformers/models/embedding_gemma2/`. **5.18.0 does not have it; 5.19.0 (2026-10-06) does.** Pinning to the
  card's version number would have produced an unknown-architecture error at load.
- **torch:** 5.19.0 needs torch ≥ 2.5 (some paths ≥ 2.9). torch 2.10 / cu128 was chosen because the other
  `model-*-on-databricks` repos already run it on serverless GPU and through env_pack.
- **Lesson:** for a model that is days or weeks old, verify support in the released wheel before writing anything.

## 2. Video decoding ❌ → ✅

- **Symptom:** a video passed as a file path failed with
  `TypeError: open() got an unexpected keyword argument 'metadata_errors'`.
- **Root cause:**
  1. transformers prefers torchcodec, which needs a system FFmpeg. Without it, it warns and falls back to
     `torchvision.io.read_video`, which is deprecated.
  2. That torchvision path calls PyAV with an argument PyAV 19 no longer accepts.
- **Fix:** `embeddinggemma2_core.decode_video`.
  - It decodes with PyAV (whose wheels bundle FFmpeg) and keeps only the frames at the processor's fps (1 by default;
    `processing_kwargs.video.fps` overrides it).
  - It passes them as sentence-transformers' video dict (`array` + `video_metadata`). The processor then samples at
    the same fps from that metadata, so it keeps these frames (its `max_frames` cap still applies).
- **Side benefits:**
  - No dependency on torchvision or torchcodec for video.
  - No temporary files.
  - Only the sampled frames are held in memory, so a long video does not load every frame.
- PyAV is pinned to 16.1.0, the version the decoder was tested with.

## 3. bf16 vectors not unit length ❌ → ✅

- **Symptom:** the first `notebooks/test_serving_model` run on the A10 failed its norm check:
  `AssertionError: ('doc', 768)`. The install, the model load and the first row had passed.
- **Diagnosis (before spending another GPU run):**
  - The model's last module (`2_Normalize`) runs in the model dtype, bf16 on the GPU.
  - Locally, in bf16 on the CPU, sentence-transformers' `normalize_embeddings=True` returned norms 1.0018 and 0.9987.
    bf16 holds about three significant digits, so its unit length is only good to ~2e-3.
- **Fix:** `Embedder._encode` now:
  1. takes the raw output;
  2. casts it to fp32;
  3. truncates it to `truncate_dim`;
  4. L2-normalises it in fp32.

  Norms are then 1.0 to float precision.
- **Quality:** bf16 vs fp32 vectors of the same text: cosine 0.99998. bf16 is still the GPU default.
- **Why it matters beyond the test:** with Matryoshka truncation, a vector must be renormalised after it is cut.
  Skipping it silently degrades ranking (model card), and doing it in bf16 leaves the same ±2e-3 error.
- **float16** is refused by `Embedder`: the model's activations overflow it and come back NaN (model card).

## 4. Serving: a custom PyFunc, not an MLflow flavor

Both flavors were considered; `mlflow.sentence_transformers` was tried locally (MLflow 3.12.0).

| | `mlflow.transformers` | `mlflow.sentence_transformers` (tried) | custom PyFunc (adopted) |
|---|---|---|---|
| Embedding | `feature-extraction` returns token states; pooling, the 512→768 projection and normalisation are left to the caller | correct (the SentenceTransformer pipeline) | correct |
| Images / video / audio | no | **no**: the input is strings only | yes, also interleaved |
| `prompt_name` / `truncate_dim` | — | **no**: the logged signature has no params | via `params_json` |
| Per-row errors | — | no: one bad row fails the request | yes |
| Weights | — | bundled (548 MB, text-only config) | downloaded from the Hub at load |
| OpenAI-style `llm/v1/embeddings` | — | the task is recorded in the metadata, but the loaded pyfunc had `task=None` locally and returned bare arrays | no |

The flavor's one real advantage is an OpenAI-compatible embeddings endpoint, which would be convenient for Vector Search
and `ai_query`. That is not worth losing the multimodal input, which is the reason to use this model. The PyFunc also
keeps the `text` + `params_json` contract of the other `model-*-on-databricks` repos.

## 5. Serving: env_pack, and the traps avoided

The endpoint is registered with env_pack (`notebooks/register_serving.py`) because the model needs torch 2.10 / cu128.
The following were **known from Irodori-TTS** (its `docs/REPORT.md` §2) and handled before the first run; none of
them occurred here:

| Trap | Handled by |
|---|---|
| Standard GPU serving replaces torch with its own build (`libcudart.so.12` not found) | env_pack: the registering notebook's environment is packed and restored as is |
| env_pack on classic compute: `is_client_image=False` | registration runs as a serverless GPU job (`environments[].spec.client: "4"`) |
| Environment tar upload times out | `databricks-sdk>=0.102.0` pinned |
| `ValueError` about `hf_transfer` on download | `HF_HUB_ENABLE_HF_TRANSFER=0` set before huggingface_hub is imported |
| `partially initialized module 'torchvision'` | `torchvision==0.25.0` (the build for torch 2.10) pinned |
| Notebook kernel breaks on protobuf | `protobuf>=5.29.5,<6` pinned |

The checkpoint is not bundled: the endpoint downloads it (≈ 1.5 GB) at load, as Irodori does. This keeps the
registration small, at the cost of Hub egress and download time on every cold start.

## 6. API design

- **Two string columns, `text` + `params_json`.** MLflow schema enforcement cannot drop a key, new keys need no
  re-registration, and the contract matches the other model repos.
- **Batching with isolation.**
  - Rows that agree on `truncate_dim` / `normalize` / `processing_kwargs` are encoded in one `encode` call.
  - If that call fails, only that group is retried row by row. Only the bad row reports an `error`, and the HTTP
    status stays 200.
  - Validation errors (unknown prompt, bad dimension, undecodable media, a modality the endpoint did not load) are
    raised per row before encoding.
- **Encoder sets per registration** (`MODALITIES`: `all` 740M, `text` 270M, `text+vision`, `text+audio`).
  - Every set embeds into the same space. For example, a cheap `text` endpoint can embed queries against a corpus
    embedded by an `all` endpoint.
- **The model card's rules applied by the core:**
  - task prefixes go on text only, and are dropped from media-only rows (noted in `messages`);
  - `title` builds the `title: {title} | text: ` document prefix;
  - missing `<|image|>` / `<|video|>` / `<|audio|>` placeholders are appended, and too many is an error;
  - `truncate_dim` is limited to 768 / 512 / 256 / 128.
- **Self-description:** `{"__list_params__": true}` returns the model's prompt table, the accepted inputs, the
  precision and the pinned revision. Clients read prompt names from there rather than from documentation.

## 7. How it was verified (and kept cheap)

1. **Locally first**, in a separate venv with the pinned versions (Apple-silicon CPU, fp32):
   - `Embedder` with every input kind and bad rows;
   - `generate.py`;
   - the PyFunc through `log_model` → `load_model` → `predict`;
   - both notebooks, run with a stubbed `dbutils` and the install cell skipped.

   This found the transformers version and the video decoder problems before any GPU time was spent.
2. **On Databricks, in order:**
   1. `test_serving_model`;
   2. registration;
   3. the endpoint;
   4. the notebook job and the AIR batch.

   The only failure (§3) was reproduced locally before the second GPU run.
3. **Media without binaries in the repo:** `scripts/sample_media.py` generates a red / blue circle PNG, a 440 Hz WAV
   and a red-bar MP4 for the tests and the notebook demo. For ways 1 and 2 the same files were uploaded to a Volume to
   exercise `*_path` and `media_dir`.

## 8. Known limits and open items

- **The modality gap.** A text query scores text documents higher than media of the same subject. With the query
  "a red circle":
  - the red-circle image scored 0.757 and the blue-circle image 0.684, so the ranking within images is right;
  - but the red-circle image ranked below the *text* "青い円が描かれた紙。" (0.776).

  Rank within one modality, or calibrate thresholds per modality.
- **Request size.** Media travel as base64 in the JSON body, so large videos can exceed the Model Serving payload
  limit. Embed them from a Volume with way 1 or 2.
- **Cold start from scale-to-zero** has not been measured. It includes the ≈ 1.5 GB Hub download.
- **Not tried:** Vector Search integration, CPU serving of the 270M `text` set, an OpenAI-compatible wrapper.

## 9. Lessons that carry over to other models

- Check support in the released wheel, not the model card's version string.
- Pin every library that touches the input path (here PyAV), and test the decoder you ship.
- In bf16, do any post-processing that needs precision (normalisation, truncation) in fp32.
- Test the serving PyFunc in-process (`notebooks/test_serving_model`) before registering. Run the notebooks locally
  with a `dbutils` stub before spending GPU time.
- MLflow flavors are convenient only when the model's inputs fit them; for multimodal inputs, a PyFunc is simpler.
