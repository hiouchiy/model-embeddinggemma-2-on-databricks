# EmbeddingGemma 2 on Databricks — multimodal embeddings (text, images, video, audio) on one A10 GPU

## What this is

Run Google's multimodal embedding model **[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)**
(`google/embeddinggemma-2`, 740M, Apache 2.0) on **Databricks**. It maps text (incl. code, 100+ languages), images,
video and audio, alone or interleaved in one input, into **one shared 768-dimensional space**, so a text query can
retrieve images, videos or sounds. It is trained with Matryoshka Representation Learning: vectors can be cut to 512,
256 or 128 dimensions.

One API exposes the whole model:

- **any input kind** per row: text, `image_b64`, `video_b64`, `audio_b64` (one item or a list), or text with
  `<|image|>` / `<|video|>` / `<|audio|>` placeholders for one joint embedding;
- **task prefixes** (`prompt_name`: `SearchQuery`, `Document`, `Classification`, ...; or `title`), **Matryoshka
  size** (`truncate_dim`), and processor overrides (vision token budget, video fps), discovered with
  `{"__list_params__": true}`.

One request row is one embedding (8,192 tokens shared by all modalities).

It needs transformers 5.19 (the first release with the model) and sentence-transformers 6.1 on torch 2.10 / CUDA 12.8.
The endpoint is registered with **env_pack (GPU Express Deployment)**, which restores the registering notebook's
environment as is, the same route as the other `model-*-on-databricks` repos.

**Licence:** Apache 2.0, with the model card's requirement to follow the Gemma Prohibited Use Policy. See
[Licence](#licence).

## Choosing the encoders

The model is a 270M text model plus two encoders that can be left unloaded (model card, "Selective Encoder Loading").
Every way to run takes the same `modalities` value; for serving, it is chosen once per registration and each set gets
its own UC model and endpoint.

| `modalities` | Inputs | Size | Use it for |
|---|---|---|---|
| `all` (default) | text, images, video, audio | 740M | cross-modal search over everything |
| `text` | text (incl. code) | 270M | RAG / document search, classification, clustering |
| `text+vision` | text, images, video | 440M | image / video search, visual documents |
| `text+audio` | text, audio | 570M | spoken-query search, sound search |

A request with a media kind the endpoint did not load returns that row's `error` and leaves the other rows alone.
The vectors of every set live in the same space: a `text` endpoint's query vectors match an `all` endpoint's image
vectors.

## Ways to run

| | Way | Entry point | Use it for |
|---|---|---|---|
| 1 | **Notebook on serverless GPU** | [`notebooks/generate.py`](notebooks/generate.py) | interactive: embed a small corpus and media folder, rank it against a query |
| 2 | **Batch with the AI Runtime CLI** | [`air_job/generate.yaml`](air_job/generate.yaml) (`air run`) | an unattended job: a JSONL corpus (text, or media paths on a Volume) → Parquet on a Volume |
| 3 | **Model Serving** (GPU endpoint) | [`notebooks/register_serving.py`](notebooks/register_serving.py), [`serving/`](serving/) | a REST endpoint for apps; API in [docs/SERVING-API.md](docs/SERVING-API.md) |

All three use the same core, [`embeddinggemma2_core.py`](embeddinggemma2_core.py). It holds:
- the runtime pins and why they are what they are;
- the pinned checkpoint revision (`HF_REVISION`) and the encoder sets (`MODALITIES`);
- `Embedder`: the request contract, the serving endpoint's request handling (media decoding, prefixes, placeholder
  handling, batching, per-row errors).

The notebooks and the AIR job install the runtime with the same [`scripts/install.sh`](scripts/install.sh).

## Prerequisites and configuration

| Requirement | Notes |
|---|---|
| A **Databricks workspace** | **Unity Catalog**, **serverless GPU / AI Runtime** (A10) and serverless jobs, and **GPU Model Serving** (`GPU_MEDIUM` = A10). |
| **Permissions** | `CREATE MODEL` on a UC schema, write to a Volume, create serving endpoints. |
| **Hub egress** | The weights (≈ 1.5 GB) come from the Hugging Face Hub at run time. The model is not gated: no token needed. |
| **Databricks CLI** (`databricks`) | authenticated: `databricks auth login --host https://<workspace> --profile <PROFILE>`. |
| **AI Runtime CLI** (`air`) | Only for way 2. |

Placeholders used below — replace them with your values:

| Placeholder | Meaning | Where it is set |
|---|---|---|
| `<PROFILE>` | CLI profile | every `--profile` |
| `<CATALOG>.<SCHEMA>` | where the model is registered and the Volume lives | the `uc_model` parameter of `notebooks/register_serving` |
| `<VOLUME>` | UC Volume for embeddings and media: `/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>` | the notebook's `output_dir` / `media_dir` widgets; [`air_job/generate.yaml`](air_job/generate.yaml) `DEST` |
| `<ENDPOINT>` | serving endpoint name, e.g. `embeddinggemma2` | way 3 |
| `<you>` | your workspace user name (`databricks current-user me`) | workspace paths |

```bash
databricks volumes create <CATALOG> <SCHEMA> <VOLUME> MANAGED --profile <PROFILE>   # once
```

Layout:

```
embeddinggemma2_core.py          runtime pins, pinned checkpoint, encoder sets (MODALITIES); Embedder (shared by 1–3)
scripts/install.sh               runtime install on serverless GPU (notebooks and the AIR job)
scripts/sample_media.py          synthetic image / tone / video (base64) for the tests and the notebook demo
notebooks/generate.py            way 1
notebooks/test_serving_model.py  runs serving/embeddinggemma2_pyfunc.py in-process on a GPU, before registering
notebooks/register_serving.py    way 3: install → smoke test → log + register to UC with env_pack
air_job/generate.yaml            way 2;  generate.py is the CLI it runs
serving/embeddinggemma2_pyfunc.py  the MLflow PyFunc (models-from-code): the whole API
examples/corpus.jsonl            sample corpus (Japanese / English / code)
docs/SERVING-API.md              endpoint API (single source)
```

## 1. Notebook on serverless GPU

1. Put the repo in the workspace, so the notebook has `../embeddinggemma2_core.py` and `../scripts/` next to it:
   - **Git folder** (recommended): *Workspace → Create → Git folder* with this repo's URL; or
   - **sync** a local clone: `databricks sync . /Workspace/Users/<you>/model-embeddinggemma-2-on-databricks --profile <PROFILE>`.
2. Open `notebooks/generate` and attach it to **serverless GPU** compute: accelerator **A10**, environment version **4**.
3. Fill in the widgets and *Run all*.
   - `modalities`: the encoders to load ([Choosing the encoders](#choosing-the-encoders)).
   - `corpus`: one document per line (embedded with the `Document` prefix).
   - `media_dir`: a Volume folder; each image / audio / video file in it is one more document. Left empty, synthetic
     samples are added (a red and a blue circle, a 440 Hz tone, a red bar video).
   - `query` + `query_prompt`: ranked against all documents. `truncate_dim`: the Matryoshka size.
   - `output_dir`: a folder in your Volume.

   The first cell installs the runtime and restarts Python. The ranking is displayed, and the embeddings are written
   to `output_dir` as Parquet (`document`, `embedding`, `modality`, ...).

The same notebook runs unattended as a job on serverless GPU:

```bash
databricks jobs submit --no-wait --profile <PROFILE> --json '{
  "run_name": "embeddinggemma2-generate",
  "tasks": [{"task_key": "generate",
             "notebook_task": {"notebook_path": "/Workspace/Users/<you>/model-embeddinggemma-2-on-databricks/notebooks/generate",
                               "base_parameters": {"output_dir": "/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>/embeddinggemma2"}},
             "environment_key": "gpu", "compute": {"hardware_accelerator": "GPU_1xA10"}}],
  "environments": [{"environment_key": "gpu", "spec": {"client": "4"}}]}'
```

`environments[].spec.client: "4"` is required for serverless GPU.

## 2. Batch with the AI Runtime CLI

Set `DEST` in [`air_job/generate.yaml`](air_job/generate.yaml) and the `generate.py` flags in its `command`, then:

```bash
COPYFILE_DISABLE=1 air run --file air_job/generate.yaml --watch --profile <PROFILE>
```

The job:
1. runs `scripts/install.sh`;
2. runs [`generate.py`](generate.py), which embeds every line of the JSONL and writes one Parquet (`id`, `text`,
   `embedding`, `dim`, `modality`, `applied_params`, `error`), then prints the corpus ranked against `--query`;
3. copies the Parquet to the Volume.

One JSONL line is one row of the endpoint: `{"id": "...", "text": "...", "params": {...}}`, where `params` takes any
`params_json` key ([docs/SERVING-API.md](docs/SERVING-API.md)). Media on a Volume go in as paths, e.g.
`{"id": "img-1", "params": {"image_path": "/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>/photos/1.jpg"}}`.
`--modalities text` loads only the text model; `--params '{"truncate_dim": 256}'` applies to every line.

## 3. Model Serving

```bash
# 1) put the repo in the workspace (as in way 1), then register on serverless GPU A10.
#    "modalities": "all" (default), "text", "text+vision" or "text+audio"; give each its own uc_model.
databricks jobs submit --no-wait --profile <PROFILE> --json '{
  "run_name": "embeddinggemma2-register",
  "tasks": [{"task_key": "register",
             "notebook_task": {"notebook_path": "/Workspace/Users/<you>/model-embeddinggemma-2-on-databricks/notebooks/register_serving",
                               "base_parameters": {"modalities": "all", "uc_model": "<CATALOG>.<SCHEMA>.embeddinggemma2"}},
             "environment_key": "gpu", "compute": {"hardware_accelerator": "GPU_1xA10"}}],
  "environments": [{"environment_key": "gpu", "spec": {"client": "4"}}]}'
databricks jobs get-run <run_id> --profile <PROFILE>              # wait for TERMINATED/SUCCESS
databricks jobs get-run-output <task_run_id> --profile <PROFILE>  # notebook_output.result: {"uc_model", "modalities", "version", "smoke"}

# 2) endpoint (GPU_MEDIUM = A10, the same GPU family the environment was packed on)
databricks serving-endpoints create --no-wait --profile <PROFILE> --json '{
  "name": "<ENDPOINT>",
  "config": {"served_entities": [{"name": "embeddinggemma2-<VERSION>",
               "entity_name": "<CATALOG>.<SCHEMA>.embeddinggemma2", "entity_version": "<VERSION>",
               "workload_type": "GPU_MEDIUM", "workload_size": "Small", "scale_to_zero_enabled": true}],
             "traffic_config": {"routes": [{"served_model_name": "embeddinggemma2-<VERSION>", "traffic_percentage": 100}]}}}'
databricks serving-endpoints get <ENDPOINT> --profile <PROFILE> | jq '.state'   # READY / NOT_UPDATING
```

To move an existing endpoint to a new version, use `databricks serving-endpoints update-config` with the same
`served_entities`. Run `notebooks/test_serving_model` first after changing `serving/` or `embeddinggemma2_core.py`;
it loads the PyFunc in-process on a GPU without touching any endpoint.

**Calling the endpoint** ([docs/SERVING-API.md](docs/SERVING-API.md) is the full contract):

```bash
curl -sS -X POST "https://<workspace>/serving-endpoints/<ENDPOINT>/invocations" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"dataframe_records":[{"text":"What causes the northern lights?","params_json":"{\"prompt_name\":\"SearchQuery\"}"}]}'
# -> {"predictions": [{"embedding": [...768 floats], "dim": 768, "modality": "text", ..., "error": ""}]}
```

The first request after scale-to-zero takes minutes; `databricks serving-endpoints query` gives up after ~60 s, so
call the REST API with a 600–900 s timeout.

## Measured results

1× A10G, bf16, `modalities=all` (2026-10-15, workspace `fevm-lxejgv`):

| | |
|---|---|
| `notebooks/test_serving_model` (job) | all 8 input rows OK (text, 256-d, image, text + image, audio, video), bad rows reported; ≈ 7.9 min including the install. GPU memory peak 1.74 GiB |
| Registration job (env_pack) | ≈ 8.9 min. Smoke: query~doc 0.871, peak 1.41 GiB |
| Endpoint creation → READY | 394 s (`GPU_MEDIUM`, Small) |
| Endpoint, warm | 1 text row 0.24–0.32 s, 64 text rows 0.54 s, 1 image 0.32 s, 1 video (3 s) 0.41 s, round trip from a laptop in Japan |
| Endpoint, first call | 10 rows of every kind in 3.0 s |
| Retrieval sanity (endpoint) | "a red circle" → text "A red ball…" 0.814, text + red-circle image 0.818, red-circle image 0.757, red video 0.696, blue-circle image 0.684, tone 0.640, unrelated text 0.654 |
| bf16 vs fp32 | cosine 0.99998 on the same text (CPU) |

Cold start from scale-to-zero has not been measured yet. Locally (Apple-silicon CPU, fp32) the same checks pass, and
an image at 70 vs 280 vision tokens gives cosine 0.985.

### Why a custom PyFunc, not an MLflow flavor

- **`mlflow.transformers`:** its `feature-extraction` pipeline returns token states. The mean pooling, the 512→768
  projection, the normalisation, the prompts and the media inputs would all be left to the caller.
- **`mlflow.sentence_transformers`:** it logs this model fine (tried with MLflow 3.12; weights bundled, 548 MB text-only).
  But its input is strings only, so **no images, video or audio**. Its signature has no params, so `prompt_name` /
  `truncate_dim` cannot be passed, and one bad row fails the whole request. With `task="llm/v1/embeddings"`, the
  loaded pyfunc still had `task=None` locally and returned bare arrays.

The PyFunc keeps the multimodal input and the same `text` + `params_json` contract as the other
`model-*-on-databricks` repos.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `model type embedding_gemma2 ... not recognized` | transformers older than 5.19.0 (5.18.0 does not have it). It is pinned in `embeddinggemma2_core.RUNTIME_REQUIREMENTS`; rerun `scripts/install.sh`. |
| Several images or texts in one input come out in the wrong order | sentence-transformers older than 6.1.0 (the model's own requirement). Pinned. |
| NaN or nonsense embeddings | float16. The model's activations overflow it. Use bf16 (GPU default here) or fp32; `Embedder` refuses anything else. |
| `TypeError: open() got an unexpected keyword argument 'metadata_errors'` on video | transformers' path-based video loading fell back to torchvision, which breaks with newer PyAV. `Embedder` decodes video itself (`decode_video`), so this only happens if you pass a video path to sentence-transformers directly. |
| `... input needs an encoder this model was loaded without` | The endpoint / notebook loaded a smaller `modalities` set. Use `all` (or the matching set). |
| Scores between a text query and images look low | Expected: text documents score higher than media. Rank within one modality or calibrate per modality ([docs/SERVING-API.md](docs/SERVING-API.md) §4). |
| Query and corpus vectors cannot be compared | Different `truncate_dim`. Both sides must use the same size. |
| env_pack: `is_client_image=False` | Registration ran on classic compute. Use serverless GPU with `environments[].spec.client: "4"`. |
| `TimeoutError` uploading `model_environment.tar` | `databricks-sdk` older than 0.102.0. It is pinned. |
| `partially initialized module 'torchvision'` | The GPU image's torchvision is built for torch 2.7.1. `torchvision==0.25.0` (for torch 2.10) is pinned. |
| `ValueError` about `hf_transfer` on download | Serverless GPU presets `HF_HUB_ENABLE_HF_TRANSFER=1` without the package. `embeddinggemma2_core` sets it to `0`. |
| HTTP 400 / payload too large with video | Media travel as base64 in the request. Send one large file per request, or use way 1 / 2 with `video_path` on a Volume. |
| `serving-endpoints query` fails after ~60 s | Cold start. Retry later, or call REST with a long timeout. |
| A notebook's `print` is missing from `jobs get-run-output` | Serverless only returns `dbutils.notebook.exit`; the notebooks return their results that way. |

The checkpoint is pinned to `google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`
(`embeddinggemma2_core.HF_REVISION`).

## Licence

- **This repo's code:** Apache 2.0, like the model (see [NOTICE.md](NOTICE.md)).
- **EmbeddingGemma 2 weights:** [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license), © Google DeepMind. Not
  gated. Commercial use, modification and redistribution are allowed under Apache 2.0's conditions (keep the licence
  and notices). Not legal advice; read the terms.
- **The model card adds:** "Deployments must adhere to the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)."
- It is a pre-trained embedding model with no post-training alignment or output moderation; application-level
  safeguards (retrieval filtering, fairness testing) are the deployer's responsibility (model card).
- Check the [model card](https://huggingface.co/google/embeddinggemma-2) for current terms before use.
