# Databricks notebook source
# MAGIC %md
# MAGIC # EmbeddingGemma 2 — embed text, images, video and audio on serverless GPU, and search across them
# MAGIC
# MAGIC Attach this notebook to **serverless GPU compute, accelerator A10** (environment version 4). Open it from a
# MAGIC Git folder (or a synced copy) of this repo, so `../embeddinggemma2_core.py` and `../scripts/` are next to it.
# MAGIC
# MAGIC 1. Fill in the widgets.
# MAGIC    - `corpus` is one document per line.
# MAGIC    - `media_dir` is an optional Volume folder: every image / audio / video file in it becomes one more document.
# MAGIC      Left empty, a few synthetic samples are added: a red and a blue circle, a tone, a red bar video.
# MAGIC    - `query` is ranked against all of them in one shared vector space.
# MAGIC 2. Run all. The first run installs the runtime (torch 2.10/cu128, ~5 min) and downloads the 740M checkpoint.
# MAGIC
# MAGIC The embeddings are written to `output_dir` as Parquet. The request handling is the serving endpoint's
# MAGIC (`embeddinggemma2_core.Embedder`).

# COMMAND ----------

dbutils.widgets.dropdown("modalities", "all", ["all", "text", "text+vision", "text+audio"], "modalities (README: Choosing the encoders)")
dbutils.widgets.text("query", "a red circle", "query")
dbutils.widgets.dropdown("query_prompt", "SearchQuery",
                         ["SearchQuery", "QuestionAnswering", "FactChecking", "CodeRetrieval", "SentenceSimilarity", "none"],
                         "query_prompt")
dbutils.widgets.text("corpus", "\n".join([
    "A red ball on a white table.",
    "青い円が描かれた紙。",
    "The northern lights are caused by charged particles from the sun.",
    "今日の昼ごはんはカレーでした。",
    "A pure sine tone at 440 Hz, the A above middle C.",
]), "corpus (one document per line)")
dbutils.widgets.text("media_dir", "", "media_dir (optional Volume folder)")
dbutils.widgets.dropdown("truncate_dim", "768", ["768", "512", "256", "128"], "truncate_dim (Matryoshka)")
dbutils.widgets.text("output_dir", "/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>/embeddinggemma2", "output_dir (UC Volume)")

# COMMAND ----------

# Install the runtime into THIS notebook's Python (same recipe as the AIR job), then restart Python.
import os
import subprocess
import sys

REPO = os.path.abspath("..")
subprocess.run(["bash", os.path.join(REPO, "scripts", "install.sh")], check=True,
               env={**os.environ, "PYTHON": sys.executable})
dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.abspath("..")
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import embeddinggemma2_core as core  # noqa: E402
import sample_media  # noqa: E402

out_dir = dbutils.widgets.get("output_dir")
assert "<" not in out_dir, "set output_dir to a UC Volume folder, e.g. /Volumes/main/default/embeddings/embeddinggemma2"
DIM = int(dbutils.widgets.get("truncate_dim"))

t0 = time.time()
emb = core.Embedder(dbutils.widgets.get("modalities"))
core.log(f"{emb.modalities} loaded on {emb.device}/{emb.precision} in {time.time() - t0:.0f}s")

# COMMAND ----------

# The documents: the corpus lines (with the document prefix), then the media (no prefix: the model card says media take none).
EXT = {"image": (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"), "audio": (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
       "video": (".mp4", ".mov", ".webm", ".mkv", ".avi")}
docs = [(line, line, {"prompt_name": "Document"}) for line in dbutils.widgets.get("corpus").splitlines() if line.strip()]
media_dir = dbutils.widgets.get("media_dir").strip()
if media_dir:
    for name in sorted(os.listdir(media_dir)):
        kind = next((k for k, exts in EXT.items() if name.lower().endswith(exts)), None)
        if kind in emb.inputs:
            docs.append((f"[{kind}] {name}", "", {f"{kind}_path": os.path.join(media_dir, name)}))
else:
    samples = [("image", "red circle", sample_media.image_b64("red")), ("image", "blue circle", sample_media.image_b64("blue")),
               ("audio", "440 Hz tone", sample_media.audio_b64(440)), ("video", "red bar sweeping right", sample_media.video_b64((255, 0, 0)))]
    docs += [(f"[{k}] {label}", "", {f"{k}_b64": b}) for k, label, b in samples if k in emb.inputs]

rows = emb.handle_rows([(text, {**p, "truncate_dim": DIM}) for _, text, p in docs])
bad = [(d[0], r["error"]) for d, r in zip(docs, rows) if r["error"]]
assert not bad, bad

qp = dbutils.widgets.get("query_prompt")
q = emb.handle(dbutils.widgets.get("query"), {"truncate_dim": DIM, **({} if qp == "none" else {"prompt_name": qp})})
assert not q["error"], q["error"]
scores = np.stack([r["embedding"] for r in rows]) @ np.array(q["embedding"])
ranked = pd.DataFrame({"score": scores.round(4), "modality": [r["modality"] for r in rows],
                       "document": [d[0] for d in docs]}).sort_values("score", ascending=False)
display(ranked)

# COMMAND ----------

os.makedirs(out_dir, exist_ok=True)
dest = os.path.join(out_dir, f"{int(time.time())}.parquet")
pd.DataFrame([{"document": d[0], **r} for d, r in zip(docs, rows)]).to_parquet(dest, index=False)
print("wrote", dest)

# COMMAND ----------

import torch  # noqa: E402
dbutils.notebook.exit(json.dumps({"file": dest, "documents": len(docs), "dim": DIM, "top": ranked.iloc[0]["document"],
                                  "embed_seconds": rows[0]["embed_seconds"],
                                  "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}, ensure_ascii=False))
